from pathlib import Path
from datetime import datetime

import pandas as pd

from hooks.aws_rds_hook import AWSRDSHook

from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator 

ROOT_DIR = Path(__file__).resolve().parents[1]
NOW_TIME = datetime.now().strftime("%y%m%d")

@dag(
    dag_id="midas_hate_speech_dag", 
    schedule_interval="@daily",
    start_date=datetime(2024, 10, 1),
    catchup=False,
)
def midas_hate_speech_dag():
    tables = ["recycle_board", "recycle_comment", "question_board", "question_comment"]

    def check_init_dataset():
        """
        PostgreSQL의 comment 테이블에 초기 데이터가 존재하는지 확인합니다.

        Returns:
            str: 데이터가 없으면 'process_init_dataset', 있으면 'skip_init_dataset' 반환
        """
        hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM comment")
        result = cursor.fetchone()[0]

        return "process_init_dataset" if not result else "skip_init_dataset"


    @task
    def process_init_dataset(**context):
        """
        unsmile dataset을 로드하여 혐오 표현 라벨을 통합하고 
        comment 테이블에 삽입하기 위한 형태로 전처리합니다.

        XCom을 통해 전처리된 JSON 데이터를 push합니다.
        """
        def check_label(label):
            if label["여성/가족"] == 1:     return 0
            elif label["남성"] == 1:        return 1
            elif label["성소수자"] == 1:    return 2
            elif label["인종/국적"] == 1:   return 3
            elif label["연령"] == 1:        return 4
            elif label["지역"] == 1:        return 5
            elif label["종교"] == 1:        return 6
            elif label["기타 혐오"] == 1:   return 7
            elif label["악플/욕설"] == 1:   return 8
            elif label["개인지칭"] == 1:    return 9
            else:                          return 10

        # Load dataset
        origin_train_dataset = pd.read_csv(Path(ROOT_DIR)/"data"/"hatespeech"/"unsmile_train_v1.0.tsv", sep="\t")
        origin_valid_dataset = pd.read_csv(Path(ROOT_DIR)/"data"/"hatespeech"/"unsmile_valid_v1.0.tsv", sep="\t")
        origin_dataset_df = pd.concat([origin_train_dataset, origin_valid_dataset])
        origin_dataset_df.rename(columns={"문장": "content"}, inplace=True)

        # Process dataset
        origin_dataset_df["created_date"] = "2024-06-01 00:00:00"
        origin_dataset_df["type"] = origin_dataset_df.apply(check_label, axis=1)
        columns_to_combine = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오", "악플/욕설", "clean", "개인지칭"]
        origin_dataset_df["origin_type"] = origin_dataset_df[columns_to_combine].apply(lambda row: ", ".join(row.index[row==1]), axis=1)
        origin_dataset_df.drop(columns=columns_to_combine, inplace=True)

        context["ti"].xcom_push(key="origin_dataset_json", value=origin_dataset_df.to_json(orient="records"))


    @task
    def insert_init_dataset(**context):
        """
        XCom으로 전달받은 혐오 표현 초기 데이터를 PostgreSQL comment 테이블에 삽입합니다.
        
        Args:
            context (dict): Airflow 컨텍스트 객체
        """
        origin_dataset_json = context["ti"].xcom_pull(key="origin_dataset_json")
        origin_dataset_df = pd.read_json(origin_dataset_json)

        hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = hook.get_conn()
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO comment (content, created_date, type, origin_type)
            VALUES (%s, %s, %s, %s)
        """

        for _, row in origin_dataset_df.iterrows():
            type_value = None if pd.isna(row["type"]) else row["type"]
            cursor.execute(insert_sql, (row["content"], row["created_date"], type_value, row["origin_type"]))

        conn.commit()
        cursor.close()
        conn.close()


    
    @task
    def get_last_fetch_time_from_postgres(**context):
        """
        PostgreSQL의 comment 테이블에서 가장 최근 created_date를 조회하여,
        이후 삽입된 데이터만 필터링할 수 있도록 기준 시간을 XCom으로 전달합니다.

        XCom:
            key: "last_fetch_time", value: 가장 최근 댓글 생성 시간 (str)
        """
        hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = hook.get_conn()
        
        row = pd.read_sql('''
            SELECT MAX(created_date) AS created_date 
            FROM comment
        ''', con=conn)

        last_fetch_time = row["created_date"].iloc[0]
        context["ti"].xcom_push(key="last_fetch_time", value=str(last_fetch_time))


    @task
    def fetch_data_by_table_from_aws_rds(table, **context):
        """
        AWS RDS로부터 지정된 테이블에서 최근 수집 시점 이후의 데이터를 조회합니다.

        Args:
            table (str): 조회할 테이블 이름 (recycle_board 등)

        XCom:
            key: "{table}_content_json", value: content와 날짜 정보를 포함한 JSON 문자열
        """
        last_fetch_time = context["ti"].xcom_pull(key="last_fetch_time")

        hook = AWSRDSHook("aws_default")
        conn, _ = hook.get_conn()
        
        content_df = pd.read_sql(f'''
            SELECT content, created_date AS date
            FROM {table}
            WHERE content IS NOT NULL AND created_date > '{last_fetch_time}'
        ''', con=conn)

        context["ti"].xcom_push(key=f"{table}_content_json", value=content_df.to_json(orient="records"))


    @task
    def process_content_data_by_table(table, **context):
        """
        테이블별 content 데이터에서 HTML, JSON 문자열 등을 정제하여 순수 텍스트 형태로 변환합니다.

        Args:
            table (str): 대상 테이블 이름

        XCom:
            key: "{table}_comment_json", value: 정제된 댓글 데이터 JSON 문자열
        """
        def extract_content(content):
            """
            다양한 content 포맷을 텍스트로 추출하는 함수
            - JSON 형식 ({"comment": "text"}) → text
            - HTML <p>태그 → 제거
            """
            if content[0] == "{":
                content = content.split('"')[3]
            elif content.startswith("<p>"):
                content = content.replace("<p>", "").replace("</p>", "")
            return content

        comment_json = context["ti"].xcom_pull(key=f"{table}_content_json")
        comment_df = pd.read_json(comment_json)

        comment_df["content"] = comment_df["content"].apply(lambda content: extract_content(content))
        
        context["ti"].xcom_push(key=f"{table}_comment_json", value=comment_df.to_json(orient="records"))


    @task
    def insert_data_to_postgres(**context):
        """
        여러 테이블에서 정제된 댓글 데이터를 PostgreSQL에 삽입합니다.
        단, 실제 INSERT SQL은 생략되어 있고 현재는 수집 결과를 출력만 합니다.
        """
        comment_df = pd.DataFrame()

        for table in tables:
            comment_json = context["ti"].xcom_pull(key=f"{table}_comment_json")
            if comment_json:
                new_df = pd.read_json(comment_json)
                comment_df = pd.concat([comment_df, new_df], ignore_index=True)

        print(comment_df)  # 실제 삽입은 추후 구현 필요

        
    """
    1. 데이터가 존재하는지 확인
    - 없다면: 초기 데이터 처리 → 삽입
    - 있다면: RDS에서 신규 댓글 데이터 수집 → 전처리 → PostgreSQL 저장

    2. 수집/삽입 후, midas_training_dag을 트리거하여 KoBERT 모델 재학습 실행
    """

    check_init_dataset = BranchPythonOperator(
        task_id="check_init_dataset",
        python_callable=check_init_dataset
    )

    process_init_dataset = process_init_dataset()
    insert_init_dataset = insert_init_dataset()
    skip_init_dataset = DummyOperator(task_id="skip_init_dataset")
    get_last_fetch_time_from_postgres = get_last_fetch_time_from_postgres()
    insert_data_to_postgres = insert_data_to_postgres()

    trigger_train_model = TriggerDagRunOperator(
        task_id="trigger_train_model",
        trigger_dag_id="midas_training_dag", 
        wait_for_completion=False,
    )

    # 테이블 순회하며 fetch & process task 구성
    fetch_tasks, process_tasks = [], []
    for table in tables:
        fetch_data = fetch_data_by_table_from_aws_rds.override(task_id=f"fetch_{table}_content_from_aws_rds")(table=table)
        process_data = process_content_data_by_table.override(task_id=f"process_{table}_content")(table=table)
        
        fetch_data >> process_data
        fetch_tasks.append(fetch_data)
        process_tasks.append(process_data)

    # 전체 Task 흐름 연결
    check_init_dataset >> [process_init_dataset, skip_init_dataset]
    process_init_dataset >> insert_init_dataset
    skip_init_dataset >> get_last_fetch_time_from_postgres >> fetch_tasks
    process_tasks >> insert_data_to_postgres >> trigger_train_model
 


@dag(
    dag_id="midas_training_dag", 
    schedule_interval=None,
    max_active_tasks=1,
    catchup=False
)
def midas_training_dag():
    """
    혐오 발언 감지용 KoBERT 모델을 학습하고, 학습된 모델을 S3에 저장하는 DAG입니다.
    다른 DAG에서 TriggerDagRunOperator를 통해 실행됩니다.
    """

    @task
    def train_kcbert_model():
        """
        학습 파라미터가 정의된 YAML 파일을 기반으로 KoBERT 모델을 학습합니다.
        학습은 내부적으로 PyTorch + Transformers 기반으로 이루어지며,
        TensorBoard 로깅, F1 스코어 및 PR Curve 로그도 포함됩니다.
        """
        from models.kcbert.model import KcbertModel

        model = KcbertModel(Path(ROOT_DIR)/"params.yaml")
        model.train()

    # 모델 학습 후 학습된 state_dict를 S3에 업로드
    save_model_to_s3 = LocalFilesystemToS3Operator(
        task_id="save_model_to_s3",
        filename=Path(ROOT_DIR)/"data"/"model"/"kcbert"/"state_dict"/f"{NOW_TIME}.pt",
        dest_key=f"s3://midas-bucket-1/models/kcbert/model/{NOW_TIME}",
        aws_conn_id="aws_default",
        replace=True,
        retries=3
    )

    # Task 연결
    train_kcbert_model() >> save_model_to_s3


midas_hate_speech_dag()
midas_training_dag()