from pathlib import Path
from datetime import datetime

import pandas as pd

from hooks.aws_rds_hook import AWSRDSHook

from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator

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
        hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM comment")
        result = cursor.fetchone()[0]

        return "process_init_dataset" if not result else "skip_init_dataset"

    @task
    def process_init_dataset(**context):
        import numpy as np

        # https://github.com/smilegate-ai/korean_unsmile_dataset/tree/main
        origin_train_dataset = pd.read_csv(Path(ROOT_DIR)/"data"/"hatespeech"/"unsmile_train_v1.0.tsv", sep="\t")
        origin_valid_dataset = pd.read_csv(Path(ROOT_DIR)/"data"/"hatespeech"/"unsmile_valid_v1.0.tsv", sep="\t")

        origin_dataset_df = pd.concat([origin_train_dataset, origin_valid_dataset])
        origin_dataset_df.rename(columns={"문장": "content"}, inplace=True)
        
        columns_to_combine = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오", "악플/욕설", "clean", "개인지칭"]
        origin_dataset_df["origin_type"] = origin_dataset_df[columns_to_combine].apply(lambda row: ", ".join(row.index[row==1]), axis=1)
        origin_dataset_df["created_date"] = "2024-06-01 00:00:00"
        origin_dataset_df["type"] = None
        origin_dataset_df.drop(columns=columns_to_combine, inplace=True)

        context["ti"].xcom_push(key="origin_dataset_json", value=origin_dataset_df.to_json(orient="records"))


    @task
    def insert_init_dataset(**context):
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
            type_value = None if pd.isna(row['type']) else row['type']

            cursor.execute(insert_sql, (row["content"], row["created_date"], type_value, row["origin_type"]))

        conn.commit()
        cursor.close()
        conn.close()

    
    @task
    def get_last_fetch_time_from_postgres(**context):
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
        def extract_content(content):
            '''
            format : 
                recycle_board : content
                recycle_comment : {"comment": "content"}
                question_board : <p> content </p>
                question_comment : {"comment": "content"}
            to :
                content
            '''
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
        comment_df = pd.DataFrame()

        for table in tables:
            comment_json = context["ti"].xcom_pull(key=f"{table}_comment_json")
            
            if comment_json:
                new_df = pd.read_json(comment_json)
                comment_df = pd.concat([comment_df, new_df], ignore_index=True)

        print(comment_df)
        

    @task
    def train_kcbert_model():
        from models.kcbert.model import KcbertModel

        model = KcbertModel(Path(ROOT_DIR)/"params.yaml")
        model.train()

    # save_model_to_s3 = LocalFilesystemToS3Operator(
    #     task_id="save_model_to_s3",
    #     filename=Path(ROOT_DIR)/"data"/"model"/"kcbert"/"state_dict"/f"{NOW_TIME}.pt",
    #     dest_key=f"s3://midas-bucket-1/models/kcbert/model/{NOW_TIME}",
    #     aws_conn_id="aws_default",
    #     replace=True
    # )



    
    check_init_dataset = BranchPythonOperator(
        task_id="check_init_dataset",
        python_callable=check_init_dataset
    )
    process_init_dataset = process_init_dataset()
    insert_init_dataset = insert_init_dataset()
    skip_init_dataset = DummyOperator(task_id="skip_init_dataset")
    get_last_fetch_time_from_postgres = get_last_fetch_time_from_postgres()
    


    '''
    Task sequence
    '''
    fetch_tasks, process_tasks = [], []
    for table in tables:
        fetch_data = fetch_data_by_table_from_aws_rds.override(task_id=f"fetch_{table}_content_from_aws_rds")(table=table)
        process_data = process_content_data_by_table.override(task_id=f"process_{table}_content")(table=table)
        
        fetch_data >> process_data
        
        fetch_tasks.append(fetch_data)
        process_tasks.append(process_data)


    check_init_dataset >> [process_init_dataset, skip_init_dataset]
    process_init_dataset >> insert_init_dataset
    skip_init_dataset >> get_last_fetch_time_from_postgres >> fetch_tasks
    process_tasks >> insert_data_to_postgres()

    # train_kcbert_model() >> save_model_to_s3


midas_hate_speech_dag()