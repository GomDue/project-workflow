from pathlib import Path
from datetime import datetime

from airflow.decorators import dag
from airflow.models import Variable

from operators.process import GoogleSheetToCSVOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

ROOT_DIR = Path(__file__).resolve().parents[1]
NOW_TIME = datetime.now().strftime("%y%m%d")


@dag(
    dag_id="midas_dag", 
    schedule_interval="@daily",
    start_date=datetime(2024, 10, 1),
    catchup=False,
)
def midas_dag():
    """
    매일 Google Sheet에서 데이터를 다운로드하고 Spark로 전처리하는 DAG.

    Workflow:
    1. Google Sheet에서 분리배출 솔루션 데이터를 CSV로 저장
    2. Spark 애플리케이션을 통해 해당 데이터를 전처리
    """

    download_gdrive_file = GoogleSheetToCSVOperator(
        task_id="download_gdrive_file",
        gcp_conn_id="google_cloud_default",
        gsheet_id=Variable.get("gcp_sheet_id"),
        range=Variable.get("gcp_sheet_name"),
        file_name=f"recycle_solution_{NOW_TIME}.csv",
        save_path=Path(ROOT_DIR)/"data"/"solutions",
    )
    """
    Google Sheet 데이터를 로컬 CSV 파일로 저장하는 Task.

    - Google Drive 연동을 통해 실시간 데이터를 수집
    - 파일명은 실행 날짜 기준으로 동적으로 생성됨
    """

    process_recycle_solution = SparkSubmitOperator(
        task_id='process_recycle_solution',
        application=str(Path(ROOT_DIR)/"spark"/"app"/"process_recycle_solution.py"),
        conn_id="spark_default",
        conf={
            "spark.master":"spark://spark:7077"
        },
    )
    """
    다운로드된 CSV 파일을 Spark로 전처리하는 Task.

    - SparkSubmitOperator를 통해 외부 Python 스크립트를 실행
    - Spark 클러스터 환경에서 실행되며, 데이터 변환/정제 수행
    """

    download_gdrive_file >> process_recycle_solution


midas_dag()
