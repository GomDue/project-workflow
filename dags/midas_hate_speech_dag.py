from pathlib import Path
from datetime import datetime

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator 

ROOT_DIR = Path(__file__).resolve().parents[1]
NOW_TIME = datetime.now().strftime('%y%m%d')

@dag(
    dag_id='midas_hate_speech_dag', 
    schedule_interval=None
)
def midas_hate_speech_dag():
    @task(task_id="train_kcbert_model")
    def train_kcbert_model():
        from models.kcbert.model import KcbertModel

        model = KcbertModel(Path(ROOT_DIR)/"params.yaml")
        model.train()

    save_model_to_s3 = LocalFilesystemToS3Operator(
        task_id="save_model_to_s3",
        filename=f"./data/model/kcbert/state_dict/{NOW_TIME}.pt",
        dest_key=f"s3://midas-bucket-1/models/kcbert/model/{NOW_TIME}",
        aws_conn_id="aws_default",
        replace=True
    )
    
    # Task sequence
    train_kcbert_model() >> save_model_to_s3


midas_hate_speech_dag()