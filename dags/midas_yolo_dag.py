from pathlib import Path
from datetime import datetime

from airflow.decorators import dag, task

NOW_TIME = datetime.now().strftime('%y%m%d')
ROOT_DIR = Path(__file__).resolve().parents[1]


@dag(
    dag_id='midas_yolo_dag', 
    schedule_interval=None
)
def midas_yolo_dag():

    @task(task_id="process_yolo_data")
    def process_yolo_data():
        from models.yolo.dataset import dataset

        yolo_dataset = dataset()
        yolo_dataset.preprocess()


    @task(task_id="train_yolo_model")
    def train_yolo_model():
        from models.yolo.model import YoloModel

        model = YoloModel(Path(ROOT_DIR)/"recycle.yaml", Path(ROOT_DIR)/"params.yaml")
        model.train()

    # Task sequence
    process_yolo_data() >> train_yolo_model()

midas_yolo_dag()