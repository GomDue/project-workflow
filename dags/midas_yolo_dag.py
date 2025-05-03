from pathlib import Path
from datetime import datetime

from airflow.decorators import dag, task

NOW_TIME = datetime.now().strftime('%y%m%d')
ROOT_DIR = Path(__file__).resolve().parents[1]


@dag(
    dag_id='midas_yolo_dag', 
    schedule_interval=None,
    description="YOLO 모델 학습을 수행하는 DAG"
)
def midas_yolo_dag():
    """
    객체 인식 모델인 YOLO의 학습을 수행하는 DAG입니다.

    - `recycle.yaml`: 클래스 정보 정의 파일
    - `params.yaml`: 학습 하이퍼파라미터 설정 파일
    - 학습된 모델은 내부적으로 저장되며 추후 배포 및 예측에 활용됩니다.
    """

    @task(task_id="train_yolo_model")
    def train_yolo_model():
        """
        YOLO 모델 학습을 수행하는 Task입니다.

        지정된 설정 파일을 기반으로 모델을 초기화하고 학습을 진행합니다.
        학습 결과는 설정된 디렉토리에 저장됩니다.
        """
        from models.yolo.model import YoloModel

        model = YoloModel(Path(ROOT_DIR)/"recycle.yaml", Path(ROOT_DIR)/"params.yaml")
        model.train()

    train_yolo_model()


midas_yolo_dag()
