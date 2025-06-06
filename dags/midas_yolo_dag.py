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

    @task
    def update_latest_model_yaml():
        """
        latest_model.yaml을 S3에서 불러와 해당 model_type 정보만 업데이트 후 다시 업로드합니다.
        """
        import os
        import yaml
        import boto3

        model_type = "yolo"
        model_name = f"{model_type}_{NOW_TIME}.pt"
        s3_model_path = f"models/{model_type}/model/{model_name}"   

        # AWS S3 연결
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        s3 = session.resource("s3")
        bucket = s3.Bucket(os.getenv("AWS_S3_BUCKET_NAME"))
        key = "models/latest_model.yaml"
        local_path = Path(ROOT_DIR) / "latest_model.yaml"

        # 기존 latest_model.yaml 다운로드 시도
        try:
            bucket.download_file(key, str(local_path))
            with open(local_path, "r") as f:
                latest_model = yaml.safe_load(f)
            if latest_model is None:
                latest_model = {"models": {}}
        except Exception:
            latest_model = {"models": {}}

        # 해당 model_type만 업데이트
        latest_model["models"][model_type] = {
            "name": model_name,
            "path": s3_model_path
        }

        # 다시 파일로 저장하고 업로드
        with open(local_path, "w") as f:
            yaml.dump(latest_model, f)

        bucket.upload_file(str(local_path), key)
        print(f"Updated {model_type} in latest_model.yaml and uploaded to s3://{bucket.name}/{key}")


    train_yolo_model() >> update_latest_model_yaml()


midas_yolo_dag()
