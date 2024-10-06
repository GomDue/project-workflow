from pathlib import Path
from datetime import datetime

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.sensors.sqs import SqsSensor
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator 
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

NOW_TIME = datetime.now().strftime('%y%m%d')
ROOT_DIR = Path(__file__).resolve().parents[1]


@dag(
    dag_id='midas_yolo_dag', 
    schedule_interval=None
)
def midas_yolo_dag():

    # wait_for_image_in_s3 = SqsSensor(
    #     task_id="wait_for_image_in_s3",
    #     sqs_queue="https://sqs.ap-northeast-2.amazonaws.com/975050037075/midas-s3-airflow",
    #     aws_conn_id="aws_default",
    #     max_messages=10,
    #     num_batches=3,
    # )

    @task(task_id="get_s3_images_by_month")
    def get_s3_images_by_month(**context):
        import pytz
        from datetime import datetime, timedelta
        
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook

        s3_hook = S3Hook(aws_conn_id="aws_default")

        kst = pytz.timezone("Asia/Seoul")

        now = datetime.now().astimezone(kst).replace(hour=0, minute=0, second=0, microsecond=0)
        from_datetime = now.replace(day=1)
        to_datetime = (from_datetime + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)

        images = [
            image for image in s3_hook.list_keys(
                bucket_name="midas-bucket-1", 
                prefix="midas-service/keywords/",
                from_datetime=from_datetime,
                to_datetime=to_datetime
            )
        ]

        context["ti"].xcom_push(key="s3_images", value=images)


    @task(task_id="load_")
    def check_sqs(**context):
        images = context["ti"].xcom_pull(key="s3_images", task_ids="get_s3_images_by_month")

        print(images)


    @task(task_id="get_verified_image")
    def get_verified_image(**context):
        import pandas as pd

        from hooks.aws_rds_hook import AWSRDSHook

        hook = AWSRDSHook("aws_default")
        conn, _ = hook.get_conn()

        verified_images = pd.read_sql('''
            SELECT w.id, w.image_url, c.name AS category 
            FROM waste w 
            INNER JOIN category c ON w.id = c.waste_id 
            WHERE w.image_url IS NOT NULL
        ''', con=conn)

        verified_images.to_csv(Path(ROOT_DIR)/"data"/"images"/f"images_{NOW_TIME}.csv", index=False)


    @task(task_id="change_category_to_yolo_format")
    def change_category_to_yolo_format():
        import yaml
        import pandas as pd

        with open(Path(ROOT_DIR)/"recycle.yaml") as f:
            image_classes = yaml.safe_load(f)["names"]

        print(image_classes)

        verified_images = pd.read_csv(Path(ROOT_DIR)/"data"/"images"/f"images_{NOW_TIME}.csv")
        print(verified_images)



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

    # save_model_to_s3 = LocalFilesystemToS3Operator(
    #     task_id="save_model_to_s3",
    #     filename=f"./data/model/yolo/state_dict/{NOW_TIME}.pt",
    #     dest_key=f"s3://midas-bucket-1/models/yolo/model/{NOW_TIME}",
    #     aws_conn_id="aws_default",
    #     replace=True
    # )

    # Task sequence
    # process_yolo_data() >> train_yolo_model() >> save_model_to_s3
    # wait_for_image_in_s3 >> check_sqs()
    # get_verified_image() >> change_category_to_yolo_format()
    train_yolo_model()

midas_yolo_dag()
