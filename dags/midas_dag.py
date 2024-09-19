# Temporary code
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]

import os
from datetime import datetime

NOW_TIME = datetime.now().strftime('%y%m%d')

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from operators.hatespeechOperator import HateSpeechOperator
from operators.recycleSolutionOperator import RecycleSolutionOperator
from operators.yoloOperator import YOLOOperator


# load images
def _connect_aws_s3():
    from airflow.hooks.S3_hook import S3Hook

    s3_hook = S3Hook(aws_conn_id='aws_s3')
    bucket = 'midas-bucket-1'

    output_path = os.path.expanduser('~/data/test')
    os.makedirs(output_path, exist_ok=True)

    for image in s3_hook.list_keys(bucket_name=bucket):
        print(image)
        # if not os.path.exists(os.path.join(output_path, image)) and '/' not in image:
        #     download_image_path = s3_hook.download_file(key=image, bucket_name=bucket, local_path=output_path)
        #     new_image_path = os.path.join('/'.join(download_image_path.split('/')[:-1]), image)

        #     os.rename(src=download_image_path, dst=new_image_path)

def _ready_connection():
    pass



def _process_dataset():
    pass


def _train_kcbert_model(kcbert_param, **context):
    from models.KcBert.model import CustomeKcBertModel

    model = CustomeKcBertModel(kcbert_param)

    model.train()



    # from datasets import Dataset
    # from transformers import (
    #     AutoTokenizer, 
    #     AutoModelForSequenceClassification, 
    #     Trainer, 
    #     TrainingArguments,
    #     logging
    # )

    # logging.set_verbosity_warning()

    # # load the configuration file 
    # with open(kcbert_param) as f:
    #     params = yaml.safe_load(f)["kcbert"]

    # # Load and preprocess dataset
    # dataset = pd.read_csv(dataset_path)
    # tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")

    # # Tokenize the dataset
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], padding=True, truncation=True)

    # # Convert DataFrame to Huggingface Dataset
    # dataset = Dataset.from_pandas(dataset)
    # dataset = dataset.map(tokenize_function, batched=True)
    # dataset = dataset.rename_column("label", "labels")

    # # Split dataset into training and validation
    # train_test_split = dataset.train_test_split(test_size=0.15, seed=42)
    # train_dataset = train_test_split['train']
    # valid_dataset = train_test_split['test']

    # # Load model
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "beomi/kcbert-base", num_labels=11
    # )

    # # Set up Trainer
    # training_args = TrainingArguments(
    #     output_dir=os.path.join(ROOT_DIR, "data", "model", "KcBert"),
    #     evaluation_strategy=params["evaluation_strategy"],
    #     eval_steps=params["eval_steps"],
    #     learning_rate=float(params["learning_rate"]),
    #     per_device_train_batch_size=params["per_device_train_batch_size"],
    #     per_device_eval_batch_size=params["per_device_eval_batch_size"],
    #     num_train_epochs=params["num_train_epochs"],
    #     weight_decay=params["weight_decay"],
    #     logging_strategy=params["logging_strategy"],
    #     logging_steps=params["logging_steps"],
    #     logging_dir=params["logging_dir"],
    #     report_to=params["report_to"]
    # )

    # def compute_metrics(pred):
    #     from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    #     labels = pred.label_ids
    #     preds = pred.predictions.argmax(-1)

    #     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    #     acc = accuracy_score(labels, preds)

    #     return {
    #         'accuracy': acc,
    #         'f1': f1,
    #         'precision': precision,
    #         'recall': recall,
    #     }


    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )
    
    # trainer.train()

    

    


        
        

def _process_yolo_data(**context):
    from models.YOLO.dataset import dataset

    yolo_dataset = dataset()
    yolo_dataset.preprocess()


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 8, 29),
}
with DAG(
    dag_id='midas_dag', 
    default_args=default_args, 
    schedule_interval=None
) as dag:
    connect_aws_s3 = PythonOperator(
        task_id='connect_aws_s3',
        python_callable=_connect_aws_s3,
        provide_context=True
    )

    connect_aws_rds = HateSpeechOperator(
        task_id='connect_aws_rds',
        conn_id='aws_rds_operator_test'
    )

    download_gdrive_file = RecycleSolutionOperator(
        task_id="download_gdrive_file",
        gcp_conn_id="google_cloud_default",
        gsheet_id=Variable.get("gcp_sheet_id"),
        range=Variable.get("gcp_sheet_name"),
        file_name=f"recycle_solution_{NOW_TIME}.csv",
        save_path=Path(ROOT_DIR)/"data"/"solutions",
    )

    process_recycle_solution = SparkSubmitOperator(
        task_id='process_recycle_solution',
        application=str(Path(ROOT_DIR)/"spark"/"app"/"process_recycle_solution.py"),
        conn_id="spark_default",
        conf={
            "spark.master":"spark://spark:7077"
        },
    )

    ready_connection = DummyOperator(
        task_id='data_preparation',
    )


    process_hatespeech_dataset = DummyOperator(
        task_id="process_dataset",
        # python_callable=_process_dataset
    )

    train_kcbert_model = PythonOperator(
        task_id="train_kcbert_model",
        python_callable=_train_kcbert_model,
        op_kwargs={
            "kcbert_param": Path(ROOT_DIR)/"params.yaml",
        }
    )


    # process_yolo_data = PythonOperator(
    #     task_id="process_yolo_data",
    #     python_callable=_process_yolo_data,
    # )

    # train_yolo_model = YOLOOperator(
    #     task_id="train_yolo_model",
    #     conn_id="yolo_default",
    #     yolo_class=Path(ROOT_DIR)/"recycle.yaml",
    #     yolo_param=Path(ROOT_DIR)/"params.yaml"
    # )



    # Task sequence
    download_gdrive_file >> process_recycle_solution
    process_hatespeech_dataset >> train_kcbert_model
    # process_yolo_data
    # train_yolo_model

    connect_aws_s3 >> ready_connection
    connect_aws_rds >> ready_connection