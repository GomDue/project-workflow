# Temporary code
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import CONFIG

GCP_CONFIG = CONFIG["GCPConfig"]
AFL_CONFIG = CONFIG["AirflowConfig"]
SPK_CONFIG = CONFIG["PysparkConfig"]
RCS_CONFIG = CONFIG["RecycleSolutionConfig"]


from datetime import datetime

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from operators.hatespeechOperator import HateSpeechOperator
from operators.recycleSolutionOperator import RecycleSolutionOperator


# load images
def _connect_aws_s3():
    import os
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


def _load_kcbert_model(**context):
    pass


def _train_kcbert_model(dataset_path, learning_rate, epochs, **context):
    import mlflow
    import pandas as pd

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset

    # 1. Load and preprocess dataset
    dataset = pd.read_csv(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True)

    # Convert DataFrame to Huggingface Dataset
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")

    # Split dataset into training and validation
    train_test_split = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test_split['train']
    valid_dataset = train_test_split['test']

    # 2. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/kcbert-base", num_labels=11
    )

    # 3. Set up Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="mlflow"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    # 4. Train and log with MLflow
    with mlflow.start_run():
        trainer.train()

        # Log model and parameters with MLflow
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
        })

        # Log metrics (evaluation)
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics(eval_metrics)

        # Log the trained model with MLflow
        signature = mlflow.models.infer_signature(
            train_dataset[:5], 
            trainer.predict(valid_dataset).predictions[:5]
        )

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="hatespeech_classifier",
            task="text-classification",
            signature=signature,
            input_example={"text": "This is a sample input text."}
        )


def _save_model():
    pass


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
        gcp_conn_id=GCP_CONFIG.GOOGLE_CONN,
        gsheet_id=GCP_CONFIG.GOOGLE_SHEET_ID,
        range=GCP_CONFIG.GOOGLE_SHEET_NAME,
        file_name=RCS_CONFIG.RECYCLE_SOLUTION_FILE_NAME,
        destination=RCS_CONFIG.RECYCLE_SOLUTION_DIR,
    )

    process_recycle_solution = SparkSubmitOperator(
        task_id='process_recycle_solution',
        application=SPK_CONFIG.PYSPARK_APP__RECYCLE_SOLUTION_PATH,
        conn_id=SPK_CONFIG.PYSPARK_CONN,
        conf=SPK_CONFIG.PYSPARK_CONF,
    )

    ready_connection = DummyOperator(
        task_id='data_preparation',
    )


    process_dataset = DummyOperator(
        task_id="process_dataset",
        # python_callable=_process_dataset
    )

    load_kcbert_model = DummyOperator(
        task_id="load_kcbert_model",
    )

    train_kcbert_model = PythonOperator(
        task_id="train_kcbert_model",
        python_callable=_train_kcbert_model,
        op_kwargs={
            "dataset_path": "/home/user/airflow/data/hatespeech/hatespeech.csv",
            "learning_rate": 1e-5,
            "epochs": 5,
        }
    )

    save_model = DummyOperator(
        task_id="save_model",
    )

    # Task sequence
    download_gdrive_file >> process_recycle_solution
    connect_aws_s3 >> ready_connection
    connect_aws_rds >> ready_connection
    process_dataset >> load_kcbert_model >> train_kcbert_model >> save_model
    