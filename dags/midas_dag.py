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

from models.KcbertModel import KcbertModel
from model_train import train_model

import torch
from torch.utils.data import Dataset, DataLoader

import mlflow
from mlflow.models import infer_signature


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


def _load_kcbert_model(base_model_name, model_param_save_path, **context):
    model = KcbertModel(
        base_model_name=base_model_name,
        model_param_save_path=model_param_save_path
    )

    context["ti"].xcom_push(key="base_model_name", value=base_model_name)
    context["ti"].xcom_push(key="model_param_save_path", value=model_param_save_path)


def _train_kcbert_model(learning_rate, epochs, **context):
    model = KcbertModel(
        base_model_name=context["ti"].xcom_pull(key="base_model_name"),
        model_param_save_path=context["ti"].xcom_pull(key="model_param_save_path")
    )

    print("Success load model")
    # Temporary code
    if model: return

    model, tokenizer = model.getModelAndTokenizer()
    device = model.getDevice()
    dataset = context["dataset_path"]

    # Separate train, validation data
    from sklearn.model_selection import train_test_split
    train, valid = train_test_split(dataset, test_size=0.15, random_state=42)

    # Log the model with MLflow
    with mlflow.start_run():
        batch_size = 11
        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True)

        # Set hyperparameter 
        epochs = context["epoch"]
        learning_rate = context["learning_rate"]
        train_accuracies, train_losses = [], []
        valid_accuracies, valid_losses = [], []
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            batch_train_loss, batch_valid_loss = [], []
            total = correct = 0

            model.train()
            for i, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_preds = logits.argmax(dim=1) 
                correct += (train_preds == labels).sum().item()
                total += labels.size(0)

                batch_train_loss.append(loss.item())


            model.eval()
            total = correct = 0
            with torch.no_grad():
                for val_batch in valid_dataloader:
                    val_input_ids = val_batch['input_ids']
                    val_attention_mask = val_batch['attention_mask']
                    val_labels = val_batch['label']

                    val_input_ids = val_input_ids.to(device)
                    val_attention_mask = val_attention_mask.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                    val_logits = val_outputs.logits

                    val_loss = criterion(val_logits, val_labels)

                    val_preds = val_logits.argmax(dim=1)
                    correct += (val_preds == val_labels).sum().item()
                    total += val_labels.size(0)

                    batch_valid_loss.append(val_loss.item())

            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)
            train_loss = sum(batch_train_loss) / len(train_dataloader)
            train_losses.append(train_loss)

            valid_accuracy = correct / total
            valid_accuracies.append(valid_accuracy)
            valid_loss = sum(batch_valid_loss) / len(train_dataloader)
            valid_losses.append(valid_loss)

            # Log validation accuracy and loss
            if epoch % 10 == 0:
                mlflow.log_metric(key="train_accuracies", value=train_accuracy, step=epoch)
                mlflow.log_metric(key="train_losses", value=train_loss, step=epoch)
                mlflow.log_metric(key="valid_accuracies", value=valid_accuracy, step=epoch)
                mlflow.log_metric(key="valid_losses", value=valid_loss, step=epoch)


        # Log the parameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )

        signature = infer_signature()

        mlflow.transformers.log_model(
            transformers_model={'model': model, 'tokenizer': tokenizer},
            artifact_path='kcbert_hatespeech_classifier',
            task="hatespeech-classification",
            signature=signature
        )


def _save_model():
    mlflow.sklearn.log_model(
        clf,
        "iris_rf", 
        signature=signature, 
        input_example=input_example
    )



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

    train_recycle_classfication_model = PythonOperator(
        task_id="_train_recycle_classfication_model",
        python_callable=train_model,
    )


    process_dataset = DummyOperator(
        task_id="process_dataset",
        # python_callable=_process_dataset
    )

    load_kcbert_model = PythonOperator(
        task_id="load_kcbert_model",
        python_callable=_load_kcbert_model,
        op_kwargs={
            "base_model_name": "beomi/kcbert-base",
            "model_param_save_path": "/home/user/airflow/kcbert_hatespeech_classifier.pth",
        }
    )

    train_kcbert_model = PythonOperator(
        task_id="train_kcbert_model",
        python_callable=_train_kcbert_model,
        op_kwargs={
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
    #pyspark_test >> train_recycle_classfication_model 
    