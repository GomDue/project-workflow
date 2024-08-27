from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime
import os

from operators.hatespeechOperator import HateSpeechOperator
from operators.recycleSolutionOperator import RecycleSolutionOperator

from model_train import train_model

AIRFLOW_DIR = os.path.abspath("./airflow")
PYSPARK_DIR = os.path.join(AIRFLOW_DIR, "spark", "app")

RECYCLE_SOLUTION_DIR = os.path.join(AIRFLOW_DIR, "data", "solutions")
RECYCLE_SOLUTION_FILE_NAME = "recycle_solution"

GOOGLE_SHEET_ID = "1O6ryWUCco2_tOlzvtcPCs3eDTPnIQkKDH2955tCUdxg"
GOOGLE_SHEET_NAME = "시트1"

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


# load by dvc
def model_load():
    # model_name = "beomi/kcbert-base"
    # model_save_path = 'kc_bert_emotion_classifier.pth'
    
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)
    # model.load_state_dict(torch.load(model_save_path))

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    pass


def data_preparation():
    pass

def model_train_and_validation():
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_dataloader, 1):
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
            total_loss += loss.item()

            if i % 100 == 0: 
                print(f"Epoch {epoch+1}/{epochs} - Batch {i}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")


        model.eval()
        val_total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_batch in valid_dataloader:
                # Validation 데이터 가져오기
                val_input_ids = val_batch['input_ids']
                val_attention_mask = val_batch['attention_mask']
                val_labels = val_batch['label']

                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                val_labels = val_labels.to(device)

                # 모델 예측
                val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                val_logits = val_outputs.logits

                # 손실 계산
                val_loss = criterion(val_logits, val_labels)
                val_total_loss += val_loss.item()

                # 정확도 계산
                val_preds = val_logits.argmax(dim=1)
                correct += (val_preds == val_labels).sum().item()
                total += val_labels.size(0)

        val_avg_loss = val_total_loss / len(valid_dataloader)
        val_accuracy = correct / total
        print(f"Validation Loss: {val_avg_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")


def model_export():
    pass


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 6, 2),
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
        gsheet_id=GOOGLE_SHEET_ID,
        range=GOOGLE_SHEET_NAME,
        file_name=f"{RECYCLE_SOLUTION_FILE_NAME}_{datetime.now().strftime('%y%m%d')}.csv",
        destination=RECYCLE_SOLUTION_DIR,
    )

    process_recycle_solution = SparkSubmitOperator(
        task_id='process_recycle_solution',
        application=os.path.join(PYSPARK_DIR, 'process_recycle_solution.py'),
        conn_id='spark_default',
        conf={"spark.master":"spark://spark:7077"},
    )

    data_preparation = PythonOperator(
        task_id='data_preparation',
        python_callable=data_preparation,
        provide_context=True,
        depends_on_past=True
    )

    train_recycle_classfication_model = PythonOperator(
        task_id="_train_recycle_classfication_model",
        python_callable=train_model,
    )

    # Task sequence
    download_gdrive_file >> process_recycle_solution
    connect_aws_s3 >> data_preparation
    connect_aws_rds >> data_preparation
    #pyspark_test >> train_recycle_classfication_model 
    