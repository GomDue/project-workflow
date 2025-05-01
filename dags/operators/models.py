from airflow.models import BaseOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

class StoreModelInS3Operator(BaseOperator):
    """
    학습된 모델 파일을 AWS S3에 저장하는 커스텀 Airflow Operator.

    Args:
        conn_id (str): Airflow에 등록된 AWS S3 연결 ID (기본값: "aws_s3").
        **kwargs: Airflow Operator 인자 (task_id 등).
    """

    def __init__(self, conn_id="aws_s3", bucket: str = "", prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self._conn_id = conn_id
        self.bucket = bucket
        self.prefix = prefix

    def execute(self, context):
        """
        S3 버킷 내 지정된 prefix 하위의 객체 목록을 출력합니다.

        예시 용도이며, 실제 모델 파일 업로드 로직으로 확장 가능.
        """
        s3_hook = S3Hook(aws_conn_id=self._conn_id)
        keys = s3_hook.list_keys(bucket_name=self.bucket, prefix=self.prefix)

        if not keys:
            self.log.info("No objects found in the specified S3 prefix.")
        else:
            for key in keys:
                self.log.info(f"Found object: {key}")
