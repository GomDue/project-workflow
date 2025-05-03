from airflow.models import BaseOperator

class HateSpeechOperator(BaseOperator):
    """
    AWS RDS에서 데이터를 조회하여 혐오 표현 분석 등 후속 처리를 위한 데이터프레임 형태로 반환하는 Airflow 커스텀 Operator.

    Args:
        conn_id (str): Airflow에 등록된 AWS RDS 연결 ID.
        **kwargs: Airflow Operator 인자 (task_id 등).
    """

    def __init__(self, conn_id, **kwargs):
        super().__init__(**kwargs)
        self._conn_id = conn_id

    def execute(self, context):
        """
        RDS에 연결하여 account 테이블에서 데이터를 조회한 후, Pandas DataFrame 형태로 출력합니다.
        후속 처리(예: 혐오 표현 감지 등)를 위한 데이터 수집 단계로 활용됩니다.
        """
        import pandas as pd
        from hooks.aws_rds_hook import AWSRDSHook

        rds_hook = AWSRDSHook(self._conn_id)
        rds_conn, rds_cur = rds_hook.get_conn()

        self.log.info("Executing SQL query to retrieve data from 'account' table...")
        df = pd.read_sql("SELECT * FROM account", rds_conn)
        self.log.info(f"Retrieved {len(df)} records from account table.")
        print(df)

        rds_hook.close()
