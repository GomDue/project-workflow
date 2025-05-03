from airflow.hooks.base import BaseHook

class AWSRDSHook(BaseHook):
    """
    Airflow에서 AWS RDS(MySQL)에 연결하기 위한 커스텀 Hook 클래스.

    Airflow Variables를 사용해 RDS 접속 정보를 받아와 pymysql을 통해 연결하며,
    연결된 커넥션과 커서를 반환합니다.
    """

    def __init__(self, conn_id):
        """
        Args:
            conn_id (str): Airflow 연결 ID (사용되지 않지만 확장성 고려)
        """
        super().__init__()
        self._conn_id = conn_id
        self._conn = None
        self._cur = None

    def get_conn(self):
        """
        Airflow Variable을 기반으로 RDS에 연결하고 (conn, cursor) 튜플을 반환합니다.

        Returns:
            tuple: (pymysql.Connection, pymysql.Cursor)

        Raises:
            Exception: 연결 실패 시 에러 메시지 출력
        """
        from airflow.models import Variable
        import pymysql

        if self._conn is None:
            try:
                self._conn = pymysql.connect(
                    host=Variable.get('aws_rds_host'),
                    user=Variable.get('aws_rds_user'),
                    password=Variable.get('aws_rds_password'),
                    port=3306,
                    database=Variable.get('aws_rds_database'),
                )
                self._cur = self._conn.cursor()
            except Exception as e:
                print(f"Failed to connect to RDS: {e}")
                self._conn, self._cur = None, None

        return self._conn, self._cur

    def close(self):
        """
        연결된 커서와 커넥션을 종료합니다.
        """
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()
