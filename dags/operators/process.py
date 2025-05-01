import os

from airflow.models.dag import BaseOperator
from airflow.providers.google.suite.hooks.sheets import GSheetsHook
from airflow.utils.decorators import apply_defaults

from typing import Optional, Union, Sequence


class GoogleSheetToCSVOperator(BaseOperator):
    """
    Google Sheet 데이터를 로컬 CSV 파일로 저장하는 커스텀 Airflow Operator.

    이 오퍼레이터는 Google Sheets API를 통해 특정 Sheet 범위의 데이터를 가져와
    지정된 경로에 CSV 파일로 저장합니다. Spark나 Pandas 기반의 전처리 작업 전
    정형 데이터 수집용으로 유용하게 사용됩니다.

    Example:
        GoogleSheetToCSVOperator(
            task_id="download_gdrive_file",
            gsheet_id="your_spreadsheet_id",
            range="Sheet1!A1:E100",
            file_name="data.csv",
            save_path="/path/to/save",
            gcp_conn_id="google_cloud_default"
        )
    """

    @apply_defaults
    def __init__(
        self,
        gsheet_id: str,
        range: str,
        file_name: str,
        save_path: str,
        gcp_conn_id: str,
        delegate_to: Optional[str] = None,
        impersonation_chain: Optional[Union[str, Sequence[str]]] = None,
        api_version: str = 'v4',
        **kwargs
    ) -> None:
        """
        Args:
            gsheet_id (str): 구글 시트의 고유 ID
            range (str): 불러올 셀 범위 (예: "시트1!A1:E100")
            file_name (str): 저장할 CSV 파일 이름
            save_path (str): CSV 파일을 저장할 로컬 경로
            gcp_conn_id (str): Airflow에서 설정한 GCP 연결 ID
            delegate_to (Optional[str]): 위임할 계정
            impersonation_chain (Optional[Union[str, Sequence[str]]]): 권한 위임 체인
            api_version (str): 사용할 Google Sheets API 버전 (기본값: 'v4')
        """
        super().__init__(**kwargs)
        self.api_version = api_version
        self.gcp_conn_id = gcp_conn_id
        self.delegate_to = delegate_to
        self.impersonation_chain = impersonation_chain
        self.gsheet_id = gsheet_id
        self.range = range
        self.file_name = file_name
        self.save_path = save_path

    def execute(self, context):
        """
        Google Sheet 데이터를 불러와 지정된 경로에 CSV 파일로 저장합니다.

        1. GSheetsHook을 이용해 지정된 시트 ID와 범위로부터 데이터를 로드
        2. pandas DataFrame으로 변환
        3. 지정된 경로에 CSV 파일로 저장
        """
        import pandas as pd

        hook = GSheetsHook(
            gcp_conn_id=self.gcp_conn_id,
        )
        self.log.info("GSheetsHook 인스턴스 생성 완료")

        spreadsheet = hook.get_values(spreadsheet_id=self.gsheet_id, range_=self.range)
        self.log.info(f"Google Sheet 데이터 로드 완료: {hook.get_spreadsheet(spreadsheet_id=self.gsheet_id)}")

        os.makedirs(self.save_path, exist_ok=True)

        df = pd.DataFrame(spreadsheet[2:], columns=spreadsheet[1])
        file_path = os.path.join(self.save_path, self.file_name)
        df.to_csv(file_path, index=False)
        self.log.info(f"CSV 파일 저장 완료: {file_path}")
