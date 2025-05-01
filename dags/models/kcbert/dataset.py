from torch.utils.data import Dataset

class CustomHateSpeechDataset(Dataset):
    """
    PostgreSQL에서 댓글 데이터를 불러와 혐오 발언 분류 학습에 활용되는 커스텀 Dataset 클래스.

    텍스트와 라벨을 로드하고, 지정된 tokenizer로 전처리하여 HuggingFace Transformers 모델 학습에 맞는 포맷으로 반환합니다.
    """

    def __init__(
        self, 
        tokenizer, 
        max_length=128
    ) -> None:
        """
        Args:
            tokenizer: HuggingFace의 BERT 기반 tokenizer
            max_length (int): 토큰 시퀀스 최대 길이 (default: 128)
        """
        import pandas as pd
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = hook.get_conn()

        self.dataset = pd.read_sql("""
            SELECT content, type
            FROM comment 
            WHERE type IS NOT NULL
        """, con=conn)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def getText(self, index):
        """
        주어진 인덱스의 원문 텍스트를 반환 (TensorBoard 시각화용)

        Args:
            index (int): 데이터 인덱스

        Returns:
            str: 텍스트 원문
        """
        return self.dataset.iloc[index, 0]

    def __len__(self) -> int:
        """
        전체 데이터 개수를 반환

        Returns:
            int: 데이터셋 길이
        """
        return len(self.dataset)
    
    def __getitem__(self, index) -> dict:
        """
        학습용으로 토크나이즈된 데이터와 라벨을 반환

        Args:
            index (int): 데이터 인덱스

        Returns:
            dict: {
                'input_ids': torch.Tensor,
                'attention_mask': torch.Tensor,
                'label': int
            }
        """
        text = self.dataset.iloc[index, 0]
        label = self.dataset.iloc[index, 1]

        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }
