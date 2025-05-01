class KcbertModel():
    """
    KoBERT 기반 혐오 표현 분류 모델 학습 클래스.

    텍스트 분류를 위한 파이프라인을 구성하며, 학습 과정 전반을 TensorBoard로 로깅하고
    모델 상태를 저장합니다. 데이터셋 로딩, 학습/검증 루프, 정확도 및 손실, F1 점수,
    PR 커브 시각화, 학습 텍스트 로깅 등을 포함합니다.
    """

    def __init__(self, param_path: str) -> None:
        """
        Args:
            param_path (str): 모델 학습 파라미터와 경로가 정의된 YAML 파일 경로
        """
        import yaml
        from torch.utils.tensorboard import SummaryWriter
        import transformers

        with open(param_path) as f:
            self.params = yaml.safe_load(f)["kcbert"]

        self.writer = SummaryWriter(
            log_dir="s3://midas-bucket-1/models/kcbert/",
            filename_suffix=f"kcbert_{NOW_TIME}"
        )

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.params["base_model_name"], num_labels=11
        )
        self.model.load_state_dict(torch.load(self.params["model_param_save_path"]))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.params["base_model_name"])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self):
        """
        모델 학습 메서드.

        - 사용자 정의 Dataset 로드
        - 학습/검증 데이터 분할
        - 정확도, 손실, F1 점수 기록 및 TensorBoard 시각화
        - 학습 완료 후 모델 파라미터 저장
        """
        from models.kcbert.dataset import CustomHateSpeechDataset
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim

        learning_rate = float(self.params["learning_rate"])
        batch_size = self.params["batch_size"]
        epochs = self.params["epochs"]

        dataset = CustomHateSpeechDataset(tokenizer=self.tokenizer, max_length=128)
        train, valid = train_test_split(
            dataset,
            test_size=self.params["test_size"], 
            random_state=self.params["random_state"]
        )

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            batch_train_loss, batch_valid_loss = [], []
            total = correct = 0

            self.model.train()
            for batch in train_dataloader:
                labels = batch["label"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                batch_train_loss.append(loss.item())

            # Validation
            all_val_labels, all_val_preds, all_val_probs = [], [], []
            total = correct = 0
            self.model.eval()
            with torch.no_grad():
                for val_batch in valid_dataloader:
                    val_labels = val_batch["label"].to(self.device)
                    val_input_ids = val_batch["input_ids"].to(self.device)
                    val_attention_mask = val_batch["attention_mask"].to(self.device)

                    val_outputs = self.model(val_input_ids, attention_mask=val_attention_mask)
                    val_loss = criterion(val_outputs.logits, val_labels)
                    val_preds = val_outputs.logits.argmax(dim=1)

                    correct += (val_preds == val_labels).sum().item()
                    total += val_labels.size(0)

                    all_val_labels.extend(val_labels.cpu().numpy())
                    all_val_preds.extend(val_preds.cpu().numpy())
                    all_val_probs.extend(torch.softmax(val_outputs.logits, dim=1).cpu().numpy())
                    batch_valid_loss.append(val_loss.item())

            train_accuracy = correct / total
            valid_accuracy = correct / total
            train_loss = sum(batch_train_loss) / len(train_dataloader)
            valid_loss = sum(batch_valid_loss) / len(valid_dataloader)

            self.log_accuracy_loss(train_accuracy, train_loss, valid_accuracy, valid_loss, epoch)
            f1 = self.log_pr_curve_and_f1(all_val_labels, all_val_preds, all_val_probs, epoch)
            self.log_dataset_text(dataset, max_texts=30)

        self.writer.add_hparams({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "test_size": self.params["test_size"],
            "random_state": self.params["random_state"]
        }, {
            'Accuracy/train': train_accuracy,
            'Accuracy/test': valid_accuracy,
            'F1 Score/test': f1
        })

        self.writer.close()
        torch.save(self.model.state_dict(), self.params["save_state_dict_dir"] + f"{NOW_TIME}.pt")

    def log_accuracy_loss(self, train_accuracy, train_loss, valid_accuracy, valid_loss, epoch):
        """
        TensorBoard에 정확도 및 손실 로그 기록

        Args:
            train_accuracy (float)
            train_loss (float)
            valid_accuracy (float)
            valid_loss (float)
            epoch (int)
        """
        self.writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/test", valid_accuracy, epoch)
        self.writer.add_scalar("Loss/test", valid_loss, epoch)

    def log_pr_curve_and_f1(self, all_val_labels, all_val_preds, all_val_probs, epoch):
        """
        PR 커브 및 F1 점수 TensorBoard 기록

        Args:
            all_val_labels (List[int])
            all_val_preds (List[int])
            all_val_probs (List[List[float]])
            epoch (int)

        Returns:
            float: macro F1 score
        """
        import numpy as np
        from sklearn.metrics import precision_recall_curve, f1_score

        f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        self.writer.add_scalar("F1 Score/test", f1, epoch)

        for class_idx in range(len(set(all_val_labels))):
            class_labels = np.array([1 if label == class_idx else 0 for label in all_val_labels])
            class_probs = np.array([prob[class_idx] for prob in all_val_probs])
            self.writer.add_pr_curve(f'PR Curve/Class {class_idx}', class_labels, class_probs, epoch)

        return f1

    def log_dataset_text(self, dataset, max_texts=30):
        """
        TensorBoard에 샘플 텍스트 기록

        Args:
            dataset (CustomHateSpeechDataset)
            max_texts (int): 최대 기록할 문장 수
        """
        for i in range(max_texts):
            text = dataset.getText(i)
            self.writer.add_text(f'Dataset Text {i}', text, global_step=0)
