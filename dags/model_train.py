from ultralytics import YOLO

def train_model():
    model = YOLO("/home/user/airflow/yolo.pt")

if __name__ == "__main__":
    train_model()