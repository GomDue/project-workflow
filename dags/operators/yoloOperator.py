from airflow.models import BaseOperator

class YOLOOperator(BaseOperator):
    def __init__(
        self, 
        conn_id: str, 
        yolo_class: str,
        yolo_param: str,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._conn_id = conn_id
        self._yolo_class = yolo_class
        self._yolo_param = yolo_param
    
    def execute(self, context):
        import yaml

        from ultralytics import YOLO, settings

        # Use mlflow in YOLO model
        settings.update({"mlflow": True})

        # load the configuration file 
        with open(self._yolo_param) as f:
            params = yaml.safe_load(f)["yolo"]

        # Load a pretrained YOLO model (recommended for training)
        model = YOLO(params["model"])

        # Train the model using the 'recycle.yaml' dataset for n epochs
        model.train(
            data=self._yolo_class,
            epochs=params["epochs"],
            patience=params["patience"],
            batch=params["batch"],
            imgsz=params["imgsz"],
            cache=params["cache"],
            device=params["device"],
            workers=params["workers"],
            name=params["name"],
            pretrained=params["pretrained"],
            optimizer=params["optimizer"],
            verbose=params["verbose"],
            lr0=params["lr0"],
            momentum=params["momentum"],
            weight_decay=params["weight_decay"],
            box=params["box"],
            cls=params["cls"],
            dropout=params["dropout"],
            val=params["val"],
        )

        # Evaluate the model's performance on the validation set
        model.val()
