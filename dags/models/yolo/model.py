from datetime import datetime

NOW_TIME = datetime.now().strftime('%y%m%d')

class YoloModel:
    def __init__(
        self, 
        class_path: str,
        param_path: str,
    ) -> None:
        import yaml
        from ultralytics import YOLO, settings

        self.class_path = class_path
        
        # load the configuration file 
        with open(param_path) as f:
            self.params = yaml.safe_load(f)["yolo"]

        # Load a pretrained YOLO model
        self.model = YOLO(self.params["model"])

        settings.update({
            "runs_dir": self.params["runs_dir"],
            "tensorboard": True,
        })
        
    def train(self):
        # Train the model using the 'recycle.yaml' dataset for n epochs
        self.model.train(
            data=self.class_path,
            epochs=self.params["epochs"],
            patience=self.params["patience"],
            batch=self.params["batch"],
            imgsz=self.params["imgsz"],
            cache=self.params["cache"],
            device=self.params["device"],
            workers=self.params["workers"],
            name=self.params["name"]+NOW_TIME,
            pretrained=self.params["pretrained"],
            optimizer=self.params["optimizer"],
            verbose=self.params["verbose"],
            lr0=self.params["lr0"],
            momentum=self.params["momentum"],
            weight_decay=self.params["weight_decay"],
            box=self.params["box"],
            cls=self.params["cls"],
            dropout=self.params["dropout"],
            val=self.params["val"],
            project=f"{NOW_TIME}"
        )

        # Evaluate the model's performance on the validation set
        self.model.val()

        import torch
        torch.save(self.model.state_dict(), f"./data/model/yolo/state_dict/{NOW_TIME}.pt")
