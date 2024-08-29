import torch
import transformers

class KcbertModel():
    """
    Init KcBert model 
    """
    def __init__(
        self,
        base_model_name: str,
        model_param_save_path: str,
    ) -> None:
        # Load a pre-trained model from Hugging Face Hub
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=11)
        self.model.load_state_dict(torch.load(model_param_save_path))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)

        # Set device for model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    # return model and tokenizer
    def getModelAndTokenizer(self, ):
        return self.model, self.tokenizer
    
    # return device
    def getDevice(self, ):
        return self.device