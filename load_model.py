from transformers import AutoTokenizer, AutoModel
import torch
from load_llava import load_llava_model
import os


class ProcessModel:
    def init(self, model_path, ):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.load_model_and_tokenizer(model_path)

    def load_model_and_tokenizer(self):
        if "llava" in self.model_path:
            self.tokenizer, self.model, self.image_processor = load_llava_model(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)

    def set_hook(self, hook_type="hidden_state"):
        if hook_type == "hidden_state":
            pass
        else:
            raise ValueError(f"Input hook_type {hook_type} is not surpported!")

if __name__ == "__main__":
    m = ProcessModel("../models/llava-v1.5-7b")