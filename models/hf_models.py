from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F

class HFCLIP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def encode_image(self, data, normalize=True):
        x = self.model.get_image_features(data)
        return F.normalize(x, dim=-1) if normalize else x
    def encode_text(self, data, normalize=True): 
        x = self.model.get_text_features(**data)
        return F.normalize(x, dim=-1) if normalize else x

def load_model(args):
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    preprocessing = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = lambda x: preprocessing(text=x, 
                                        return_tensors='pt', 
                                        max_length=77 if 'siglip' not in args.model_name.lower() else 64, 
                                        padding="max_length", 
                                        truncation=True)
    preprocess = lambda x: preprocessing(images=x, return_tensors='pt').pixel_values[0]
    return HFCLIP(model), preprocess, tokenizer