from transformers import CLIPModel, CLIPProcessor
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
        x = self.model.get_text_features(data)
        return F.normalize(x, dim=-1) if normalize else x

def load_model(args):
    model = CLIPModel.from_pretrained(args.model_name)
    preprocessing = CLIPProcessor.from_pretrained(args.model_name)
    tokenizer = lambda x: preprocessing.tokenizer(x, return_tensors='pt', max_length=77, padding="max_length", truncation=True)['input_ids']
    preprocess = lambda x: preprocessing.image_processor(x, return_tensors='pt').pixel_values[0]
    return HFCLIP(model), preprocess, tokenizer