HF_MODELS = ['patrickjohncyh/fashion-clip']

def load_model(args):
    if args.model_name in HF_MODELS:
       from .hf_models import load_model as _load
       model, preprocess, tokenizer = _load(args) 
    else:
        from .open_clip_model import load_model as _load
        model, preprocess, tokenizer = _load(args) 
    return model, preprocess, tokenizer