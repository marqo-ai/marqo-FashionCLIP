import open_clip
import torch
import logging

HF_OPENCLIP_MODELS = ['Marqo/marqo-fashionCLIP','Marqo/marqo-fashionSigLIP']

def load_model(args):
    if args.model_name == 'OpenFashionCLIP':
        return load_open_fashion_clip(args)
    
    model_name = args.model_name if args.model_name not in HF_OPENCLIP_MODELS else 'hf-hub:'+args.model_name
    pretrained = args.pretrained if args.model_name not in HF_OPENCLIP_MODELS else None
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=args.cache_dir)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

def load_open_fashion_clip(args):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', cache_dir=args.cache_dir)
    state_dict = torch.load(args.pretrained)
    model.load_state_dict(state_dict['CLIP'])
    logging.info(f'Loaded ViT-B-32 weights from {args.pretrained}.')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer