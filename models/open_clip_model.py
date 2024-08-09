import open_clip
import torch
import logging
from open_clip.factory import _get_hf_config

HF_OPENCLIP_MODELS = ['Marqo/marqo-fashionCLIP','Marqo/marqo-fashionSigLIP']

def _get_hf_vit_name(model_name):
    try:
        return _get_hf_config(model_name)['model_cfg']['text_cfg']['hf_tokenizer_name'].split('/')[1] # ex.ViT-B-16-SigLIP
    except:
        return None
    
def load_model(args):
    if args.model_name == 'OpenFashionCLIP':
        return load_open_fashion_clip(args)
    
    model_name = args.model_name if args.model_name not in HF_OPENCLIP_MODELS else 'hf-hub:'+args.model_name
    pretrained = args.pretrained if args.model_name not in HF_OPENCLIP_MODELS else None
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=args.cache_dir)
    model.eval()
    
    # This is necessary for some models like SigLIP that don't load preprocess properly
    pretrained_list = dict(open_clip.list_pretrained())
    if args.model_name in pretrained_list:
        logging.info("Getting processor")
        _, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=pretrained_list[args.model_name], cache_dir=args.cache_dir)
    elif (args.model_name in HF_OPENCLIP_MODELS) and (_get_hf_vit_name(args.model_name) in pretrained_list):
        logging.info("Getting processor")
        _, _, preprocess = open_clip.create_model_and_transforms(_get_hf_vit_name(args.model_name), pretrained=pretrained_list[_get_hf_vit_name(args.model_name)], cache_dir=args.cache_dir)
    
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