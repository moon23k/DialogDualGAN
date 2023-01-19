import os, argparse, torch
from tqdm import tqdm
from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader

from transformers import (set_seed,
                          T5Config,
                          T5TokenizerFast, 
                          T5ForConditionalGeneration)



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode

        self.clip = 1
        self.lr = 1e-4
        self.n_epochs = 10
        self.batch_size = 16
        self.iters_to_accumulate = 4
        
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.ckpt = f'ckpt/model.pt'

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer():
    assert os.path.exists('data/tokenizer')
    tokenizer = T5TokenizerFast.from_pretrained('data/tokenizer', model_max_length=300)
    return tokenizer


def inference(config, model, tokenizer):
    return


def print_model_desc(model):
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")


def load_model(config):
    model_cfg = T5Config()
    model_cfg.vocab_size = config.vocab_size
    model_cfg.update({'decoder_start_token_id': config.pad_id})

    model = T5ForConditionalGeneration(model_cfg)
    print(f"Model for {config.mode.upper()} has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']        
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {config.ckpt}")

    print_model_desc(model)
    return model.to(config.device)        


def main(args):
    set_seed(42)
    config = Config(args)
    tokenizer = load_tokenizer()
    setattr(config, 'pad_id', tokenizer.pad_token_id)
    setattr(config, 'vocab_size', tokenizer.vocab_size)
    model = load_model(config)    
    

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, test_dataloader, tokenizer)
        tester.test()
    
    elif config.mode == 'inference':
        inference(config, model, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-step', required=True)
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    #assert args.step in ['first', 'second', 'last']
    assert args.mode in ['train', 'test', 'inference']

    main(args)