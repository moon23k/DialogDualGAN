import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader, load_model, 
    Trainer, Tester, SeqGenerator
)




def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.ckpt = 'ckpt/model.pt'
        self.tokenizer_path = 'data/tokenizer.json'
        
        use_cuda = torch.cuda.is_available()
        device_condition = use_cuda and self.mode != 'inference'
        self.device_type = 'cuda' if device_condition else 'cpu'
        self.device = torch.device(self.device_type)    


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)
    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer



def main(args):
    set_seed(42)
    config = Config(args)
    tokenizer = load_tokenizer()
    model = load_model(config)    
    

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, test_dataloader, tokenizer)
        tester.test()
    
    elif config.mode == 'inference':
        inference(config, model, tokenizer)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']

    main(args)