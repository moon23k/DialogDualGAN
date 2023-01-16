import numpy as np
import sentencepiece as spm
import os, yaml, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from module.model import load_model
from module.data import load_dataloader

from module.test import Tester
from module.train import Trainer
from module.search import Search



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    
        self.step = args.step
        self.mode = args.mode
        self.ckpt = f"ckpt/{self.step}.pt"

        self.iters_to_accumulate = 4

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.device_type = 'cuda'
        else:
            self.device_type = 'cpu'

        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')

        
        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def load_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')
    return tokenizer


def inference(config, model, tokenizer):
    search_module = Search(config, model, tokenizer)

    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #Enc Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        if config.search_method == 'beam':
            output_seq = search_module.beam_search(input_seq)
        else:
            output_seq = search_module.greedy_search(input_seq)
        print(f"Model Out Sequence >> {output_seq}")       



def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)    
    tokenizer = load_tokenizer()


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, test_dataloader, tokenizer)
        tester.test()
        tester.inference_test()
    
    elif config.mode == 'inference':
        translator = inference(config, model, tokenizer)
        translator.translate()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-step', required=True)
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.step in ['first', 'second', 'third']
    assert args.mode in ['train', 'test', 'inference']

    main(args)