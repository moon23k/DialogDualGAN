import torch, math, time, evaluate
import torch.nn as nn
import torch.nn.functional as F



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader



    def test(self):
        return