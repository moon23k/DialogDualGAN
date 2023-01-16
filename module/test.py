import torch, math, time
import torch.nn as nn
import torch.nn.functional as F
from module.search import Search



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.batch_size = config.batch_size        
        self.vocab_size = config.vocab_size
        self.search = Search(config, self.model, tokenizer)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, 
                                             label_smoothing=0.1).to(self.device)



