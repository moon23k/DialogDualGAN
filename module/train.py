import time, math, json, torch
import torch.amp as amp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__()

        self.model = model
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.strategy = config.strategy
        self.vocab_size = config.vocab_size

        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.generate_freq = config.generate_freq
        self.iters_to_accumulate = config.iters_to_accumulate        

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.ckpt = config.ckpt
        self.record_path = f"ckpt/{self.strategy}_train.json"
        self.record_keys = ['epoch', 'train_loss', 'train_ppl', 
                            'valid_loss', 'valid_ppl', 
                            'learning_rate', 'train_time']


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              Valid Loss: {record_dict['valid_loss']:.3f}\n""".replace(' ' * 14, ''))


    def train(self):
        best_loss, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss >= val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)



    def get_loss(self, batch, idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        if (self.strategy == 'generative') and ((idx + 1) % self.generate_freq == 0):
            outputs = self.model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=300, use_cache=True,
                                          output_scores=True, return_dict_in_generate=True)
            
            gen_logits = torch.stack(outputs.scores)
            pad_len = gen_logits.size(0) - labels.size(1)
            
            if pad_len < 0:
                labels = labels[:, :gen_logits.size(0)]
            elif pad_len > 0:
                pad_tensor = torch.zeros(labels.size(0), pad_len, dtype=torch.long)
                labels = torch.cat((labels, pad_tensor.to(self.device)), dim=-1)

            gen_loss = F.cross_entropy(gen_logits.contiguous().view(-1, self.vocab_size), 
                                    labels.contiguous().view(-1))
            gen_loss.requires_grad_(True)
            return gen_loss


        base_loss = self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels).loss
      
        if self.strategy == 'base' or \
        (self.strategy == 'generative') and ((idx + 1) % self.generate_freq):
            return base_loss

        elif self.strategy == 'token':
            decoder_input_ids = torch.zeros(input_ids.size(0), 1, dtype=torch.long).to(self.device)
            tok_logits = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids).logits
            
            tok_loss = F.cross_entropy(tok_logits.contiguous().view(-1, self.vocab_size),
                                       labels[:, 0].contiguous())

            return (base_loss * 0.5) + (tok_loss.item() * 0.5)
          


    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        for idx, batch in enumerate(self.train_dataloader):

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.get_loss(batch, idx)
                loss = loss / self.iters_to_accumulate
                
            #Backward Loss
            self.scaler.scale(loss).backward()        
            
            if (idx + 1) % self.iters_to_accumulate == 0:
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)    
        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        tot_len = len(self.valid_dataloader)
        
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_dataloader):
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = self.get_loss(batch, idx)
                epoch_loss += loss.item()
                
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)    
        return epoch_loss, epoch_ppl