import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids']
        attention_mask = self.data[idx]['attention_mask']
        labels = self.data[idx]['labels']
        return input_ids, attention_mask, labels



def load_dataloader(config, split):
    global pad_id
    pad_id = config.pad_id    

    def collate_fn(batch):
        ids_batch, mask_batch, labels_batch = [], []
        
        for ids, mask, labels in batch:
            src_batch.append(torch.LongTensor(src))
            trg_batch.append(torch.LongTensor(trg))
        
        ids_batch.append(torch.LongTensor(ids)) 
        mask_batch.append(torch.LongTensor(mask))
        labels_batch.append(torch.LongTensor(labels))


        ids_batch = pad_sequence(ids_batch,
                                 batch_first=True,
                                 padding_value=pad_id)
        
        mask_batch = pad_sequence(mask_batch, 
                                  batch_first=True, 
                                  padding_value=pad_id)

        labels_batch = pad_sequence(labels_batch, 
                                    batch_first=True, 
                                    padding_value=pad_id)
        
        return {'input_ids': input_ids_batch,
                'attention_mask': mask_batch,
                'labels': labels_batch}


    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2)