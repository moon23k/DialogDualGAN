import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence





class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):        
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx]['x']).ids
        y = self.tokenizer.encode(self.data[idx]['y']).ids

        return torch.LongTensor(x), torch.LongTensor(y)




class Collator(object):
    def __init__(self, pad_id):
        self.pad_args = {'batch_first': True, 'padding_value': pad_id}


    def __call__(self, batch):
        x_batch, y_batch, cluster_batch = zip(*batch)
        x_batch = pad_sequence(x_batch, **self.pad_args)
        y_batch = pad_sequence(y_batch, **self.pad_args)

        return {'x': x_batch, 'y': y_batch}




def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(config.pad_id),
        pin_memory=True,
        num_workers=2
    )