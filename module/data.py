import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence





class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, n_cluster):
        super().__init__()
        self.dial_data, self.cluster_data = self.load_data(split, n_cluster)


    @staticmethod
    def load_data(split):
        
        with open(f"data/{split}.json", 'r') as f:
            dial_data = json.load(f)

        with open(f"data/cluster.json", 'r') as f:
            cluster_data = json.load(f)

        return dial_data, cluster_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.dial_data[idx]['x']
        y = self.dial_data[idx]['y']
        cluster = self.cluster_data[idx]['cluster']
        return x, y, cluster




class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id


    def __call__(self, batch):
        x_batch, y_batch, cluster_batch = zip(*batch)

        return {'x': self.pad_batch(x_batch),
                'y': self.pad_batch(y_batch),
                'cluster': cluster_batch}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True, 
            padding_value=self.pad_id
        )




def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, config.task, split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(config.pad_id),
        pin_memory=True,
        num_workers=2
    )