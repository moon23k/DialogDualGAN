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
        input_ids = self.data[idx]['input_ids']
        attention_mask = self.data[idx]['attention_mask']
        labels = self.data[idx]['labels']
        return input_ids, attention_mask, labels



def load_dataloader(config, split):
    self.pad_id = config.pad_id    

    def collate_fn(batch):
        ids_batch, mask_batch, labels_batch = [], [], []
        
        
        return {'input_ids': ids_batch,
                'attention_mask': mask_batch,
                'labels': labels_batch}


    return DataLoader(Dataset(split, config.n_cluster), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2)