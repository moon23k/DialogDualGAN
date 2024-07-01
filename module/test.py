import numpy as np
import torch, evaluate
from scipy.spatial.distance import cdist
from transformers import AutoModel, AutoTokenizer




class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.bos_id = config.bos_id
        self.device = config.device
        self.max_len = config.max_len
        
        mname = 'bert-base-uncased'
        self.rouge_module = evaluate.load('rouge')
        self.balance_tokenizer = AutoTokenizer.from_pretrained(mname)
        self.balance_model = AutoModel.from_pretrained(mname).to(self.device)
        self.balance_centroids = np.load(f'data/centroids_{config.n_clusters}.npy')



    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['x'].to(self.device)
                y = self.tokenize(batch['y'])

                pred = self.predict(x)
                pred = self.tokenize(pred)
                
                rouge_batch_score, 
                squad_batch_score = self.evaluate(pred, y)
                rouge_batch_score += rouge_score
                diverse_score += diverse_batch_score


        txt = f"TEST Results\n"
        txt += f"-- ROUGE Score: {round(rouge_score/len(self.dataloader), 2)}\n"
        txt += f"-- SQUAD Score: {round(squad_score/len(self.dataloader), 2)}\n"
        print(txt)




    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, self.max_len))
        pred = pred.type(torch.LongTensor).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.pad_mask(x)
        memory = self.model.encoder(x, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_out = self.model.decoder(y, memory, e_mask, None)

            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

        return pred



    def calc_squad_score(self, pred):
        
        encodings = self.balance_tokenizer(
            pred, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            semantics = self.balance_model(**encodings).last_hidden_state[:, 0, :]


        #cluster_dist = self.balance_centroids.predict(semantic.detach().to('cpu').numpy())
        cluster_distance = cdist(self.balance_centroids, [semantics])
        cluster_distribution = np.argmin(cluster_distance)

        max_cnt, min_cnt = np.max(cluster_distribution), np.min(cluster_distribution)
        score = 100 * (1 - (max_cnt - min_cnt) / self.n_cluster)
        
        return round(score, 2)



    def calc_rouge_score(self, pred, label):
        if all(elem == '' for elem in pred):
            return 0.0, 0.0

        score = self.rouge_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['rouge2'] * 100

        return round(score, 2)
