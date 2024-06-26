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
        self.metric_module = evaluate.load('rouge')
        self.metric_tokenizer = AutoTokenizer.from_pretrained(mname)
        self.metric_model = AutoModel.from_pretrained(mname).to(self.device)
        self.centroids = np.load(f'data/centroids_{config.n_clusters}.npy')


    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['x'].to(self.device)
                y = self.tokenize(batch['y'])

                pred = self.predict(x)
                pred = self.tokenize(pred)
                
                rouge_batch_score, diverse_batch_score = self.evaluate(pred, y)
                rouge_batch_score += rouge_score
                diverse_score += diverse_batch_score


        txt = f"TEST Results\n"
        txt += f"-- Rouge Score: {round(score/len(self.dataloader), 2)}\n"
        txt += f"-- Diverse Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


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



    def evaluate(self, pred, label):
        if all(elem == '' for elem in pred):
            return 0.0, 0.0


        #Get Rouge Score Process        
        rouge_score = self.metric_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['rouge2'] * 100


        #Get Diverse Score Process
        encodings = self.metric_tokenizer(
            pred, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            semantic = self.metric_model(**encodings).last_hidden_state[:, 0, :]
            
        semantic = semantic.detach().to('cpu').numpy()
        distance = cdist(self.centroids, [semantic])
        pred_cluster = np.argmin(distance)

        
        cluster_acc = None

        return rouge_score, diverse_score