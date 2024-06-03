import os, re, json, torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from run import Config

from datasets import load_dataset
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer





def preprocess_data(orig_data):
    volumn_cnt = 0
    uttr_list, resp_list, processed = [], [], []

    for dial in orig_data:
        dial_list = []
        dial_turns = len(dial)
        
        for uttr in dial:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr)
            dial_list.append(_uttr.strip().lower())
        
        if dial_turns < 2:
            continue

        elif dial_turns == 2:
            uttr_list.append(dial_list[0])
            resp_list.append(dial_list[1])
            continue  #To avoid duplicate on below condition

        #Incase of dial_turns is even
        elif dial_turns % 2 == 0:
            uttr_list.extend(dial_list[0::2])
            resp_list.extend(dial_list[1::2])

            uttr_list.extend(dial_list[1:-1:2])
            resp_list.extend(dial_list[2::2])
        
        #Incase of dial_turns is odds
        elif dial_turns % 2 == 1:
            uttr_list.extend(dial_list[0:-1:2])
            resp_list.extend(dial_list[1::2])
            
            uttr_list.extend(dial_list[1::2])
            resp_list.extend(dial_list[2::2])   

    assert len(uttr_list) == len(resp_list)
    for uttr, resp in zip(uttr_list, resp_list):
        temp_dict = dict()
        temp_dict['uttr'] = uttr
        temp_dict['resp'] = resp
        processed.append(temp_dict)
    
    return processed




def batchify(data, batch_size=16):
    for idx in range(0, len(data), batch_size):
        yield data[idx : idx+batch_size]




def get_clusters(model, tokenizer, data_obj, n_clusters=20):
    model.eval()
    responses = [elem['resp'] for elem in data_obj]
    batchified = batchify(responses)

    semantics = []
    for batch in tqdm(batchified):    
        encodings = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
        input_ids, attention_mask = encodings.input_ids, encodings.attention_mask
        max_len = attention_mask.size(1)
    
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        semantic = torch.matmul(attention_mask.type(torch.float32).view(-1, 1, max_len), output).squeeze()
        
        if semantic.dim() == 1:
            semantic = semantic.unsqueeze(0)

        semantics.extend(semantic.detach().to('cpu').numpy())


    semantics = np.array(semantics)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++').fit(semantics)

    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    np.save('data/centroids', centroids)

    return clusters




def tokenize_data(tokenizer, data_obj):
    tokenized = []

    for elem in data_obj:
        uttr_encodings = tokenizer(elem['uttr'])

        input_ids = uttr_encodings.input_ids
        attention_mask = uttr_encodings.attention_mask

        labels = tokenizer(elem['resp']).input_ids

        tokenized.append({'input_ids': input_ids,
                          'attention_mask': attention_mask,
                          'labels': labels})    

    return tokenized





def save_data(data_obj): 
    train, valid, test = data_obj[:-6000], data_obj[-6000:-3000], data_obj[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/gen_{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/gen_{key}.json')





def main():
    #Prerequisite
    config = Config()
    mname = config.classifier_name
    tokenizer = AutoTokenizer.from_pretrained(mname)
    classifier = AutoModel.from_pretrained(mname).to(config.device)


    #Load orig_data
    orig_data = load_dataset('daily_dialog')


    #preprocess orig data
    processed_data = preprocess_data(orig_data['train']['dialog']) +\
                     preprocess_data(orig_data['validation']['dialog']) +\
                     preprocess_data(orig_data['test']['dialog'])

    
    #Add cluster info up to the processed data
    cluster_data = get_clusters(sem_model, sem_tokenizer, processed_data)


    #tokenize data and add up cluster info
    tokenized_data = tokenize_data(tokenizer, processed_data)
    for elem, cluster in zip(tokenized_data, cluster_data):
        elem.update({'cluster': cluster})

    
    #save the data
    save_data(tokenized_data)



if __name__ == '__main__':
    main()