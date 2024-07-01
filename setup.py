import os, re, json, yaml, argparse, numpy

import datasets
from datasets import load_dataset
from sklearn.cluster import KMeans
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents





def get_pre_fn(d_name):
    common_fn = lambda x: re.sub(r'\.$', '', re.sub(r'。', '', re.sub(r"\s*'\s*", "'", x))).lower().strip()

    if d_name == 'daily':
        return lambda x: re.sub(r"\s([?,.!](?:\s|$))", r'\1', common_fn(x))
    elif d_name == 'blend':
        return lambda x: common_fn(x).replace('  ', ' ')




def process_elem(elem, d_name, max_len=300):
    pre_fn = get_pre_fn(d_name)
    cond_fn = lambda seq: len(seq) <= max_len and not re.search(r'[;:]', seq)
    lst_fn = lambda x, fn: [fn(seq) for seq in x if cond_fn(fn(seq))]

    if d_name == 'daily':
        orig_turns = len(elem['dialog'])
        uttr_list = lst_fn(elem['dialog'], pre_fn)

    elif d_name == 'blend':
        p_elem, f_elem, g_elem = elem['previous_utterance'], elem['free_messages'], elem['guided_messages']
        uttr_list, free_list, guid_list = lst_fn(p_elem, pre_fn), lst_fn(f_elem, pre_fn), lst_fn(g_elem, pre_fn)
        orig_turns = len(p_elem) + len(f_elem) + len(g_elem)

        for free, guided in zip(free_list, guid_list):
            uttr_list.append(pre_fn(free))
            uttr_list.append(pre_fn(guided))

    return uttr_list if len(uttr_list) == orig_turns else []




def split_uttrs(dial_list):
    x_data, y_data = [], []
    dial_turns = len(dial_list)

    if dial_turns < 2:
        return

    elif dial_turns == 2:
        x_data.append(dial_list[0])
        y_data.append(dial_list[1])
        return  x_data, y_data #To avoid duplicate on below condition

    #Incase of dial_turns is even
    elif dial_turns % 2 == 0:
        x_data.extend(dial_list[0::2])
        y_data.extend(dial_list[1::2])

        x_data.extend(dial_list[1:-1:2])
        y_data.extend(dial_list[2::2])

    #Incase of dial_turns is odds
    elif dial_turns % 2 == 1:
        x_data.extend(dial_list[0:-1:2])
        y_data.extend(dial_list[1::2])

        x_data.extend(dial_list[1::2])
        y_data.extend(dial_list[2::2])

    assert len(x_data) == len(y_data)
    return x_data, y_data




def process_data():
    x_data, y_data = [], []

    data_dict = {
        'daily': load_dataset('daily_dialog', trust_remote_code=True),
        'blend': load_dataset('blended_skill_talk', trust_remote_code=True)
    }

    for data_name, data_obj in data_dict.items():
        for split in ['train', 'validation', 'test']:
            for elem in data_obj[split]:

                uttr_list = process_elem(elem, data_name)
                if not uttr_list:
                    continue

                x_uttrs, y_uttrs = split_uttrs(uttr_list)
                x_data.extend(x_uttrs)
                y_data.extend(y_uttrs)

    return [{'x': x, 'y': y} for x, y in zip(x_data, y_data)]





def extract_semantics(processed_data):

    def batchify(data, batch_size=128):
        for idx in range(0, len(data), batch_size):
            yield data[idx : idx+batch_size]


    #Prerequisites
    mname = 'google-bert/bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(mname)
    classifier = AutoModel.from_pretrained(mname).to(device)
    classifier.eval()


    #Extract Semantic Vectors
    semantics = []
    batchified = batchify([elem['y'] for elem in processed_data])
    for batch in batchified:
        encodings = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            semantic = classifier(**encodings).last_hidden_state[:, 0, :]
            
        semantics.append(semantic.detach().to('cpu').numpy())


    return numpy.vstack(semantics)




def cluster_data(processed_data, n_cluster, total_volumn=111000):

    #KMeans Clustering
    semantics = extract_semantics(processed_data)
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, init='k-means++').fit(semantics)
    clusters = kmeans.labels_.tolist()

    #Balance Dataset based on Cluster
    corpus = []
    clustered_data = defaultdict(list)
    volumn_count = defaultdict(int)
    cluster_volumn = total_volumn // n_cluster
    
    for idx, cluster in enumerate(clusters):
        if volumn_count[cluster] < cluster_volumn:
            clustered_data[cluster].append({
                'x': processed_data[idx]['x'], 
                'y': processed_data[idx]['y']
            })
            volumn_count[cluster] += 1

    return clustered_data




def balance_data(clustered_data, n_cluster):
    corpus, balanced_data = [], []
    for i in range(len(clustered_data[0])):
        for j in range(n_cluster):
            x, y = clustered_data[j][i]['x'], clustered_data[j][i]['y']
            corpus.append(x)
            corpus.append(y)            
            balanced_data.append({'x': x, 'y': y, 'cluster': j})

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    return balanced_data




def train_tokenizer():
    corpus_path = 'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        tok_config = yaml.load(f, Loader=yaml.FullLoader)['tokenizer']

    tokenizer = Tokenizer(BPE(unk_token=tok_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=tok_config['vocab_size'], 
        special_tokens=[
            tok_config['pad_token'], 
            tok_config['unk_token'],
            tok_config['bos_token'],
            tok_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")




def save_data(data_obj):


    return




def main(n_cluster):

    if not os.path.exists('data/raw_data.json'):
        processed_data = process_data()

    clustered_data = cluster_data(processed_data, n_cluster)
    balanced_data = balance_data(clustered_data, n_cluster)

    train_tokenizer()
    save_data(balanced_data)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_cluster', required=True)
    args = parser.parse_args()
    
    main(args)    