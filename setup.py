import os, re, json, yaml, argparse

import datasets
from datasets import load_dataset

import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents






def load_orig_data(d_name):
    if d_name == 'daily':
        return load_dataset('daily_dialog', trust_remote_code=True)
    elif d_name == 'blend':
        return load_dataset('blended_skill_talk', trust_remote_code=True)




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




def process_data(d_name):
    x_data, y_data = [], []
    orig_data = load_orig_data(d_name)
    
    for split in ['train', 'validation', 'test']:
        for elem in orig_data[split]:

            uttr_list = process_elem(elem, d_name)
            if not uttr_list:
                continue

            x_uttrs, y_uttrs = split_uttrs(uttr_list)
            x_data.extend(x_uttrs)
            y_data.extend(y_uttrs)

    return [{'x': x, 'y': y} for x, y in zip(x_data, y_data)]




def clustering(data_semantics, mname, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++').fit(data_semantics)
    clusters = kmeans.labels_.tolist()

    f_name = f"data/clusters/{mname}_cluster_{n_clusters}.json"
    with open(f_name, 'w') as f:
        json.dump(clusters, f)




def balance_data():
    return






def main(args):
    process_raw_data('daily')
    process_raw_data('blend')
    
    mname = 'google-bert/bert-base-uncased' if self.model == 'bert' else 'microsoft/DialoGPT-small'
    tokenizer = AutoTokenizer.from_pretrained(mname)
    classifier = AutoModel.from_pretrained(mname).to(config.device)
    classifier.eval()



    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-cluster', required=True)
    
    args = parser.parse_args()
    assert args.model in ['bert', 'dialogpt']
    assert args.cluster in [10, 20, 30, 50]
    
    main(args)    