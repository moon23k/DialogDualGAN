import os, re, json 
from datasets import load_dataset
from transformers import T5TransformerFast



def preprocess_data(orig_data, volumn=36000):
    volumn_cnt = 0
    src_list, trg_list = [], []
    concat, processed = [], []

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
            src_list.append(dial_list[0])
            trg_list.append(dial_list[1])
            continue  #To avoid duplicate on below condition

        #Incase of dial_turns is even
        elif dial_turns % 2 == 0:
            src_list.extend(dial_list[0::2])
            trg_list.extend(dial_list[1::2])

            src_list.extend(dial_list[1:-1:2])
            trg_list.extend(dial_list[2::2])
        
        #Incase of dial_turns is odds
        elif dial_turns % 2 == 1:
            src_list.extend(dial_list[0:-1:2])
            trg_list.extend(dial_list[1::2])
            
            src_list.extend(dial_list[1::2])
            trg_list.extend(dial_list[2::2])   

    assert len(src_list) == len(trg_list)
    for src, trg in zip(src_list, trg_list):
        temp_dict = dict()
        temp_dict['src'] = src
        temp_dict['trg'] = trg
        
        concat.append(src + trg)
        processed.append(temp_dict)

        #End Condition
        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    return processed


def train_tokenizer(orig_data, max_vocab_size=30000):
    old_tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    tokenizer = old_tokenizer.train_new_from_iterator(orig_data, max_vocab_size)
    tokenizer.save_pretrained('data/tokenizer.json')
    del old_tokenizer
    
    return tokenizer



def tokenize_data(tokenized, tokenizer):
    tokenized_data = []
    for elem in tokenized:

        temp_dict = dict()
        encodings = tokenizer(elem['src'])

        temp_dict['input_ids'] = encodings.input_ids
        temp_dict['attention_mask'] = encodings.attention_mask
        temp_dict['labels'] = tokenizer.encode(elem['trg'])

        tokenized_data.append(temp_dict)
    
    return tokenized_data



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-6000], data_obj[-6000:-3000], data_obj[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')
    



def main(task):
    orig = load_dataset('daily_dialog', split='train')['dialog']
    tokenizer = train_tokenizer(orig)
    processed = preprocess_data(orig)

    #Tokenize Datasets
    tokenized = tokenize_data(processed, tokenizer)

    #Save Data
    save_data(tokenized)



if __name__ == '__main__':
    main()