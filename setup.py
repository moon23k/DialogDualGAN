import os, re, json 
import yaml, argparse
import sentencepiece as spm
from run import load_tokenizer
from datasets import load_dataset




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
        
    with open('data/concat.txt', 'w') as f:
        f.write('\n'.join(concat))
    
    return processed



def build_vocab(task):
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    assert os.path.exists(f'data/concat.txt')
    opt = f"--input=data/concat.txt\
            --model_prefix=data/spm\
            --vocab_size={vocab_config['vocab_size']}\
            --character_coverage={vocab_config['coverage']}\
            --model_type={vocab_config['type']}\
            --pad_id={vocab_config['pad_id']} --pad_piece={vocab_config['pad_piece']}\
            --unk_id={vocab_config['unk_id']} --unk_piece={vocab_config['unk_piece']}\
            --bos_id={vocab_config['bos_id']} --bos_piece={vocab_config['bos_piece']}\
            --eos_id={vocab_config['eos_id']} --eos_piece={vocab_config['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove(f'data/concat.txt')



def tokenize_data(task, tokenized, tokenizer):
    tokenized_data = []
    for elem in tokenized:
        temp_dict = dict()
        
        if task == 'sum':
            temp = []
            for seq in elem['src']:
                temp.append(tokenizer.EncodeAsIds(seq))
            temp_dict['src'] = temp
        else:    
            temp_dict['src'] = tokenizer.EncodeAsIds(elem['src'])
        
        temp_dict['trg'] = tokenizer.EncodeAsIds(elem['trg'])
        tokenized_data.append(temp_dict)
    
    return tokenized_data


def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-6000], data_obj[-6000:-3000], data_obj[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')
    


def main(task):
    orig = load_dataset('daily_dialog', split='train')['dialog']
    processed = preprocess_data(orig)

    #Build Vocab
    build_vocab(task)

    #Tokenize Datasets
    tokenizer = load_tokenizer(task)
    tokenized = tokenize_data(task, processed, tokenizer)

    #Save Data
    save_data(task, tokenized)


if __name__ == '__main__':
    main(args.task)