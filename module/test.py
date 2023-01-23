class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.semantic_model = SentenceTransformer('bert-base-nli-mean-tokens')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    @staticmethod
    def print_rst(rst_dict):
        print('\n--- Test Results ---')
        print(f"  >> Semantic Similarity Score: {rst_dict['semantic_score']:.2f}")
        print(f"  >> Sequence Similarity Score: {rst_dict['sequence_score']:.2f}")
        print(f"  >> Labels Repeat Ratio: {rst_dict['labels_repeat_ratio']:.2f}")
        print(f"  >> Preds Repeat Ratio: {rst_dict['preds_repeat_ratio']:.2f}")
        print(f"  >> Spent Time: {rst_dict['spent_time']}")


    def get_sim_scores(self, preds, labels):
        semantic_score, sequence_score = 0, 0

        for p, l in zip(preds, labels):
            #semantic_score
            pred_emb = self.semantic_model.encode([p])
            label_emb = self.semantic_model.encode([l])
            semantic_score += cosine_similarity(pred_emb, label_emb).item()

            #sequence_score
            pred_bytes = list(bytes(p, 'utf-8'))
            label_bytes = list(bytes(l, 'utf-8'))
            sequence_score += SequenceMatcher(None, pred_bytes, label_bytes).ratio()
            
        return semantic_score / len(preds), sequence_score / len(preds)


    def test(self):
        self.model.eval()
        labels_list, preds_list = [], []
        sequence_scores, semantic_scores = 0, 0
        
        start_time = time.time()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.dataloader)):    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)          
                                
                preds = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            max_new_tokens=300, use_cache=True)
                
                labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                
                if not idx:
                    input_ids = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    order_list = ['first', 'second', 'third', 'fourth', 'fifth']
                    print('\n--- Test Output Examples ---')
                    for i in range(5):
                        print(f"\n{order_list[i].upper()}")
                        print(f"  >> uttr: {input_ids[i]}")
                        print(f"  >> resp: {labels[i]}")
                        print(f"  >> pred: {preds[i]}")
                    print('\n-------------------------------\n')

                labels_list.extend(labels) 
                preds_list.extend(preds)

                sem_score, seq_score = self.get_sim_scores(labels, preds)
                semantic_scores += sem_score
                sequence_scores += seq_score
        
        semantic_scores = round(semantic_scores * 100 / idx, 2)
        sequence_scores = round(sequence_scores * 100 / idx, 2)
        
        rst_dict = {'semantic_score': semantic_scores,
                    'sequence_score': sequence_scores, 
                    'labels_repeat_ratio': len(labels_list) / len(set(labels_list)),
                    'preds_repeat_ratio':  len(preds_list) / len(set(preds_list)),
                    'spent_time': self.measure_time(start_time, time.time())}
        
        #print and save test result
        self.print_rst(rst_dict)
