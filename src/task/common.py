import sys, os

def NER_linesplit(lines, label2id:dict):
    n_type = len(lines[0].strip().split())
    if n_type == 2:
        tokens, labels = [], []
        temp={'t':[],'l':[]}
        for line in lines:
            line = line.split()
            match len(line):
                case 0:
                    if len(temp['t']):
                        tokens.append(temp['t'])
                        labels.append(temp['l'])
                        temp['t'], temp['l'] = [], []
                case 2:
                    temp['t'].append(line[0])
                    temp['l'].append(label2id[line[1]])
        return tokens, labels, None
    elif n_type == 3:
        tokens, labels, checks = [], [], []
        temp = {'t':[], 'l':[], 'c':[]}
        for line in lines:
            line = line.split()
            match len(line):
                case 0:
                    if len(temp['t']):
                        tokens.append(temp['t'])
                        labels.append(temp['l'])
                        checks.append(temp['c'])
                        temp['t'], temp['l'], temp['c'] = [], [], []
                case 3:
                    temp['t'].append(line[0])
                    temp['l'].append(label2id[line[1]])
                    temp['c'].append(int(line[2]))
        return tokens, labels, checks

def NER_line2data(tokenizer, tokens:list, labels:list, checks:list=None):
    data = []

    C = tokenizer.cls_token_id
    L = tokenizer.model_max_length

    if checks:
        for toks, labs, chks in zip(tokens, labels, checks):
            assert len(toks) == len(labs)
            item_t = [C]
            item_l = [-100]
            item_c = [0]

            for tok, lab, chk in zip(toks, labs, chks):
                if not len(tok):
                    continue

                ids = tokenizer(tok, add_special_tokens=False).input_ids

                lab = [lab]*len(ids)
                lab = [l+1 if i and l%2 else l for i, l in enumerate(lab)]

                chk = [chk]*len(ids)

                item_t += ids
                item_l += lab
                item_c += chk

            item_t = item_t[:L]
            item_l = item_l[:L]
            item_c = item_c[:L]

            if len(item_t) > 1:
                data.append({
                    "input_ids"     : item_t,
                    "labels"        : item_l,
                    "checkers"      : item_c,
                    "attention_mask": [1]*len(item_t)
                })
    else:
        for toks, labs in zip(tokens, labels):
            assert len(toks) == len(labs)
            item_t = [C]
            item_l = [-100]

            for tok, lab in zip(toks, labs):
                if not len(tok):
                    continue

                ids = tokenizer(tok, add_special_tokens=False).input_ids
                lab = [lab]*len(ids)
                lab = [l+1 if i and l%2 else l for i, l in enumerate(lab)]

                item_t += ids
                item_l += lab
                
            item_t = item_t[:L]
            item_l = item_l[:L]

            if len(item_t) > 1:
                data.append({
                    "input_ids" : item_t,
                    "labels" : item_l,
                    "attention_mask" : [1]*len(item_t)
                })
    return data

class NER_dataset:
    def _get_data_from_file(self, filename, tokenizer, label2id:dict):
        with open(filename, 'r') as f:
            lines = f.readlines()
        tokens, labels, checks = NER_linesplit(lines=lines, label2id=label2id)
        
        self.data = NER_line2data(tokenizer, tokens, labels, checks)
        
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self, ):
        return len(self.data)