from random import random, gauss, seed
from tokenizers import AddedToken
from transformers import DataCollatorForTokenClassification

class DatasetForMELM:
    def __init__(self, dataset, tokenizer, mask_prop:float=0.7, max_length:int=512, random_seed:int=None):
        if random_seed:
            seed(random_seed)
        label_names = set([label[2:] for label in dataset.id2label.values() if len(label)>2])
        label_heads = [f"[{lab.upper()}]" for lab in label_names]

        tokenizer.add_tokens([AddedToken(head, special=True) for head in label_heads])

        label2head = { # label id(on entity) : header token id(on tokenizer)
            dataset.label2id[f"B-{label}"] : tokenizer.convert_tokens_to_ids(head)
            for label, head in zip(label_names, label_heads)
        }
        
        MASK = tokenizer.mask_token_id

        self.items = []
        for i in range(dataset.__len__()):
            item = dataset.__getitem__(i)
            # item : 'input_ids', 'labels', 'attention_mask'
            i = 0
            tokens, labels = [], []
            while True:
                # Catch 'B-entity' token
                ###  label = 1,3,5...           
                if (item['labels'][i] > 0) and (item['labels'][i] % 2) and (random() < mask_prop): 
                    entity_label = item['labels'][i]

                    tokens.append(label2head[entity_label])
                    labels.append(-1)

                    tokens.append(MASK)
                    labels.append(item['input_ids'][i])

                    j = i + 1
                    while(item['labels'][j] == entity_label+1):
                        tokens.append(MASK)
                        labels.append(item['input_ids'][j])
                        j += 1

                        if j >= len(item['labels']):
                            break

                    tokens.append(label2head[entity_label])   
                    labels.append(-1)
                    i = j
                        
                else:
                    tokens.append(item['input_ids'][i])
                    labels.append(-1)
                    i += 1
                if i >= len(item['input_ids']):
                    break
            
            tokens, labels = tokens[:max_length], labels[:max_length]
            self.items.append({ 
                'input_ids'     : tokens,
                'labels'        : labels,
                'attention_mask': [1]*len(tokens),
            })
        return
    
    def __len__(self, ):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class DataCollatorForMELM:
    def __init__(self,
            tokenizer,
            id2label,
            mask_prop:float=0.7,
            mask_KG_prop:float=0,
            max_length=512,
            mask_with_gaussian=False,
            keep_original_labels=False,
            random_seed:int=None):
        if random_seed:
            seed(random_seed)
        
        label_names = [label for label in id2label.values() if len(label)>2]
        label_heads = [f"[{lab.upper()}]" for lab in label_names]

        tokenizer.add_tokens([AddedToken(head, special=True) for head in label_heads])

        label2id = {v:k for k, v in id2label.items()}

        self.label2head = { # label id(on entity) : header token id(on tokenizer)
            label2id[label] : tokenizer.convert_tokens_to_ids(head)
            for label, head in zip(label_names, label_heads)
        }
        
        self.max_length = max_length
        self.mask_prop = mask_prop
        self.mask_KG_prop = mask_KG_prop
        self.mask_with_gaussian = mask_with_gaussian
        self.keep_original_labels = keep_original_labels
        self.MASK = tokenizer.mask_token_id

        self.Collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    def get_mask_prop(self, labels, idx):
        if self.mask_with_gaussian:
            k=1
            idx = int(idx)
            while (idx+k<len(labels)) and (labels[idx+k] == labels[idx]+1):
                k += 1

            p = gauss(self.mask_prop, 1/k**2)
            while p<=0 or 1<=p :
                p = gauss(self.mask_prop, 1/k**2)
            return p
        else:
            return self.mask_prop

    def __call__(self, items):
        items_out = []

        for item in items:
            i = 0
            tokens, labels = [], []

            while i<len(item['input_ids']):
                if 'checkers' in item:
                    checker_KG = self.mask_KG_prop * item['checkers'][i]
                else:
                    checker_KG = 0
                # Catch 'B-entity' token
                ###  label = 1,3,5...           
                if (item['labels'][i] > 0) and \
                        (random() < self.get_mask_prop(item['labels'], i)) or \
                        (random() < checker_KG): 
                    entity_label = item['labels'][i]

                    tokens.append(self.label2head[entity_label])
                    labels.append(-100)

                    tokens.append(self.MASK)
                    labels.append(item['input_ids'][i])

                    tokens.append(self.label2head[entity_label])   
                    labels.append(-100)
                    
                    i += 1
                    
                    """j = i + 1
                    while(j<len(item['labels'])) and (item['labels'][j] == entity_label+1):
                        tokens.append(self.MASK)
                        labels.append(item['input_ids'][j])
                        j += 1

                        #if j >= len(item['labels']):
                        #    break

                    tokens.append(self.label2head[entity_label])   
                    labels.append(-100)
                    i = j
                    """    
                else:
                    tokens.append(item['input_ids'][i])
                    labels.append(-100)
                    i += 1
                
                
            tokens, labels = tokens[:self.max_length], labels[:self.max_length]
            
            out = { 
                'input_ids'     : tokens,
                'labels'        : labels,
                'attention_mask': [ 0 if tok == self.MASK else 1 for tok in tokens],
            }
            if self.keep_original_labels:
                out['ori_labels'] = item['labels']
            
            items_out.append(out)
        return self.Collator(items_out)

from random import choices
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

class DataGeneratorForMELM:
    def __init__(self, model, tokenizer, dataset, mask_prop:float=0.5, max_length:int=512, random_seed:int=None):
        if random_seed:
            seed(random_seed)

        collator = DataCollatorForMELM(tokenizer, dataset.id2label, mask_prop=mask_prop, max_length=max_length, keep_original_labels=True)
        self.loader = DataLoader(dataset, batch_size=1, collate_fn=collator)
        # batch=1 due to the tensor shape incompatibility error.
        self.model = model

        self.label2head = collator.label2head
        self.PAD = tokenizer.pad_token_id

    ## ToDo : generation with Batch
    def generate(self, result_dir:str=None, rounds:int=3):
        augmented = []

        self.model.eval()

        bar = tqdm(total=self.loader.__len__() * rounds, desc="data generating")
        for _ in range(rounds):
            for data in self.loader:
                bar.update()
                for key, value in data.items():
                    data[key] = value.to(self.model.device)

                ori_labels =  data.pop('ori_labels').squeeze().to('cpu').tolist()
                with torch.no_grad():
                    out = self.model(**data)

                for t_in, t_out in zip(data['input_ids'].to('cpu'), out.logits.to('cpu')):
                
                    result, added = [], []
                    i = 0
                    while i < len(t_in):
                        tok = t_in[i]

                        if tok == self.PAD:
                            i += 1
                            continue

                        if tok not in self.label2head.values():
                            result.append(tok.tolist())
                            added.append(0)
                        else:
                            j = i+1
                            while (j < len(t_in)) and (t_in[j] != tok):
                                prop, topk_idx = t_out[j].topk(5)

                                result.append(choices(topk_idx, )[0].tolist()) #weights=prop
                                added.append(1)

                                j += 1
                            i = j
                        
                        i += 1

                        if len(result)==len(ori_labels):
                            break
                
                ori_labels = ori_labels[:len(result)]
                assert len(result) == len(ori_labels) , f"{len(result)} != {len(ori_labels)}"#== len(added) 
                augmented.append({
                    'input_ids':result,
                    'labels':ori_labels,
                    'added_tokens':added,
                })
        
        if result_dir:
            with open(result_dir, 'w') as f:
                json.dump(augmented, f)
        
        return augmented
            

#################
def filter_MELM_augmentation(model, tokenizer, augmented, id2label, out_dir=None):
        
    collator = DataCollatorForTokenClassification(tokenizer)
    
    if isinstance(augmented, str):
        with open(augmented, 'r') as f:
            augmented = json.load(f,)
    
    is_crf = bool("CRF" in model._get_name())

    items_out = []
    model.eval()

    for item in tqdm(augmented, desc="filtering..."):

        is_generated = item.pop('added_tokens')
        labels = item['labels']

        input = collator([item])

        for key, value in input.items():
            input[key] = value.to(model.device)

        with torch.no_grad():
            if is_crf:
                input.pop('labels')
                logits = model(**input)[0]
            else:
                logits = model(**input).logits.to('cpu').argmax(-1).squeeze().tolist()

        flag = True
        for i, (sign, pred, answ) in enumerate(zip(is_generated, logits, labels)):
            if sign:
                if pred != answ:
                    flag = False
                    break
        
        if flag:
            item['added_tokens'] = is_generated
            items_out.append(item)
    if out_dir:
        with open(out_dir, 'w') as f:
            json.dump(items_out, f)
    return items_out

def write_as_CONLL(data, out_file, tokenizer, id2label, merge_file:str=None):
    """
    write a CONLL format txt file with augmented data.
    + data      : list of dictionary, return of funct 'filter_MELM_augmentation'.)
    + out_file  : str, result filename.
    + tokenizer : BertTokenizer.
    + id2label  : dict, 'id2label' from original dataset.
    + merge_file: str(Optional), filename of original CONLL file.
                   if used, data will be concatenated with the original data.
    """
    
    with open(out_file, 'w') as f:

        if merge_file:
            with open(merge_file, 'r') as g:
                lines = g.readlines()
            f.writelines(lines)

        for item in data:

            token, label = "", ""

            for tok, lab, mask in zip(item['input_ids'], item['labels'], item['added_tokens']):
                if lab == -100:
                    continue
                
                tok = tokenizer.convert_ids_to_tokens(tok)
                lab = id2label[lab]

                if tok in ("[CLS]", "[SEP]"):
                    continue

                if (tok.startswith("##")) and (lab.startswith("I-") or lab=="O"):
                    token += tok[2:]
                else:
                    if token:
                        f.write(f"{token} {label}\n")
                    token = tok
                    label = lab
            
            # write last line
            f.write(f"{token} {label}\n\n")
    return