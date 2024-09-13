import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from copy import copy
from tqdm import tqdm

PROMPT = "Generate a sentence with these words:"
TAGGER = '/'
SEPARATOR = ','

get_tag = lambda x:x[2:] if len(x)>2 else None

def equal(seq1, seq2):
    if len(seq1) != len(seq2):
        return False
    for s1, s2 in zip(seq1, seq2):
        if s1 != s2:
            return False
    return True

def seqSearch(subseq, seq):
    for i in range(len(seq)-len(subseq)):
        if equal(subseq, seq[i:i+len(subseq)]):
            return i
    return -1

###
class E2TDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length:int=128):
        """
        dataset : task(NER) dataset. refer 'task.sofc.py' or 'task.matscholar.py'
        tokenizer : tokenizers tokenizer(I wrote it based on 'google-t5/t5-small')
        max_length : 
        """
        self.prompt = tokenizer(PROMPT).input_ids[:-1]
        self.separator = tokenizer.convert_tokens_to_ids([SEPARATOR])
        self.tagger = tokenizer.convert_tokens_to_ids([TAGGER])

        self.items = []
        self.entities = []
        self.augmented_entities = []

        self.id2label = dataset.id2label
        self.tokenizer = tokenizer
        self.max_length = max_length

        for idx in range(dataset.__len__()):
            item = dataset.__getitem__(idx)

            newitem = {}

            inputs = []
            entities = []

            i = 1
            while i < len(item['input_ids']):
                inp, lab = item['input_ids'][i], item['labels'][i]

                if lab > 0: # is Entity
                    word = [inp]

                    j = i+1
                    while j < len(item['input_ids']) and item['labels'][j] == lab+1 :
                        word.append(item['input_ids'][j])
                        j += 1
                    
                    inputs += word
                    inputs += self.tagger
                    inputs += tokenizer(get_tag(dataset.id2label[lab])).input_ids[:-1]

                    entities.append(word + [lab])
                    #### Shape of Entity ####
                    # [tokens for entity word(len:len(tokens of word)) : label(len:1)]
                    #########################

                    inputs += self.separator

                    i = j
                else:
                    i += 1

                if len(inputs) >= self.max_length:
                    break
            inputs = inputs[:-1] ## remove last ',' token

            newitem = {}
            newitem['input_ids'] = self.prompt+inputs
            newitem['labels'] = item['input_ids'][1:i]

            self.items.append(newitem)
            self.entities.append(entities)

    def __len__(self,):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

    def augment_entities(self, seed:int=None):
        if seed:
            random.seed(seed)
        
        all_ents = [tuple(ent) for ents in self.entities for ent in ents]

        all_ents = list(set(all_ents))
        all_ents = [list(ent) for ent in all_ents]

        #self.augmented_entities
        for cur_ents in self.entities:
            if len(cur_ents):
                ### ADD
                new_ents = copy(cur_ents)
                new_i = random.randint(0, len(cur_ents)-1)

                tar_ent = new_ents[new_i]

                candidates = [ent for ent in all_ents if ent[-1]==tar_ent[-1]]
                added_ent = random.choice(candidates)

                new_ents.insert(new_i, added_ent)
                self.augmented_entities.append(new_ents)

                ### DELETE
                new_ents = copy(cur_ents)
                new_i = random.randint(0, len(cur_ents)-1)

                new_ents.pop(new_i)
                self.augmented_entities.append(new_ents)

                ### REPLACE
                new_ents = copy(cur_ents)
                new_i = random.randint(0, len(cur_ents)-1)

                tar_ent = new_ents[new_i]

                candidates = [ent for ent in all_ents if ent[-1]==tar_ent[-1]]
                added_ent = random.choice(candidates)

                new_ents[new_i] = added_ent
                self.augmented_entities.append(new_ents)

                ### SWAP
                if len(cur_ents)>1:
                    new_ents = copy(cur_ents)
                    new_i = random.randint(0, len(cur_ents)-1)

                    new_j = random.randint(0, len(cur_ents)-1)
                    while(new_j == new_i):
                        new_j = random.randint(0, len(cur_ents)-1)

                    new_ents[new_i], new_ents[new_j] = new_ents[new_j], new_ents[new_i]
                    self.augmented_entities.append(new_ents)

    
    def set_to_augment(self,):
        assert len(self.augmented_entities), "Call augment_entities() first."

        if len(self.augmented_entities) == len(self.items):
            # already processed
            return
        
        self.original_items = self.items
        self.items = []
        
        for ents in self.augmented_entities:
            inputs = copy(self.prompt)
            
            if len(ents):
                for i, ent in enumerate(ents):
                    if i:
                        inputs += self.separator
                    inputs += ent[:-1]
                    inputs += self.tagger
                    inputs += self.tokenizer(get_tag(self.id2label[ent[-1]])).input_ids[:-1]

                self.items.append({
                    "input_ids":inputs,
                    "entity":ents
                })

    def augment_w_check(self, model, batch_size=1, use_cuda=True, collator=None, sample_cases=0):
        if collator == None:
            collator = DataCollatorForSeq2SeqModeling(self.tokenizer)
        
        device = "cuda" if use_cuda else "cpu"
        pad = self.tokenizer.pad_token_id

        model.eval()
        self.augmented_results = []
        with torch.no_grad():
            for item in tqdm(DataLoader(self.items, batch_size=batch_size, collate_fn=collator), desc="generating..."):
                
                input = item['input_ids'].to(device)
                ent_lst = item['entity']

                preds = model.generate(input, max_length=self.max_length).tolist()

                for ents, pre in zip(ent_lst, preds):
                    #print(ents)
                    #print(pre)
                    #break
                    
                    
                    pre = [p for p in pre if p != pad]
                    labs = [0] * len(pre)
                    s = 0
                    for ent in ents:
                        ent_tok, ent_lab = ent[:-1], ent[-1]
                        #print("".join(tokenizer.convert_ids_to_tokens(ent[:-1])))
                        i = seqSearch(ent_tok, pre)
                        if i != -1:
                            labs[i] = ent_lab
                            for j in range(1, len(ent_tok)):
                                labs[i+j] = ent_lab + 1
                            s += 1
                    
                #return {"input_ids":pre, "labels":labs}
                
                    if s:
                        self.augmented_results.append({"input_ids":pre, "labels":labs})
                if sample_cases:
                    if len(self.augmented_results) == int(sample_cases):
                        break
        return self.augmented_results

class DataCollatorForSeq2SeqModeling:
    """
    Data collator used for language modeling, rephrased from:
    https://colab.research.google.com/drive/1OFiiaBA40EdReaeKB6MHnC7Dvl8u5a2T?usp=sharing
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids = [e["input_ids"] for e in examples]
        input_ids = self._tensorize_batch(input_ids)
    
        out = {"input_ids": input_ids}
        if 'labels' in examples[0]:
            labels = [e["labels"] for e in examples]
            labels = self._tensorize_batch(labels)
            out["labels"] = labels
        
        if 'entity' in examples[0]:
            out["entity"] = [example['entity'] for example in examples]
        return out
    
    def _tensorize_batch(self, examples):
        examples = [torch.tensor(x, dtype=int) for x in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        

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
    
    with open(out_file, 'w', encoding="utf-8") as f:

        if merge_file:
            with open(merge_file, 'r+', encoding="utf-8") as g:
                lines = g.readlines()
            f.writelines(lines)

        for item in data:

            token, label = "", ""

            for tok, lab in zip(item['input_ids'], item['labels'],):
                if lab == -100:
                    continue
                
                tok = tokenizer.convert_ids_to_tokens(tok)
                lab = id2label[lab]

                if tok in ("<pad>"):
                    continue
                
                if tok.startswith("▁"):
                    if token:
                        f.write(f"{token} {label}\n")
                    token = tok[1:]
                    label = lab
                else:
                    if ((lab.startswith("I-") and get_tag(label)==get_tag(lab)) or \
                        (lab=="O" and label=="O")):
                        token += tok
                    else:
                        f.write(f"{token} {label}\n")
                        token = tok
                        label = lab

            
            # write last line
            f.write(f"{token} {label}\n\n")
    return