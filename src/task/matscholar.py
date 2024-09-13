import sys, os

from torch.utils.data import Dataset
from tokenizers import AddedToken

class MatScholarData:
    def __init__(self, dir=None):
        
        if dir:
            with open(dir, 'r') as f:
                line_t = f.readlines()
        else:
            with open(os.path.join(os.path.dirname(__file__), '../../data/task/matscholar/train.txt'), 'r') as f:
                line_t = f.readlines()
        with open(os.path.join(os.path.dirname(__file__), '../../data/task/matscholar/dev.txt'), 'r') as f:
            line_d = f.readlines()
        with open(os.path.join(os.path.dirname(__file__), '../../data/task/matscholar/test.txt'), 'r') as f:
            line_r = f.readlines()

        lines = {'train_t':[], 'train_l':[], 'valid_t':[], 'valid_l':[], 'test_t':[], 'test_l':[],
        'label2id':       {'O': 0,  'B-APL': 1, 'I-APL': 2,
            'B-CMT': 3, 'I-CMT': 4, 'B-DSC': 5, 'I-DSC': 6,
            'B-MAT': 7, 'I-MAT': 8, 'B-PRO': 9, 'I-PRO':10,
            'B-SMT':11, 'I-SMT':12, 'B-SPL':13, 'I-SPL':14,
            }}

        for lines_temp, token_temp, label_temp in zip(
            (line_t, line_d, line_r),
            (lines['train_t'], lines['valid_t'], lines['test_t']),
            (lines['train_l'], lines['valid_l'], lines['test_l'])):
            
            t = {'t' : [], 'l' : []}

            for line in lines_temp:
                line = line.split()
                if len(line) != 2:
                    if len(t['t']):
                        token_temp.append(t['t'])
                        label_temp.append(t['l'])
                        t['t'], t['l'] = [], []
                    continue
                t['t'].append(line[0])
                t['l'].append(lines['label2id'][line[1]])

        self.label2id = lines['label2id']
        self.train = {
            'tokens' : lines['train_t'],
            'labels' : lines['train_l']
        }
        self.valid = {
            'tokens' : lines['valid_t'],
            'labels' : lines['valid_l']
        }
        self.test = {
            'tokens' : lines['test_t'],
            'labels' : lines['test_l']
        }
        

class MatScholarDataset(Dataset):
    def __init__(self, tokenizer, mode:str='train', augment_dir:str=None):
        mode = mode.lower()
        assert mode in ['train', 'dev', 'valid', 'validation', 'test']
        
        data = MatScholarData(dir=augment_dir)

        tokenizer.add_tokens(AddedToken("<nUm>", special=True))

        self.label2id = data.label2id
        self.id2label = {i:l for l, i in self.label2id.items()}

        if mode == 'train':
            mother = data.train
        elif mode =='test':
            mother = data.test
        else:
            mother = data.valid
        
        tokens, labels = mother['tokens'], mother['labels']
        self.items = []
        assert len(tokens) == len(labels)

        C = tokenizer.cls_token_id
        L = tokenizer.model_max_length
        
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
                self.items.append({
                    "input_ids" : item_t,
                    "labels" : item_l,
                    "attention_mask" : [1]*len(item_t)
                })

    def __getitem__(self, idx):
        return self.items[idx]
    
    def __len__(self,):
        return len(self.items)


from src.task.common import NER_dataset
LABEL2ID_matsch = { 'O': 0, 'B-APL': 1, 'I-APL': 2,
    'B-CMT': 3, 'I-CMT': 4, 'B-DSC': 5, 'I-DSC': 6,
    'B-MAT': 7, 'I-MAT': 8, 'B-PRO': 9, 'I-PRO':10,
    'B-SMT':11, 'I-SMT':12, 'B-SPL':13, 'I-SPL':14,
}
class MatScholarDataset_a(NER_dataset):
    def __init__(self, tokenizer, mode:str='train', augment_dir:str=None):
        self._get_label2id()

        if augment_dir:
            train_file = augment_dir
        else:
            #train_file = os.path.join(os.path.dirname(__file__), '../../data/task/matscholar/train.txt')
            train_file = os.path.join(os.path.dirname(__file__), '../../data/task/matKG_tag/matsch_train.txt')
        valid_file = os.path.join(os.path.dirname(__file__), '../../data/task/matscholar/dev.txt')
        test_file = os.path.join(os.path.dirname(__file__), '../../data/task/matscholar/test.txt')

        mode = mode.lower()
        assert mode in ['train', 'dev', 'valid', 'validation', 'test']
        
        if mode == 'train':
            filename = train_file
        elif mode =='test':
            filename = test_file
        else: # valid/validation/dev
            filename = valid_file
        
        tokenizer.add_tokens(AddedToken("<nUm>", special=True))

        self._get_data_from_file(filename, tokenizer, self.label2id)
        

    def _get_label2id(self, ):
        self.label2id = LABEL2ID_matsch

        self.id2label = {idx:label for label, idx in self.label2id.items()}