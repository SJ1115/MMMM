import sys, os

from torch.utils.data import Dataset


class SOFCData:
    def __init__(self, is_slot=False, fold:int=0, dir=None):
        task_str = "slot_" if is_slot else ""

        if dir:
            with open(dir, 'r') as f:
                line_t = f.readlines()
        else:
            with open(os.path.join(os.path.dirname(__file__), '../../data/task/sofc_process/' + task_str + 'train.txt'), 'r') as f:
                line_t = f.readlines()
        with open(os.path.join(os.path.dirname(__file__), '../../data/task/sofc_process/' + task_str + 'dev.txt'), 'r') as f:
            line_d = f.readlines()
        with open(os.path.join(os.path.dirname(__file__), '../../data/task/sofc_process/' + task_str + 'test.txt'), 'r') as f:
            line_r = f.readlines()

        if is_slot:
            label2id = {'O': 0,
            'B-anode_material': 1,          'I-anode_material': 2,
            'B-cathode_material': 3,        'I-cathode_material': 4,
            'B-conductivity': 5,            'I-conductivity': 6,
            'B-current_density': 7,         'I-current_density': 8,
            'B-degradation_rate': 9,        'I-degradation_rate': 10,
            'B-device': 11,                 'I-device': 12,
            'B-electrolyte_material': 13,   'I-electrolyte_material': 14,
            'B-fuel_used': 15,              'I-fuel_used': 16,
            'B-interlayer_material': 17,    'I-interlayer_material': 18,
            'B-open_circuit_voltage': 19,   'I-open_circuit_voltage': 20,
            'B-power_density': 21,          'I-power_density': 22,
            'B-resistance': 23,             'I-resistance': 24,
            'B-support_material': 25,       'I-support_material': 26,
            'B-thickness': 27,              'I-thickness': 28,
            'B-time_of_operation': 29,      'I-time_of_operation': 30,
            'B-voltage': 31,                'I-voltage': 32,
            'B-working_temperature': 33,    'I-working_temperature': 34,
            }
        else:
            label2id = {'O': 0,
            'B-DEVICE': 1,      'I-DEVICE': 2,
            'B-EXPERIMENT': 3,  'I-EXPERIMENT': 4,
            'B-MATERIAL': 5,    'I-MATERIAL': 6,
            'B-VALUE': 7,       'I-VALUE': 8,
            }

        lines = {'train_t':[], 'train_l':[], 'valid_t':[], 'valid_l':[], 'test_t':[], 'test_l':[],
        'label2id': label2id}

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

        lines['train_t'], lines['train_l'], lines['valid_t'], lines['valid_l'] = self.__mix_fold(
            lines['train_t'], lines['train_l'], lines['valid_t'], lines['valid_l'],
            fold=fold
        )

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
    
    def __mix_fold(self, x_1, y_1, x_2, y_2, fold:int):
        assert fold in (0, 1, 2, 3, 4, 5), "'fold' should be an int among [0,5], 0 means no fold(default)."

        if fold == 0:
            return x_1, y_1, x_2, y_2
        
        ## else:        
        res = fold-1
        x_cat = x_1 + x_2
        y_cat = y_1 + y_2

        x_a, y_a, x_b, y_b = [], [], [], []
        for i, (x, y) in enumerate(zip(x_cat, y_cat)):
            if i % 5 == res:
                x_b.append(x)
                y_b.append(y)
            else:
                x_a.append(x)
                y_a.append(y)
        
        return x_a, y_a, x_b, y_b

class SOFCDataset(Dataset):
    def __init__(self, tokenizer, mode:str='train', is_slot=True, fold=0, max_token_length:int=512, augment_dir:str=None):
        mode = mode.lower()
        assert mode in ['train', 'dev', 'valid', 'validation', 'test']
        
        data = SOFCData(is_slot=is_slot, fold=fold, dir=augment_dir)

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
        if max_token_length:
            L = max_token_length
        else:
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

class SOFCDataset_a(NER_dataset):
    def __init__(self, tokenizer, mode:str='train', is_slot:bool=False, fold:int=0, augment_dir:str=None):
        self._get_label2id(is_slot)
        task_str = "slot_" if is_slot else ""

        if augment_dir:
            train_file = augment_dir
        else:
            #train_file = os.path.join(os.path.dirname(__file__), '../../data/task/sofc_process/' + task_str + 'train.txt')
            train_file = os.path.join(os.path.dirname(__file__), '../../data/task/matKG_tag/' + task_str + 'train.txt')
        valid_file = os.path.join(os.path.dirname(__file__), '../../data/task/sofc_process/' + task_str + 'dev.txt')
        test_file = os.path.join(os.path.dirname(__file__), '../../data/task/sofc_process/' + task_str + 'test.txt')

        mode = mode.lower()
        assert mode in ['train', 'dev', 'valid', 'validation', 'test']
        
        if mode == 'train':
            filename = train_file
        elif mode =='test':
            filename = test_file
        else: # valid/validation/dev
            filename = valid_file
        
        self._get_data_from_file(filename, tokenizer, self.label2id)
        

    def _get_label2id(self, is_slot):
        if is_slot:
            self.label2id = {'O': 0,
            'B-anode_material': 1,          'I-anode_material': 2,
            'B-cathode_material': 3,        'I-cathode_material': 4,
            'B-conductivity': 5,            'I-conductivity': 6,
            'B-current_density': 7,         'I-current_density': 8,
            'B-degradation_rate': 9,        'I-degradation_rate': 10,
            'B-device': 11,                 'I-device': 12,
            'B-electrolyte_material': 13,   'I-electrolyte_material': 14,
            'B-fuel_used': 15,              'I-fuel_used': 16,
            'B-interlayer_material': 17,    'I-interlayer_material': 18,
            'B-open_circuit_voltage': 19,   'I-open_circuit_voltage': 20,
            'B-power_density': 21,          'I-power_density': 22,
            'B-resistance': 23,             'I-resistance': 24,
            'B-support_material': 25,       'I-support_material': 26,
            'B-thickness': 27,              'I-thickness': 28,
            'B-time_of_operation': 29,      'I-time_of_operation': 30,
            'B-voltage': 31,                'I-voltage': 32,
            'B-working_temperature': 33,    'I-working_temperature': 34,
            }
        else:
            self.label2id = {'O': 0,
            'B-DEVICE': 1,      'I-DEVICE': 2,
            'B-EXPERIMENT': 3,  'I-EXPERIMENT': 4,
            'B-MATERIAL': 5,    'I-MATERIAL': 6,
            'B-VALUE': 7,       'I-VALUE': 8,
            }

        self.id2label = {idx:label for label, idx in self.label2id.items()}