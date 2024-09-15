import sys, os

from torch.utils.data import Dataset
from src.task.common import NER_dataset

class SOFCDataset(NER_dataset):
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
