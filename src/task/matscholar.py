import sys, os

from torch.utils.data import Dataset
from tokenizers import AddedToken

from src.task.common import NER_dataset
LABEL2ID_matsch = { 'O': 0, 'B-APL': 1, 'I-APL': 2,
    'B-CMT': 3, 'I-CMT': 4, 'B-DSC': 5, 'I-DSC': 6,
    'B-MAT': 7, 'I-MAT': 8, 'B-PRO': 9, 'I-PRO':10,
    'B-SMT':11, 'I-SMT':12, 'B-SPL':13, 'I-SPL':14,
}
class MatScholarDataset(NER_dataset):
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
