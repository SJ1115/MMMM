import os
import random
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    DataCollatorForWholeWordMask,
)

def callpath(filename):
    return os.path.join(os.path.dirname(__file__), '..', filename)

def makedir(dir):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(dir):
        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(dir)

from torch import nn
class ClassifierWithHidden(nn.Module):
    def __init__(self, main_module):
        super(ClassifierWithHidden, self).__init__()

        self.module = main_module
        self.hidden_return = None

    def forward(self, x):
        self.hidden_return = x
        return self.module(x)

def make_model_keep_hidden(model:nn.Module):
    model.classifier = ClassifierWithHidden(model.classifier)
    return

def show_prediction_results(
    out_file,
    predictions,
    labels,
    input_set,
    tokenizer,
    id2tag,
    ignore_id:int=-100,):
    if predictions.shape != labels.shape:
        predictions = predictions.argmax(-1)
    assert predictions.shape == labels.shape, "Inconsistency between predictions & labels"
    
    assert predictions.shape[0] == input_set.__len__(), "Inconsistency between dataset & predictions."

    with open(out_file, 'w') as f:
        f.write("token,prediction,label\n")
        for i, (preds, labels) in enumerate(zip(predictions, labels)):
            tokens = tokenizer.convert_ids_to_tokens(input_set.__getitem__(i)['input_ids'])

            for t , p, l in zip(tokens, preds, labels):
                if l != ignore_id:
                    t = "','" if t == ',' else t
                    f.write(f"{t}, {id2tag[p]}, {id2tag[l]}\n")
            f.write("\n")
    return

from typing import List
class MyCollatorWithABBR(DataCollatorForWholeWordMask):
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        must_indexes = [] ## Added
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if token.startswith("<ab:"): ## Added
                must_indexes.append([i])
            elif len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
            

        # Added tokens MUST have the higher priority
        random.shuffle(cand_indexes)
        cand_indexes = must_indexes + cand_indexes ## Added

        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels