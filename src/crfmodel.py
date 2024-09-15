import torch
from torch import nn, cuda
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoModelForTokenClassification,
    PreTrainedModel,
    AutoConfig,
)

from torchcrf import CRF


model_revision = 'main'


class BIO_Tag_CRF(CRF):
    def __init__(self, num_tags: int, device=None, batch_first: bool = False):
        super(BIO_Tag_CRF, self).__init__(num_tags=num_tags, batch_first=batch_first)
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        start_transitions = self.start_transitions.clone().detach()
        transitions = self.transitions.clone().detach()
        assert num_tags % 2 == 1
        #num_uniq_labels = (num_tags - 1) // 2
        #for i in range(num_uniq_labels, 2 * num_uniq_labels):
        for i in range(2, num_tags, 2): # I-tok: 2, 4, ...
            start_transitions[i] = -10000
            for j in range(0, num_tags):
                #if j == i or j + num_uniq_labels == i: continue
                if j == i or j == i-1: continue
                transitions[j, i] = -10000
        self.start_transitions = nn.Parameter(start_transitions)
        self.transitions = nn.Parameter(transitions)

    def forward(self, logits, labels, masks):
        
        new_logits, new_labels, new_attention_mask = [], [], []
        for logit, label, mask in zip(logits, labels, masks):
            new_logits.append(logit[mask])
            new_labels.append(label[mask])
            new_attention_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.uint8, device=self.device))
        
        padded_logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)
        padded_attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        loss = -super(BIO_Tag_CRF, self).forward(padded_logits, padded_labels, mask=padded_attention_mask, reduction='mean')
        
        if self.training:
            return (loss, )
        else:
            out = self.decode(padded_logits, mask=padded_attention_mask)
            assert(len(out) == len(labels))
            out_logits = torch.zeros_like(logits)
            for i in range(len(out)):
                k = 0
                for j in range(len(labels[i])):
                    if labels[i][j] == -100: continue
                    out_logits[i][j][out[i][k]] = 1.0
                    k += 1
                assert(k == len(out[i]))
            return (loss, out_logits, )

class BERT_CRF(PreTrainedModel):
    def __init__(self, model_name, num_labels, device=None,):
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        
        super(BERT_CRF, self).__init__(self.config)
        
        if not device:
            device="cuda" if cuda.is_available() else 'cpu'
        
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)
        self.crf = BIO_Tag_CRF(num_labels, device, batch_first=True)


    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,).logits

        if labels != None:
            masks = (labels != -100)
            return self.crf(logits, labels, masks)
        else:
            return self.crf.decode(logits)
    
    def resize_token_embeddings(self, args, *kwargs):
        self.encoder.resize_token_embeddings(args, *kwargs)
