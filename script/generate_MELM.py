import sys, os
sys.path.insert(0,'..')

import argparse
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Args for Model
parser.add_argument('--model', default="M3RG-IITD/MatSciBERT", type=str)
parser.add_argument('--result_dir', required=True, type=str, help='Filename for augmentation result.')
parser.add_argument('-save_intermediate', action="store_true", help="Whether to save intermediate results.")

# Args for Augmentation Data
parser.add_argument('--data', default="sofc_slot", type=str)
parser.add_argument('--data_src', default=None, type=str)
parser.add_argument('--fold', default=0, type=int, help="5-Fold idx. default:0 means original(given) set. If else, use [1,5]. Only for SOFC.")

# Args for Running
parser.add_argument('--device', default="", type=str)
parser.add_argument('--seed', default=0, type=int, help="random seed in train. default=0")
parser.add_argument('-fp16', action="store_true", help="Whether to use FP16")

# Args for MELM Train
parser.add_argument('--mlm_batch', default=8, type=int, help="Computational batch size.")
parser.add_argument('--mlm_batch_full', default=32, type=int, help="Computational batch size.")
parser.add_argument('--mlm_lr', default=1e-5, type=float, help="learing rate in train. default=1e-5")
parser.add_argument('--mlm_epochs', default=20, type=int, help="epoch size in train. default=5")
parser.add_argument('--mlm_p', default=0.5, type=float, help="Entity masking prop in MLM training step")
parser.add_argument('--KG_p', default=0.5, type=float, help="Entity masking prop in MLM training step")

# Args for MELM Generation
parser.add_argument('--rounds', default=3, type=int, help="Number of iterations for data augmentation. default=3")
parser.add_argument('--merge_file', default=None, type=str, 
    help="If set, augmented output will be concatenated to the contents of this file. Recommand to use the source train file.")
parser.add_argument('--gen_p', default=0.5, type=float, help="Entity masking prop in generation step")
parser.add_argument('--resample_length', default='geometric', type=str, help="Re-sample strategy for the generation length.")

# Args for NER(Filter as Post-Processor) Train
parser.add_argument('--ner_src', default=None, type=str, help="If you have a pre-trained NER model, you can filter out the augmentation with it.")
parser.add_argument('--ner_batch', default=8, type=int, help="Computational batch size.")
parser.add_argument('--ner_lr', default=3e-5, type=float, help="learing rate in train. default=1e-5")
parser.add_argument('--ner_epochs', default=20, type=int, help="epoch size in train. default=5")
parser.add_argument('-use_crf', action="store_true", help="If set, CRF model is used in NER filtering.")


args = parser.parse_args()

if args.device:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# For now, disable Torch2 Dynamo : caused with https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186
os.environ["TORCHDYNAMO_DISABLE"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)

from tokenizers import AddedToken

from src.task.sofc import SOFCDataset, SOFCDataset_a
from src.task.matscholar import MatScholarDataset, MatScholarDataset_a
from src.task.metrics import NER_metrics

from src.util import callpath, makedir
from MMMM.src.augment import (
    DataCollatorForMELM,
    DataGeneratorForMELM,
    filter_MELM_augmentation,
    write_as_CONLL,
)
from src.crfmodel import BERT_CRF

if args.seed:
    set_seed(args.seed)
    np.random.seed(args.seed)

############################
### Augmentation by MELM ###
############################

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_name = args.data.lower()
assert data_name in ('sofc', 'sofc_slot', 'matscholar'), "Insert the proper database name."

if data_name == 'sofc':
    dataset = SOFCDataset_a(tokenizer, 'train', fold=args.fold, is_slot=False, augment_dir=args.data_src)
    evalset = SOFCDataset_a(tokenizer, 'valid', fold=args.fold, is_slot=False, augment_dir=args.data_src)
elif data_name == 'sofc_slot':
    dataset = SOFCDataset_a(tokenizer, 'train', fold=args.fold, is_slot=True, augment_dir=args.data_src)
    evalset = SOFCDataset_a(tokenizer, 'valid', fold=args.fold, is_slot=True, augment_dir=args.data_src)
elif data_name == 'matscholar':
    #tokenizer.add_tokens([AddedToken('<nUm>', special=True)])
    dataset = MatScholarDataset_a(tokenizer, 'train', augment_dir=args.data_src)
    evalset = MatScholarDataset_a(tokenizer, 'valid', augment_dir=args.data_src)
    
else:
    NotImplementedError

makedir(callpath(args.result_dir))
logging.basicConfig(filename=callpath(args.result_dir)+"/log.log", level=logging.INFO, filemode='w')
logging.info(f"[{model_name}] tokenizer/model is used.")

### Mask generator with [Entity] token
melm_collator = DataCollatorForMELM(tokenizer, dataset.id2label, mask_prop=args.mlm_p, mask_KG_prop=args.KG_p)#random_seed=args.seed

### Model for MELM. Special tokens(entity label) is added.
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

mlm_args = TrainingArguments(
    output_dir=callpath("./temp/"+args.device),             # output directory
    logging_dir=callpath('./temp/logs'),       # directory for storing logs
    
    num_train_epochs=args.mlm_epochs,                 # total number of training epochs
    learning_rate=args.mlm_lr,

    warmup_ratio=.1,
    max_grad_norm=1.0,
    adam_beta1=.9,
    adam_beta2=.999,
    adam_epsilon=1e-8,
    
    per_device_train_batch_size=args.mlm_batch,  # batch size per device during training
    per_device_eval_batch_size=args.mlm_batch,  # batch size per device during validation
    gradient_accumulation_steps=int(args.mlm_batch_full/args.mlm_batch),

    save_total_limit=3,
    seed=args.seed,
    fp16=args.fp16
)

trainer = Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=mlm_args,                      # training arguments, defined above
    train_dataset=dataset,              # training dataset
    eval_dataset=evalset,
    data_collator=melm_collator,
)

trainer.train()


generator = DataGeneratorForMELM(model, tokenizer, dataset, mask_prop=args.gen_p, resample_length=args.resample_length, )#random_seed=args.seed

save_name = callpath(args.result_dir + "/generated.json") if args.save_intermediate else None

augmented = generator.generate(rounds=args.rounds, result_dir=save_name)
logging.info(f"{len(augmented)} cases are generated.")

del model, trainer, melm_collator
#################################
###  Post-Process : Filtering ###
#################################
try:
    #### If you already have pre-trained NER model:
    # ALERT : The model should be based on the same tokenizer.
    model = AutoModelForTokenClassification.from_pretrained(callpath(args.ner_src), num_labels=len(dataset.id2label))
    model.resize_token_embeddings(len(tokenizer))

except:
    #### If you need to train it
    if args.use_crf:
        model = BERT_CRF(model_name, num_labels=len(dataset.id2label))
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(dataset.id2label))
    model.resize_token_embeddings(len(tokenizer))

    compute_metrics = NER_metrics(dataset.id2label)
    ner_collator = DataCollatorForTokenClassification(tokenizer)

    ner_args = TrainingArguments(
        output_dir=callpath("./temp/"+args.device),             # output directory
        logging_dir=callpath('./temp/logs'),       # directory for storing logs
        
        per_device_train_batch_size=args.ner_batch,
        per_device_eval_batch_size=args.ner_batch,
        
        warmup_ratio=.1,
        learning_rate=args.ner_lr,
        max_grad_norm=1.0,
        adam_beta1=.9,
        adam_beta2=.999,
        adam_epsilon=1e-8,
        num_train_epochs=args.ner_epochs,

        save_total_limit=3,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=ner_args,
        train_dataset=dataset,
        eval_dataset=evalset,
        data_collator=ner_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

## Eval & Log NER model : NotImplemented

save_name = callpath(args.result_dir + "/filtered.json") if args.save_intermediate else None

filtered = filter_MELM_augmentation(model, tokenizer, augmented, dataset.id2label, out_dir=save_name)

logging.info(f"After filter, {len(filtered)} cases are left.")

merge_file = callpath(args.merge_file) if args.merge_file else None

write_as_CONLL(
    data=filtered,
    out_file=callpath(args.result_dir + "/train.txt"),
    tokenizer=tokenizer,
    id2label=dataset.id2label,
    merge_file=merge_file,
)