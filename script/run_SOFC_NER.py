import sys, os
sys.path.insert(0,'..')

import argparse
import json
from tqdm import tqdm
from numpy.random import randint
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Args for Model
parser.add_argument('--model', default="M3RG-IITD/MatSciBERT", type=str)
parser.add_argument('-crf', action="store_true", help="If set, CRF layer will be appended on the top of the NER model.")

parser.add_argument('--init_new', default="", type=str, choices=['mean', 'mlm', 'dict'], help="Initialization method for new terms. 'dict' is for mean of subwords(dictionary), 'mlm' for MLM, 'mean' for mean of subwords(itself). 'dict' needs --dict arguments.")
parser.add_argument('--topk', default=0, type=int, help="number of candidate to find the embedding for new abbreviation token. Used only if 'add_abbbr'=True. 0(default) means all candidates(from softmax) are used.")
parser.add_argument('--sump', default=0, type=float, help="threshold for the sum of candidates props, to find the embedding for new abbreviation token. Used only if 'add_abbbr'=True. 0(default) means all candidates(from softmax) are used.")
parser.add_argument('--dict', default=None, type=str, help="directory of dictionary.csv for the meaning of abbr.")

# Args for Contrastive Learning
parser.add_argument('-contrast', action="store_true", help="If set, contrastive learing is conceeded. You have to use -add_abbr with.")
parser.add_argument('-use_cosine_similarity', action="store_true", help="If set, cosine similarity is used in contrastive learning. If else, euclidean distance is used")
parser.add_argument('-use_O', action="store_true", help="If set, cosine similarity is used in contrastive learning. If else, euclidean distance is used")
parser.add_argument('--contrastive_lambda', default=1, type=float, help="learning rate for additional, contrastive loss.")
parser.add_argument('--contrastive_tau', default=1, type=float, help="temperature in contrastive loss.")

# Args for Task & Data
parser.add_argument('-slot', action="store_true", help="If set, the target task would be SOFC-'SLOT'")
parser.add_argument('--fold', default=0, type=int, help="fold id for 5-Fold validation")
parser.add_argument('--augment', default=None, type=str, help="Source dir of data when augmented data is used.")
parser.add_argument('-detail', action="store_true", help="If set, the label-wise score(test) will be added.")

# Args for Running
parser.add_argument("--result_json", default="SOFC_result.json", help="The path of the result json file.")
parser.add_argument('--name', default=None, type=str, help='Nickname of model. It will be the row name in result_json')
parser.add_argument('--device', default="", type=str)
parser.add_argument('--seed', default=0, type=int, help="random seed in train. default=0")
parser.add_argument('-fp16', action="store_true", help="Whether to use FP16")
# Args for Hyperparameters
parser.add_argument('--batch_size', default=16, type=int, help="Computational batch size.")
parser.add_argument('--lr', default=5e-5, type=float, help="learing rate in train. default=1e-5")
parser.add_argument('--epochs', default=40, type=int, help="epoch size in train. default=5")

args = parser.parse_args()

if args.device:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# For now, disable Torch2 Dynamo : caused with https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186
os.environ["TORCHDYNAMO_DISABLE"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)

from src.task.sofc import SOFCDataset
from src.task.metrics import NER_metrics
from src.util import callpath, make_model_keep_hidden
from src.crfmodel import BERT_CRF

#from src.embed import init_new_by_MLM, init_new_by_dict, init_new_by_MoS
#from src.contrastive_token import TrainerForTokenContrastive, NTXentLoss, DataCollatorForTokenContrast


model_name = args.model
data_name = 'sofc_slot' if args.slot else 'sofc'

if args.name:
    nickname = args.name
else:
    nickname = args.model


tokenizer = AutoTokenizer.from_pretrained(model_name)

trainset = SOFCDataset(tokenizer, mode='train', is_slot=args.slot, fold=args.fold, augment_dir=args.augment)
validset = SOFCDataset(tokenizer, mode='valid', is_slot=args.slot, fold=args.fold, augment_dir=args.augment)
testset = SOFCDataset(tokenizer, mode='test', is_slot=args.slot)

if args.crf:
    model = BERT_CRF(model_name=model_name, num_labels=len(trainset.id2label))
else:
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(trainset.id2label))

ner_metrics = NER_metrics(id2tag = trainset.id2label)

if args.seed:
    set_seed(args.seed)

training_args = TrainingArguments(
    output_dir=callpath(f'./temp/{args.device}'),             # output directory
    logging_dir=callpath('./temp/logs'),       # directory for storing logs
    
    num_train_epochs=args.epochs,                 # total number of training epochs
    warmup_ratio=.1,
    learning_rate=args.lr,
    max_grad_norm=1.0,
    adam_beta1=.9,
    adam_beta2=.999,
    adam_epsilon=1e-8,
    
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,  # batch size per device during validation
    #save_strategy='epoch',
    #load_best_model_at_end=True,
    #metric_for_best_model='accuracy',
    #greater_is_better=True,
    save_total_limit=3,
    seed=args.seed,
    fp16=args.fp16,

    remove_unused_columns=False
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=trainset,              # training dataset
    eval_dataset=validset,
    compute_metrics=ner_metrics,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    )

trainer.train()

eval_score = trainer.evaluate()
test_score = trainer.evaluate(testset)

if args.detail:
    ner_metrics = NER_metrics(id2tag = trainset.id2label, detail=True)

    tester = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=trainset,              # training dataset
    eval_dataset=validset,
    compute_metrics=ner_metrics,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    detailed = tester.evaluate()
    detailed_eval = {}
    for key in detailed.keys():
        if key not in (
            'eval_loss', 'eval_overall_f1', 'eval_overall_precision', 'eval_overall_recall', 'eval_overall_accuracy', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second'
        ):
            detailed_eval[key[5:]] = detailed[key]['f1']

    detailed = tester.evaluate(testset)
    detailed_test = {}
    for key in detailed.keys():
        if key not in (
            'eval_loss', 'eval_overall_f1', 'eval_overall_precision', 'eval_overall_recall', 'eval_overall_accuracy', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second'
        ):
            detailed_test[key[5:]] = detailed[key]['f1']


if os.path.exists(callpath(args.result_json)):
    with open(callpath(args.result_json), 'r') as f:
        final_result = json.load(f,)
else:
    final_result = []

score = {
#    'eval': eval_score,
#    'test': test_score,
}
score['model'] = nickname
score['seed'] = args.seed
score['lr'] = args.lr
score['fold'] = args.fold

score['eval_acc'] = eval_score['eval_accuracy']
score['eval_f1'] = eval_score['eval_f1']
score['eval_precision'] = eval_score['eval_precision']
score['eval_recall'] = eval_score['eval_recall']

score['test_acc'] = test_score['eval_accuracy']
score['test_f1'] = test_score['eval_f1']
score['test_precision'] = test_score['eval_precision']
score['test_recall'] = test_score['eval_recall']

if args.detail:
    score['eval_detail'] = detailed_eval
    score['test_detail'] = detailed_test

final_result.append(score)
        
with open(callpath(args.result_json), 'w') as f:
    json.dump(final_result, f)