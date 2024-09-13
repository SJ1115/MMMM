import sys, os
sys.path.insert(0,'..')

import argparse
from src.util import callpath, makedir

from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE, TransR
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--src_dir', required=True, type=str, help='Filename for augmentation result.')
#parser.add_argument('--result_dir', required=True, type=str, help='Filename for augmentation result.')
parser.add_argument('--epochs', default=100, type=int, help="Computational batch size.")
parser.add_argument('--device', default="", type=str, help="")

args = parser.parse_args()

if args.device:
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=args.device
	use_gpu = True
	# For now, disable Torch2 Dynamo : caused with https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186
	os.environ["TORCHDYNAMO_DISABLE"] = '1'
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
else:
	use_gpu = False



# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = callpath("./data/matKG/embed/"), 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
#test_dataloader = TestDataLoader(
#	in_path = callpath("data/matKG/embed/"),
#	sampling_mode = 'link')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200,
	dim_r = 200,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

makedir(callpath("matkg_embedding"))

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = use_gpu)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters(callpath("matkg_embedding/transr_transe.json"))

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = args.epochs, alpha = 1.0, use_gpu = use_gpu)
trainer.run()
transr.save_checkpoint(callpath('./matkg_embedding/transr.ckpt'))
