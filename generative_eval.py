import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data

#################
from core.data.cifar10g import GENERATIVECIFAR10
#################

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train
from core.utils import Trainer
from core.utils import seed
from core import setup

parse = parser_train()
args = parse.parse_args()

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
TMP = os.path.join(args.tmp_dir, args.desc)
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
if args.exp and not os.path.exists(TMP):
    os.makedirs(TMP, exist_ok=True)
    print('Tmp Dir: ', TMP)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

if 'imagenet' in args.data:
    setup.setup_train(DATA_DIR)
    setup.setup_val(DATA_DIR)
    args.data_dir = os.environ['TMPDIR']
    DATA_DIR = os.path.join(args.data_dir, args.data)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_STD_EPOCHS = args.num_std_epochs
NUM_ADV_EPOCHS = args.num_adv_epochs
NUM_SAMPLES_EVAL = args.num_samples_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_STD_EPOCHS = 1
    NUM_ADV_EPOCHS = 1

# To speed up training
if args.model in ['wrn-34-10', 'wrn-34-20'] or 'swish' in args.model or 'imagenet' in args.data:
    torch.backends.cudnn.benchmark = True

# Load data

seed(args.seed)
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True,
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction
)
num_train_samples = len(train_dataset)
num_test_samples = len(test_dataset)

train_indices = np.random.choice(num_train_samples, NUM_SAMPLES_EVAL, replace=False)
test_indices = np.random.choice(num_test_samples, NUM_SAMPLES_EVAL, replace=False)

pin_memory = torch.cuda.is_available()
if args.exp:
    train_eval_dataset = torch.utils.data.Subset(train_dataset, train_indices[:NUM_SAMPLES_EVAL])
    train_eval_dataloader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=BATCH_SIZE_VALIDATION,
                                                        shuffle=False,
                                                        num_workers=4, pin_memory=pin_memory)

    test_eval_dataset = torch.utils.data.Subset(test_dataset, test_indices[:NUM_SAMPLES_EVAL])
    test_eval_dataloader = torch.utils.data.DataLoader(test_eval_dataset, batch_size=BATCH_SIZE_VALIDATION,
                                                       shuffle=False,
                                                       num_workers=4, pin_memory=pin_memory)
    del train_eval_dataset, test_eval_dataset
del train_dataset, test_dataset

#################
gen_dataloader = torch.utils.data.DataLoader(GENERATIVECIFAR10(),
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=2)
#################

# Standard Training

seed(args.seed)
metrics = pd.DataFrame()
trainer = Trainer(info, args)
last_lr = args.lr

logger.log('\n\n')
logger.log('Standard training for {} epochs'.format(NUM_STD_EPOCHS))
old_score = [0.0]

gen_acc = trainer.eval_hr(gen_dataloader, False)
print("accuracy: ", gen_acc)