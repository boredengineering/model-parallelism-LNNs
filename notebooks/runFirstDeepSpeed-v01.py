import numpy as np
import argparse
import torchvision
import torch
from torch.utils.data import Dataset
import logging
from mingpt.utils import set_seed
from mingpt.utils import ImageDataset, TrainerConfig, kmeans
from mingpt.model import GPT, GPTConfig, GPT1Config
from mingpt.trainer import Trainer, DeepSpeedTrainer

# ****** Step 2 / ToDo 1: import the required library here ******
import deepspeed
# import uip

# +
def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')    
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    
    # ****** Step 2 / ToDo 2: Include parsing DeepSpeed configuration arguments here ******
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

args = add_argument()



#****** Step 2 / ToDo 3: Make sure you initialise the distributed backend with default parameters here ******
#****** No need to set the parameters ****
## FIXME
deepspeed.init_distributed()

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
set_seed(42)

# get the data CIFAR10
root = './'
train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)

# +
# get random 5 pixels per image and stack them all up as rgb values to get half a million random pixels
pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()

# run kmeans to get our codebook
ncluster = 512
with torch.no_grad():
    C = kmeans(px, ncluster, niter=8)
    
train_dataset = ImageDataset(train_data, C)
test_dataset = ImageDataset(test_data, C)
# -
# we'll do something a bit smaller
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                  n_layer=12, n_head=8, n_embd=256)
model = GPT(mconf)

tokens_per_epoch = len(train_data) * train_dataset.block_size
train_epochs = 2 # todo run a bigger model and longer, this is tiny


# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=1, learning_rate=3e-3,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='cifar10_model.pt',
                      num_workers=4,
                     cmd_args=args)
#****** Step 2 / ToDo 4: Use the corresponding Trainer here ******
trainer = DeepSpeedTrainer(model, train_dataset, test_dataset, tconf)

trainer.train()
