"""
Usage:
    train.py [options]

Options:
    -h --help                         show this screen
    --loss-type=<str>                 Which loss to use cross-ent|corr|joint. [default: cross-entropy]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]
    --dataset=<str>                   Name of the dataset [default: twitter]
    --dataset-folder=<str>            folder that contains the data [default: ./data/]
    --algo=<str>                      type of bert algorithm used [default: bert]
"""

from learner import Trainer
from model import MyModelClass
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()
from docopt import docopt
import datetime
import json
import numpy as np
import os


args = docopt(__doc__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Save hyper-parameter values ---> config.json
# Save model weights ---> filename.pt using current time
#####################################################################
filename = args['--dataset']+'_'+args['--algo']
fw = open('configs/' + filename + '.json', 'a')
model_path = filename + '.pt'
args['--checkpoint-path'] = model_path
json.dump(args, fw, sort_keys=True, indent=2)
#####################################################################
# Define Dataloaders
#####################################################################
for file in os.listdir(args['--dataset-folder']):
    file_split = file.split('_')
    if file_split[0] == args['--dataset']:
        if file_split[1].split('.')[0] == 'train':
            args['--train-path'] = args['--dataset-folder']+file
        if file_split[1].split('.')[0] == 'dev':
            args['--dev-path'] = args['--dataset-folder']+file
print("*"*50)
print(args['--train-path'])
print(args['--dev-path'])
print(args['--algo'])
print(args['--dataset'])
print("*"*50)
train_dataset = DataClass(args, args['--train-path'])
train_data_loader = DataLoader(train_dataset,
                               batch_size=int(args['--train-batch-size']),
                               shuffle=True
                               )
print('The number of training batches: ', len(train_data_loader))
dev_dataset = DataClass(args, args['--dev-path'])
dev_data_loader = DataLoader(dev_dataset,
                             batch_size=int(args['--eval-batch-size']),
                             shuffle=False
                             )
print('The number of validation batches: ', len(dev_data_loader))
#############################################################################
# Define Model & Training Pipeline
#############################################################################
model = MyModelClass(output_dropout=float(args['--output-dropout']),
                algo=args['--algo'],
                joint_loss=args['--loss-type'],
                alpha=float(args['--alpha-loss']))
#############################################################################
# Start Training
#############################################################################
learn = Trainer(model, train_data_loader, dev_data_loader, filename=filename)
learn.fit(
    num_epochs=int(args['--max-epoch']),
    args=args,
    device=device
)
