"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
    --algo=<str>                      type of bert algorithm used [default: bert]
    --dataset=<str>                   Name of the dataset [default: twitter]
    --dataset-folder=<str>            folder that contains the data [default: ./data/]
"""
from learner import EvaluateOnTest
from model import MyModelClass
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()
from docopt import docopt
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
# Define Dataloaders
#####################################################################
for file in os.listdir(args['--dataset-folder']):
    file_split = file.split('_')
    if file_split[0] == args['--dataset']:
        if file_split[1].split('.')[0] == 'test':
            args['--test-path'] = args['--dataset-folder']+file

print("*"*50)
print(args['--test-path'])
print(args['--algo'])
print(args['--dataset'])
print("*"*50)
test_dataset = DataClass(args, args['--test-path'])
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(args['--test-batch-size']),
                              shuffle=False)
print('The number of Test batches: ', len(test_data_loader))
#############################################################################
# Run the model on a Test set
#############################################################################
model = MyModelClass(algo=args['--algo'])
learn = EvaluateOnTest(model, test_data_loader, model_path='models/' + args['--dataset']+'_'+args['--algo']+'_'+'checkpoint'+'.pt')
learn.predict(device=device)