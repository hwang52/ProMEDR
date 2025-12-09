import sys
import torch
import os 


# Please replace with your own path
code_path = '/home/xxx/xxx/ProMEDR'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "xxx")) # your own dataset path
sys.path.append(os.path.join(code_path, "clip"))
sys.path.append(os.path.join(code_path, "algos"))

from trainer import Trainer
from options_ProMEDR import Options

def main_experiment(args):
    trainer = Trainer(args)
    trainer.do_training()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = Options().parse()
    print('ProME-DR Parameters:\t' + str(args))
    main_experiment(args)