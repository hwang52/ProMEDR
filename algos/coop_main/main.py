import sys
import torch
import os

# replace with your own path
code_path = '/home/xxx/mycode/ProMEDR_code'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "xxx"))
sys.path.append(os.path.join(code_path, "clip"))
sys.path.append(os.path.join(code_path, "algos"))

from trainer import Trainer
from options_prompt_tuning import Options
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main(args):
    trainer = Trainer(args)
    trainer.do_training()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)