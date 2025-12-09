import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='CLIP Options')
        parser.add_argument('-debug_mode', '--debug_mode', default=1, type=int, help='use debug model')
        parser.add_argument('-code_path', '--code_path', default='', type=str, help='code path')
        parser.add_argument('-dataset_path', '--dataset_path', default='', type=str, help='Path of datasets')
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='MFIDDR', 
                            choices=['MFIDDR', 'APTOS', 'DDR', 'DEEPDR', 'EYEPACS', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR'])
        parser.add_argument('-select_cls', '--select_cls', type=int, default=0, 
                            help='Select the class id.')
        parser.add_argument('-select_view', '--select_view', type=int, default=-1, 
                            help='Select the view id.')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam', 'adamw'], default='adam')

        # Loss weight parameters
        parser.add_argument('-alpha', '--alpha', default=1.0, type=float, help='Parameter')
        parser.add_argument('-l2', '--l2_reg', default=0.00004, type=float, help='L2 Weight Decay for optimizer')
        
        # Model parameters
        parser.add_argument('-clip_backbone', '--clip_backbone', type=str, 
                            choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], 
                            default='ViT-B/32', help='choose clip backbone')
        parser.add_argument('-seed', '--seed', type=int, default=55)
        parser.add_argument('-bs', '--batch_size', default=64, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.00001, metavar='LR', 
                            help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=20, help='Early stopping epochs')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=100, metavar='N', 
                            help='How many batches to wait before logging training status')
        parser.add_argument('-trainvalid', '--trainvalid', choices=[0, 1], default=1, type=int)
        parser.add_argument('-ac_grad', '--ac_grad', default=16, type=int)

        self.parser = parser

    
    def parse(self):
        # Parse the arguments test
        return self.parser.parse_args()