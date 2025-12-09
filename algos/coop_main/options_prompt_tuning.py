import argparse

class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='PromptTuning for CoOp')
        parser.add_argument('-code_path', '--code_path', default='', type=str, help='code path')
        parser.add_argument('-dataset_path', '--dataset_path', default='', type=str, help='Path of datasets')
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='MFIDDR', 
                            choices=['MFIDDR', 'APTOS', 'DDR', 'DEEPDR', 'EYEPACS', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR'])
        parser.add_argument('-select_cls', '--select_cls', type=int, default=0, help='Select the class id.')
        parser.add_argument('-select_view', '--select_view', type=int, default=-1, help='Select the view id.')
        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str, choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32', help='choose clip backbone')
        parser.add_argument('-tp_N_CTX', '--tp_N_CTX', type=int, default=16, help='number of text prompt context tokens')
        parser.add_argument('-tp_CTX_INIT', '--tp_CTX_INIT', type=str, default="a photo of a", help='context tokens init')
        parser.add_argument('-ts', '--training_strategy', default='VP', type=str, choices=['TP', 'VP'], 
                            help='training_strategy,TP is CoOp,VP is VPT in the paper')
        parser.add_argument('-vp_NUM_TOKENS', '--vp_NUM_TOKENS', default=10, type=int, help='number of visual prompt tokens')
        parser.add_argument('-vp_PROJECT', '--vp_PROJECT', default=-1, type=int, help='projection for visual prompt')
        parser.add_argument('-vp_INITIATION', '--vp_INITIATION', default='random', type=str, help='initiation for visual prompt')
        parser.add_argument('-vp_DEEP', '--vp_DEEP', default=False, type=bool, help='deep viusla prompt')
        parser.add_argument('-debug_mode', '--debug_mode', default=1, type=int, 
                            help='use debug model, program will break down after a few iterations')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')

        # Loss weight & reg. parameters
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input image size')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=64, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=12, help='Number of workers in data loader')

        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', 
                            help='How many batches to wait before logging training status')
        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()