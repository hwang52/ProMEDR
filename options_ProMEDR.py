"""
    ProME-DR arguments
"""
import argparse


class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='ProME-DR arguments')

        # System
        parser.add_argument('-generator_layer', '--generator_layer', default=2, type=int)
        parser.add_argument('-stage1_epochs', '--stage1_epochs', default=10, type=int, 
                            help='stage1 training epochs')
        parser.add_argument('-code_path', '--code_path', default='xxx', type=str, 
                            help='code path for your project')
        parser.add_argument('-dataset_path', '--dataset_path', default='xxx', type=str, 
                            help='path for your dataset')
        parser.add_argument('-resume', '--resume_dict', type=str)
        parser.add_argument('-data', '--dataset', default='MFIDDR', 
                            choices=['MFIDDR', 'APTOS', 'DDR', 'DEEPDR', 'EYEPACS', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR'])
        
        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str, 
                            choices=['ViT-B/16', 'ViT-B/32'], 
                            default='ViT-B/32', help='backbone')
        parser.add_argument('-CLS_NUM_TOKENS', '--CLS_NUM_TOKENS', default=5, type=int)

        # Prompts
        parser.add_argument('-VIEW_NUM_TOKENS', '--VIEW_NUM_TOKENS', default=4, type=int)
        parser.add_argument('-VIEW_PROJECT', '--VIEW_PROJECT', default=-1, type=int)
        parser.add_argument('-CLS_PROJECT', '--CLS_PROJECT', default=-1, type=int)
        parser.add_argument('-VP_INITIATION', '--VP_INITIATION', default='random', type=str)
        parser.add_argument('-GP_VIEW_NUM_TOKENS', '--GP_VIEW_NUM_TOKENS', default=1, type=int)
        parser.add_argument('-GP_CLS_NUM_TOKENS', '--GP_CLS_NUM_TOKENS', default=1, type=int)
        parser.add_argument('-tp_N_CTX', '--tp_N_CTX', default=16, type=int)
        parser.add_argument('-use_NTP', '--use_NTP', default=0, type=int)
        parser.add_argument('-debug_mode', '--debug_mode', default=1, type=int)
        parser.add_argument('-dropout', '--dropout', default=0.2, type=float)

        # Parameters
        parser.add_argument('-opt', '--optimizer', type=str, choices=['adam', 'adamw'], default='adam')
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float)
        parser.add_argument('-imsz', '--image_size', default=224, type=int)
        parser.add_argument('-seed', '--seed', type=int, default=55)
        parser.add_argument('-bs', '--batch_size', default=64, type=int)
        parser.add_argument('-t_bs', '--test_batch_size', default=100, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=4, 
                            help='number of workers in loader')
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', 
                            help='number of epochs to train in stage 2')
        parser.add_argument('-lr', '--lr', type=float, default=0.0001, metavar='LR')
        parser.add_argument('-stage2_lr', '--stage2_lr', type=float, default=0.001, metavar='LR')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M')
        parser.add_argument('-select_cls', '--select_cls', type=int, default=0) 
        parser.add_argument('-select_view', '--select_view', type=int, default=-1)
        parser.add_argument('-mask_prob', '--mask_prob', type=float, default=0.5)
        parser.add_argument('-tau', '--tau', type=float, default=0.06)
        
        # Checkpoint
        parser.add_argument('-es', '--early_stop', type=int, default=20)
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N')

        self.parser = parser


    def parse(self):
        # Parse all arguments
        return self.parser.parse_args()