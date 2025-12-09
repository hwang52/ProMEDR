import sys
import os
import torch
import math
import time

# replace with your own path
code_path = '/home/xxx/mycode/ProMEDR_code'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "xxx"))
sys.path.append(os.path.join(code_path, "clip"))
sys.path.append(os.path.join(code_path, "algos"))

from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
from dataset_pre import SingleViewDataset, MultiViewDataset
import torch.backends.cudnn as cudnn
from PIL import Image
from torch import optim
from utils import utils, GPUmanager
from utils.FLoss import FocalLoss
from utils.Evaluate import evaluate
import torchvision.transforms as transform
from mymodel import ProMEDR
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
import clip
import torch.nn.functional as F


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

gm = GPUmanager.GPUManager()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:

    def __init__(self, args):
        
        self.args = args
        print('\nLoading data...')
        self.tr_classes = ['no diabetic retinopathy', 'mild diabetic retinopathy', 
                            'moderate diabetic retinopathy', 
                            'severe diabetic retinopathy', 'proliferative diabetic retinopathy']
        self.dict_class = utils.create_dict_texts(self.tr_classes)
        self.view_calsses = ['0', '1', '2', '3'] 
        self.dict_views = utils.create_dict_texts(self.view_calsses)
        transform_train = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((224, 224)),
            transform.ToTensor()
        ]) 
        transform_valid = transform.Compose([
            transform.ToPILImage(), 
            transform.Resize((224, 224)), 
            transform.ToTensor() 
        ])
        if args.dataset == 'MFIDDR':
            dataset_train = MultiViewDataset('../../mydata/MFIDDR/train.csv', '../../mydata/MFIDDR/train', 
                                transform=transform_train, select_cls=args.select_cls, select_view=args.select_view) 
            dataset_valid = MultiViewDataset('../../mydata/MFIDDR/test.csv', '../../mydata/MFIDDR/test', transform=transform_valid) 
            dataset_test = MultiViewDataset('../../mydata/MFIDDR/test.csv', '../../mydata/MFIDDR/test', transform=transform_valid)
            test_v1 = MultiViewDataset('../../mydata/MFIDDR/test.csv', '../../mydata/MFIDDR/test', 
                                    transform=transform_valid, select_view=0)
            test_v2 = MultiViewDataset('../../mydata/MFIDDR/test.csv', '../../mydata/MFIDDR/test', 
                                    transform=transform_valid, select_view=1)
            test_v3 = MultiViewDataset('../../mydata/MFIDDR/test.csv', '../../mydata/MFIDDR/test', 
                                    transform=transform_valid, select_view=2)
            test_v4 = MultiViewDataset('../../mydata/MFIDDR/test.csv', '../../mydata/MFIDDR/test', 
                                    transform=transform_valid, select_view=3)
        # add other datasets
        elif args.dataset == 'APTOS':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/APTOS_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/APTOS_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/APTOS_crossval.txt', transform=transform_valid)
        elif args.dataset == 'DDR':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/DDR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/DDR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/DDR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'DEEPDR':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/DEEPDR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/DEEPDR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/DEEPDR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'EYEPACS':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/EYEPACS_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/EYEPACS_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/EYEPACS_crossval.txt', transform=transform_valid)
        elif args.dataset == 'FGADR':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/FGADR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/FGADR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/FGADR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'IDRID':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/IDRID_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/IDRID_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/IDRID_crossval.txt', transform=transform_valid)
        elif args.dataset == 'MESSIDOR':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/MESSIDOR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/MESSIDOR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/MESSIDOR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'RLDR':
            # self, dataset_name, txt_path, transform=None, select_cls=-1
            dataset_train = SingleViewDataset(args.dataset, '../../mydata/splits/RLDR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, '../../mydata/splits/RLDR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, '../../mydata/splits/RLDR_crossval.txt', transform=transform_valid)
        
        self.train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, pin_memory=True) 
        self.train_loader_S1 = DataLoader(dataset_train, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(dataset_valid, batch_size=100, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True) 
        self.test_loader = DataLoader(dataset_test, batch_size=100, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v1 = DataLoader(test_v1, batch_size=100, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v2 = DataLoader(test_v2, batch_size=100, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v3 = DataLoader(test_v3, batch_size=100, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v4 = DataLoader(test_v4, batch_size=100, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print('******** Data Loading Done ********')
        print('-'*80)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # CLIP model
        self.CLIP, self.preprocess = clip.load(args.clip_backbone, device, jit=False)
        self.CLIP = self.CLIP.float().eval()
        self.model = self.CLIP.to(device) 

        for p in self.model.parameters():
            p.requires_grad = True 

        self.text_template = torch.cat([clip.tokenize(f"a photo of {c}") for c in self.tr_classes]).to(device)
        self.RG = np.random.default_rng()

        if args.optimizer=='sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                       weight_decay=args.l2_reg, momentum=args.momentum, nesterov=False, lr=args.lr)
        elif args.optimizer=='adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.suffix = 'es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
                      '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)+'_seed-'+str(args.seed)

        self.path_cp = os.path.join(code_path, "algos/clip_main/saved_models")

        self.start_epoch = 0
        self.best_acc = 0.
        self.early_stop_counter = 0
        self.last_chkpt_name='init'
        self.current_epoch = 0
        self.start_epoch = 0

        if args.resume_dict is not None:
            self.resume_from_checkpoint(args.resume_dict)
    

    def do_epoch(self):
        self.model.train()
        batch_time = utils.AverageMeter()
        total_loss = utils.AverageMeter()

        # start counting time
        time_start = time.time()

        for i, (img, cls, view) in enumerate(self.train_loader):
            batch_img = img.to(device)
            batch_cls = cls.to(device)
            batch_view = view.to(device)

            self.optimizer.zero_grad()

            logits_per_image, _ = self.model(batch_img, self.text_template)
            loss_cross = F.cross_entropy(logits_per_image, batch_cls)
            img_features = self.model.encode_image(batch_img)
            text_features = self.model.encode_text(self.text_template)
            loss_ce, co, pred = utils.soft_sup_con_loss(img_features, text_features, batch_cls, device=device)
            loss = loss_ce + loss_cross 

            loss.backward()
            self.optimizer.step()
            total_loss.update(loss.item(), batch_img.size(0))

            # time
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end

        print('[Train] Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {net.val:.4f} ({net.avg:.4f})\t'
                .format(self.current_epoch+1, self.args.epochs, batch_time=batch_time, net=total_loss))

        return {'net':total_loss.avg}


    def do_training(self):

        print('---Train---')
        for self.current_epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            self.adjust_learning_rate()
            loss_dict = self.do_epoch()
            loss_train = loss_dict['net']
            print('\n---Val---')

            val_time, val_loss, val_acc, acc_list = evaluate(self.test_loader_v1, self.model)
            print('[Val-View 1] Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {net.val:.4f} ({net.avg:.4f})\t'
                'val_acc {val_acc:.4f}\t'
                .format(self.current_epoch+1, self.args.epochs, batch_time=val_time, net=val_loss, val_acc=val_acc))
            val_time, val_loss, val_acc, acc_list = evaluate(self.test_loader_v2, self.model)
            print('[Val-View 2] Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {net.val:.4f} ({net.avg:.4f})\t'
                'val_acc {val_acc:.4f}\t'
                .format(self.current_epoch+1, self.args.epochs, batch_time=val_time, net=val_loss, val_acc=val_acc))
            val_time, val_loss, val_acc, acc_list = evaluate(self.test_loader_v3, self.model)
            print('[Val-View 3] Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {net.val:.4f} ({net.avg:.4f})\t'
                'val_acc {val_acc:.4f}\t'
                .format(self.current_epoch+1, self.args.epochs, batch_time=val_time, net=val_loss, val_acc=val_acc))
            val_time, val_loss, val_acc, acc_list = evaluate(self.test_loader_v4, self.model)
            print('[Val-View 4] Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {net.val:.4f} ({net.avg:.4f})\t'
                'val_acc {val_acc:.4f}\t'
                .format(self.current_epoch+1, self.args.epochs, batch_time=val_time, net=val_loss, val_acc=val_acc))
            
            print(f'\n---Train & Val on Epoch {self.current_epoch+2}---')
    

    def adjust_learning_rate(self, min_lr=1e-8):
        lr = self.args.lr * math.pow(1e-3, float(self.current_epoch)/(self.args.epochs))
        lr = max(lr, min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    

    def resume_from_checkpoint(self, resume_dict):
        if resume_dict is not None:
            print('==> Resuming from checkpoint: ', resume_dict)
            model_path = os.path.join(self.path_cp, resume_dict+'.pth')
            checkpoint = torch.load(model_path, map_location=device)
            self.start_epoch = checkpoint['epoch']+1
            self.last_chkpt_name = resume_dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.best_acc = checkpoint['best_acc']