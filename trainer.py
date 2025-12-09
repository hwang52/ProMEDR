import sys
import os
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
import os
import torch
import math
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from dataset_pre import SingleViewDataset, MultiViewDataset
import torch.backends.cudnn as cudnn
from PIL import Image
from torch import optim
from utils import utils, GPUmanager
from utils.FLoss import FocalLoss
from utils.Evaluate import batch_evaluate
from utils.IFNCELoss import InfoNCE_LCL
import torchvision.transforms as transform
from mymodel import ProMEDR
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts


# Please replace with your own path
code_path = '/home/xxx/xxx/ProMEDR'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "xxx")) # your own dataset path
sys.path.append(os.path.join(code_path, "clip"))

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

gm = GPUmanager.GPUManager()
gpu_index = 0   
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, args):
        self.args = args
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        print('-'*80)
        print('******** Data Loading Start ********')
        self.tr_classes = ['no diabetic retinopathy', 'mild diabetic retinopathy', 
                            'moderate diabetic retinopathy', 
                            'severe diabetic retinopathy', 'proliferative diabetic retinopathy']
        self.dict_class = utils.create_dict_texts(self.tr_classes)
        self.view_calsses = ['0', '1', '2', '3'] 
        self.dict_views = utils.create_dict_texts(self.view_calsses)
        transform_train = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((224, 224)),
            transform.RandomHorizontalFlip(p=0.3),
            transform.RandomVerticalFlip(p=0.3),
            transform.RandomResizedCrop(224),
            transform.ToTensor()
        ]) 
        transform_valid = transform.Compose([
            transform.ToPILImage(), 
            transform.Resize((224, 224)), 
            transform.ToTensor() 
        ])
        if args.dataset == 'MFIDDR':
            dataset_train = MultiViewDataset('./mydata/MFIDDR/train.csv', './mydata/MFIDDR/train', 
                                    transform=transform_train, 
                                    select_cls=args.select_cls, 
                                    select_view=args.select_view) 
            dataset_valid = MultiViewDataset('./mydata/MFIDDR/test.csv', './mydata/MFIDDR/test', 
                                    transform=transform_valid) 
            dataset_test = MultiViewDataset('./mydata/MFIDDR/test.csv', './mydata/MFIDDR/test', 
                                    transform=transform_valid)
            test_v1 = MultiViewDataset('./mydata/MFIDDR/test.csv', './mydata/MFIDDR/test', 
                                       transform=transform_valid, select_view=0)
            test_v2 = MultiViewDataset('./mydata/MFIDDR/test.csv', './mydata/MFIDDR/test', 
                                       transform=transform_valid, select_view=1)
            test_v3 = MultiViewDataset('./mydata/MFIDDR/test.csv', './mydata/MFIDDR/test', 
                                       transform=transform_valid, select_view=2)
            test_v4 = MultiViewDataset('./mydata/MFIDDR/test.csv', './mydata/MFIDDR/test', 
                                       transform=transform_valid, select_view=3)
        # add other datasets
        elif args.dataset == 'APTOS':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/APTOS_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/APTOS_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/APTOS_crossval.txt', transform=transform_valid)
        elif args.dataset == 'DDR':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/DDR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/DDR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/DDR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'DEEPDR':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/DEEPDR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/DEEPDR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/DEEPDR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'EYEPACS':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/EYEPACS_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/EYEPACS_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/EYEPACS_crossval.txt', transform=transform_valid)
        elif args.dataset == 'FGADR':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/FGADR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/FGADR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/FGADR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'IDRID':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/IDRID_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/IDRID_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/IDRID_crossval.txt', transform=transform_valid)
        elif args.dataset == 'MESSIDOR':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/MESSIDOR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/MESSIDOR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/MESSIDOR_crossval.txt', transform=transform_valid)
        elif args.dataset == 'RLDR':
            dataset_train = SingleViewDataset(args.dataset, './mydata/splits/RLDR_train.txt', transform=transform_train, select_cls=args.select_cls) 
            dataset_valid = SingleViewDataset(args.dataset, './mydata/splits/RLDR_crossval.txt', transform=transform_valid) 
            dataset_test = SingleViewDataset(args.dataset, './mydata/splits/RLDR_crossval.txt', transform=transform_valid)

        self.train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, pin_memory=True) 
        self.train_loader_S1 = DataLoader(dataset_train, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(dataset_valid, batch_size=args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True) 
        self.test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v1 = DataLoader(test_v1, batch_size=args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v2 = DataLoader(test_v2, batch_size=args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v3 = DataLoader(test_v3, batch_size=args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_loader_v4 = DataLoader(test_v4, batch_size=args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print('******** Data Loading Done ********') 
        print('-'*80) 

        self.model = ProMEDR(self.args, self.dict_class, self.dict_views, device)
        self.model = self.model.to(device)
        self.best_stage1_acc = 0.0
        self.best_acc = 0.0
        self.scheduler = None
        self.focal_loss = FocalLoss(gamma=2)
        self.lcl_func = InfoNCE_LCL(args.tau) 

        self.target_layer_output = None
        self.target_layer_grad = None

        print('-'*80)
        print("******** Training Settings ********")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")
        print(f"view specific prompt dim = {self.args.VIEW_PROJECT}")
        print(f"class specific prompt dim = {self.args.CLS_PROJECT}")
        print(f"generation view prompt numbers = {self.args.GP_VIEW_NUM_TOKENS}")
        print(f"generation class prompt numbers = {self.args.GP_CLS_NUM_TOKENS}")
        str1 = "True" if self.args.use_NTP else "False"
        print(f"normal all text prompt tuning = {str1}")
        print(f"text prompt numbers = {self.args.tp_N_CTX}")
        print('-'*80)

    
    def training_set(self, stage):
        tot = 0
        for name, param in self.model.named_parameters():
            tot += param.numel()
            # print(name)
        self.current_epoch = 0
        self.start_epoch = 0
        lr = self.args.lr
        stage2_lr = self.args.stage2_lr 

        print('-'*80)

        if stage == 1:
            print("******** ProME-DR Stage1: Prompt Matching Learning Setup ********")
            train_parameters = ['specific_view_prompts','specific_class_prompts', 'specific_head', 
                                'text_prompt_learner.ctx', 'text_encoder.text_projection', 
                                'sp_view_prompt_proj','sp_cls_prompt_proj'] 
        elif stage == 2:
            print("******** ProME-DR Stage2: Prompt Emulating Learning Setup ********")
            train_parameters = ['generator', 'specific_head', 
                                'text_prompt_learner.ctx', 'text_encoder.text_projection', 
                                'feature_proj', 
                                'feature_template', 
                                'ge_cls_prompt_template', 'ge_view_prompt_template']

        if self.args.optimizer=='sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                weight_decay=0.001, momentum=self.args.momentum, 
                                nesterov=False, lr=lr)
        elif self.args.optimizer=='adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                        lr=lr, betas=(0.9, 0.999), eps=1e-8, 
                                        weight_decay=0.0001) 
        elif self.args.optimizer=='adamw':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                         lr=lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-8)
        
        if stage == 2:
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=4, T_mult=2)

        print('-'*80)
    

    def do_epoch(self, stage, epoch_idx):
        self.model.train()
        batch_time = utils.AverageMeter()
        total_loss = utils.AverageMeter()
        num_epochs = self.args.epochs
        train_loader = self.train_loader
        if stage == 1:
            num_epochs = self.args.stage1_epochs
            train_loader = self.train_loader_S1
        # Start counting time
        time_start = time.time()
        correct = 0
        correct_logits = 0
        tot = 0

        for i, (img, cls, view) in enumerate(train_loader):
            batch_img = img.to(device)
            batch_cls = cls.to(device)
            batch_view = view.to(device)
            B, C, H, W = img.size()
            index = torch.randperm(B).cuda()
            lam = np.random.beta(0.2, 0.2)
            imgs_mix = lam*batch_img + (1-lam)*batch_img[index]
            label_a, label_b = batch_cls, batch_cls[index]
            if lam >= 0.50: 
                view_mix = batch_view
                label_mix = batch_cls
            else:
                view_mix = batch_view[index]
                label_mix = batch_cls[index]
            self.optimizer.zero_grad()

            if stage == 2:
                feature, soft_label, ori_text_f, logits, ge_v, ge_c, ori_v, ori_c = self.model(batch_img, batch_view, batch_cls, stage)
                feature_mix, soft_label_mix, ori_text_f_mix, logits_mix, _, _, _, _ = self.model(imgs_mix, view_mix, label_mix, stage)
                ge_feas_v = ge_v[torch.arange(ge_v.size(0)), batch_view]
                ge_feas_c = ge_c[torch.arange(ge_c.size(0)), batch_cls]
                pos_key_v = ori_v[torch.arange(ori_v.size(0)), batch_view]
                pos_key_v = pos_key_v.detach()
                pos_key_c = ori_c[torch.arange(ori_c.size(0)), batch_cls]
                pos_key_c = pos_key_c.detach()
                mask_c = torch.ones(ge_c.size(0), ge_c.size(1), dtype=torch.bool)
                mask_c[torch.arange(ge_c.size(0)), batch_cls] = False
                neg_keys_c = ori_c[mask_c].view(ge_c.size(0), ge_c.size(1)-1, ge_c.size(2))
                neg_keys_c = neg_keys_c.detach()

                if self.args.dataset == 'MFIDDR':
                    mask_v = torch.ones(ge_v.size(0), ge_v.size(1), dtype=torch.bool)
                    mask_v[torch.arange(ge_v.size(0)), batch_view] = False
                    neg_keys_v = ori_v[mask_v].view(ge_v.size(0), ge_v.size(1)-1, ge_v.size(2))
                    neg_keys_v = neg_keys_v.detach()
                    if_vloss = self.lcl_func.infonce_lcl_loss(ge_feas_v, pos_key_v, neg_keys_v) 
                    if_closs = self.lcl_func.infonce_lcl_loss(ge_feas_c, pos_key_c, neg_keys_c) 
                    if_factor = 0.5 
                    if_loss = if_closs * if_factor + if_vloss * (1.0-if_factor)
                else:
                    if_loss = self.lcl_func.infonce_lcl_loss(ge_feas_c, pos_key_c, neg_keys_c) 
            else:
                feature, soft_label, ori_text_f, logits = self.model(batch_img, batch_view, batch_cls, stage)
                feature_mix, soft_label_mix, ori_text_f_mix, logits_mix = self.model(imgs_mix, view_mix, label_mix, stage)
                if_loss = None

            hard_labels = batch_cls
            recal_loss = torch.nn.functional.mse_loss(soft_label, ori_text_f) * 0.5
            loss_ce, co, pred = utils.soft_sup_con_loss(feature, soft_label, hard_labels, device=device)
            loss_logits = torch.nn.functional.cross_entropy(logits, batch_cls) 

            pre_one_hot = torch.argmax(logits, 1) 
            pre_correct = (pre_one_hot == batch_cls).sum().item()
            loss_focal = lam*self.focal_loss(logits_mix, label_a) + (1-lam)*self.focal_loss(logits_mix, label_b) 
            
            if (if_loss is not None) and (stage == 2): 
                loss = loss_ce + loss_logits + loss_focal + recal_loss + if_loss 
            else:
                loss = loss_ce + loss_logits + loss_focal + recal_loss 
            
            correct += co
            correct_logits += pre_correct
            tot += batch_img.size(0) 
            loss.backward() 

            if (self.scheduler is not None) and (stage == 2):
                self.scheduler.step(epoch_idx + i / len(train_loader))

            self.optimizer.step()
            total_loss.update(loss.item(), batch_img.size(0))
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end

        print('[Train] Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {net.val:.4f} ({net.avg:.4f})\t'
                    .format(self.current_epoch+1, num_epochs, i+1, 
                        len(train_loader), batch_time=batch_time, net=total_loss))

        return {'loss':total_loss.avg, 'acc':correct/tot, 'acc_logits':correct_logits/tot}
    

    def do_training(self):
        # ProME-DR Stage1: Prompt Matching Learning
        self.training_set(1)
        print("******** ProME-DR Stage1: Prompt Matching Learning Training ********")
        for self.current_epoch in range(self.start_epoch, self.args.stage1_epochs):
            s_time = time.time()
            loss = self.do_epoch(stage=1, epoch_idx=self.current_epoch)
            print(f"epoch = [{self.current_epoch}/{self.args.stage1_epochs}] | loss = {loss}")
            acc = loss['acc']
            if acc > self.best_stage1_acc:
                self.best_stage1_acc = acc
                if self.args.dataset == 'MFIDDR':
                    utils.save_checkpoint({'epoch':self.current_epoch+1, 
                                            'model_state_dict':self.model.state_dict(), 
                                            'optimizer_state_dict':self.optimizer.state_dict(), 
                                            'accuracy':acc, 
                                            'best_acc':self.best_stage1_acc}, 
                            directory='./save_models/stage1/', 
                            save_name='MFIDDR_stage1_best_acc_model', 
                            last_chkpt='MFIDDR_stage1_best_acc_model')
        # ProME-DR Stage2: Prompt Emulating Learning
        self.training_set(2)
        print("******** ProME-DR Stage2: Prompt Emulating Learning Training ********")
        for self.current_epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            loss = self.do_epoch(stage=2, epoch_idx=self.current_epoch)
            print(f"epoch = [{self.current_epoch}/{self.args.epochs}] | loss = {loss}")
            stage2_train_acc = loss['acc']
            print(f'\n******** Start Validation Performance on Epoch {self.current_epoch} ********')
            # evaluate
            val_time, val_net, _, val_acc, acc_list, _, _ = batch_evaluate(self.test_loader, self.model, stage=4)
            print('[Validation] Epoch: [{0}/{1}]\t'
                    'Time {val_batch_time.val:.3f} ({val_batch_time.avg:.3f})\t'
                    'Loss {val_net.val:.4f} ({val_net.avg:.4f})\t'
                    .format(self.current_epoch+1, self.args.epochs, 
                        val_batch_time=val_time, val_net=val_net))

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                if self.args.dataset == 'MFIDDR':
                    utils.save_checkpoint({'epoch':self.current_epoch+1, 
                                            'model_state_dict':self.model.state_dict(), 
                                            'optimizer_state_dict':self.optimizer.state_dict(), 
                                            'accuracy':val_acc, 
                                            'best_acc':self.best_acc}, 
                            directory='./save_models/stage2/', 
                            save_name='MFIDDR_stage2_best_acc_model', 
                            last_chkpt='MFIDDR_stage2_best_acc_model')

            end = time.time()
            elapsed = end-start

            print(f"Pred ACC List [{acc_list[0]:.6f}, {acc_list[1]:.6f}, {acc_list[2]:.6f}, {acc_list[3]:.6f}, {acc_list[4]:.6f}]")
            print(f'******** End Validation Performance on Epoch {self.current_epoch} ********\n')


    def _adjust_learning_rate(self, min_lr=1e-6):
        # lr adjustment
        new_lr = self.args.lr * math.pow(0.1, float(self.current_epoch)/(self.args.epochs)) 
        new_lr = max(new_lr, min_lr)
        for param_group in self.optimizer.param_groups: 
            param_group['lr'] = new_lr


    def _cosine_schedule_with_warmup(self, optimizer, warmup_steps, total_steps, base_lr=1.0):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))  # linear warm-up
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay
        return LambdaLR(optimizer, lr_lambda)


    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.cl_tau
        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss



