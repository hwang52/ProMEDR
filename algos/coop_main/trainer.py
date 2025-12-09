import sys
from tqdm import tqdm
import os 

# replace with your own path
code_path = '/home/xxx/mycode/ProMEDR_code'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "xxx"))
sys.path.append(os.path.join(code_path, "clip"))
sys.path.append(os.path.join(code_path, "algos"))

from clip.model import CLIP
import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dataset_pre import SingleViewDataset, MultiViewDataset
import math
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import utils, GPUmanager
from utils.FLoss import FocalLoss
from utils.Evaluate import evaluate
from PIL import Image
from functools import reduce
from operator import mul

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

gpu_index = 0
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.clip_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames) 
        n_ctx = cfg.tp_N_CTX 
        dtype = clip_model.dtype # float32
        ctx_dim = clip_model.ln_final.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution 
        cfg_imsize = cfg.image_size 
        assert cfg_imsize == clip_imsize, f"imsize ({cfg_imsize}) must equal to clip ({clip_imsize})"
        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device) 
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) 
        print(f'Initial context: "{prompt_prefix}"') 
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors) 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        prompts = [prompt_prefix + " " + name + "." for name in classnames] 

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 

        prefix = self.token_prefix  
        suffix = self.token_suffix 
        
        prompts = torch.cat(
            [
                prefix, 
                ctx, 
                suffix, 
            ],
            dim=1,
        )
        return prompts 


class PromptedVisionTransformer(nn.Module):
    def __init__(self, config,model:CLIP):
        super(PromptedVisionTransformer, self).__init__()
        self.input_resolution = model.visual.input_resolution
        self.output_dim = model.visual.output_dim
        self.conv1 = model.visual.conv1
        width = self.conv1.out_channels
        self.class_embedding = model.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.ln_post = model.visual.ln_post
        self.proj = model.visual.proj
        self.config = config
        patch_size = self.conv1.kernel_size
        num_tokens = self.config.vp_NUM_TOKENS
        self.num_tokens = num_tokens 
        if self.config.vp_PROJECT > -1:
            # prepend / add
            prompt_dim = self.config.vp_PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        if self.config.vp_INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim)) # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim)) 
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.config.vp_DEEP:  # noqa
                total_d_layer = self.transformer.layers - 1 
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")
        
    def incorporate_prompt(self, x):
        B = x.shape[0] 
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) 
        x = torch.cat((
            x[:, :1, :], 
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1),
            x[:, 1:, :]
        ), dim=1) 
        return x

    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = self.transformer.layers
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output) 
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)
                hidden_states= self.transformer.resblocks[i](hidden_states)
        return hidden_states

    def forward(self, x, vis=False):
        x = self.incorporate_prompt(x)
        if self.config.vp_DEEP:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.forward_deep_prompt(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        else:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP):
        super().__init__()
        self.cfg = cfg
        if cfg.training_strategy == 'TP': # Coop
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model).to(device)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
            self.text_encoder = TextEncoder(clip_model).to(device)
        else:
            self.text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)
            self.text_encoder = clip_model.encode_text
        if cfg.training_strategy == 'VP': # VPT
            self.image_encoder = PromptedVisionTransformer(cfg, clip_model)
        else :
            self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype 

    def forward(self, image): 
        image_features = self.image_encoder(image.type(self.dtype)) 
        if self.cfg.training_strategy == 'TP':
            prompts = self.prompt_learner() 
            tokenized_prompts = self.tokenized_prompts 
            text_features = self.text_encoder(prompts, tokenized_prompts) 
        else:
            text_features = self.text_encoder(self.text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits



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

        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        transform_train = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((224, 224)),
            transform.RandomHorizontalFlip(p=0.3),
            transform.RandomVerticalFlip(p=0.3),
            transform.RandomResizedCrop(224),
            transform.ToTensor(), 
            transform.Normalize(mean=im_mean, std=im_std)
        ]) 
        transform_valid = transform.Compose([
            transform.ToPILImage(), 
            transform.Resize((224, 224)), 
            transform.ToTensor(), 
            transform.Normalize(mean=im_mean, std=im_std) 
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
        
        self.text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.tr_classes]).to(device)
        self.RG = np.random.default_rng()
        self.build_model()

        self.path_cp = os.path.join(code_path, "algos/coop_main/saved_models")

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

        # Start counting time
        time_start = time.time()
        
        for i, (img, cls, view) in enumerate(self.train_loader):
            batch_img = img.to(device)
            batch_cls = cls.to(device)
            batch_view = view.to(device)

            B, C, H, W = img.size()

            self.optimizer.zero_grad()
            logits_per_image = self.model(batch_img)
            loss = F.cross_entropy(logits_per_image, batch_cls)
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
    

    def build_model(self):
        cfg = self.args
        classnames = self.tr_classes

        print(f"Loading CLIP (backbone: {cfg.clip_backbone})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(device)
        # CLIP's default precision is fp16
        clip_model.float()
        
        print("Building custom CLIP")
        
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        
        train_parameters = ['image_encoder.proj', 'image_encoder.prompt_embeddings', 'prompt_learner.ctx', 
                            'text_encoder.text_projection', "image_encoder.deep_prompt_embeddings"]
        tot = 0
        train_part = 0
        for name, param in self.model.named_parameters():
            tot += param.numel()
            print(name)
            if name not in train_parameters:
                param.requires_grad_(False)

        print("=============train==============")
        for name, param in self.model.named_parameters():
            # print(name)
            if param.requires_grad == True:
                train_part += param.numel()
                print(name)
        print(f"tot={tot}, train = {train_part}")
        self.model.to(device)

        # only give prompt_learner to the optimizer
        if self.args.optimizer=='sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                       weight_decay=self.args.l2_reg, momentum=self.args.momentum, nesterov=False, lr=self.args.lr)
        elif self.args.optimizer=='adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                        lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.l2_reg)