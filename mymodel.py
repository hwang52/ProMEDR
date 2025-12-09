import sys
import os


# Please replace with your own path
code_path = '/home/xxx/xxx/ProMEDR'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "xxx")) # your own dataset path
sys.path.append(os.path.join(code_path, "clip"))

import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP, VisionTransformer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model:CLIP):
        super(TextEncoder, self).__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) # [128,77,512]
        x = x.permute(1, 0, 2)  # N*L*D -> L*N*D 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # L*N*D -> N*L*D 
        x = self.ln_final(x).type(self.dtype)
        text_embeds = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 

        return text_embeds


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model:CLIP, device):
        super(TextPromptLearner, self).__init__()
        n_cls = len(classnames) 
        n_ctx = cfg.tp_N_CTX  # number of context tokens 16
        dtype = clip_model.dtype # float32
        self.ctx_dim = clip_model.ln_final.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution 
        cfg_imsize = cfg.image_size 
        assert cfg_imsize == clip_imsize, f"img size ({cfg_imsize}) must equal to clip ({clip_imsize})"

        ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=dtype).to(device) 
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors) 
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        # text template
        prompts = ["a photo of a " + name + " on " + "X "*n_ctx +"view." for name in classnames] 
        self.prefix_index = [length+5 for length in name_lens] 
        print('-'*80)
        print("Text Prompt Example: " + prompts[0])
        print('-'*80)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("origin_text_embedding", embedding)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts  # Tensor 

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 

        prompts = [torch.cat([self.origin_text_embedding[i,:self.prefix_index[i]],
                        ctx[i],
                        self.origin_text_embedding[i,self.prefix_index[i]+self.n_ctx:]],
                        dim=0).view(1,-1,self.ctx_dim) for i in range(self.n_cls)]
        learn_prompts = torch.cat(prompts, dim=0)

        return learn_prompts 


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super(PromptLearner, self).__init__()
        n_cls = len(classnames) 
        n_ctx = cfg.tp_N_CTX 
        dtype = clip_model.dtype 
        ctx_dim = clip_model.ln_final.weight.shape[0] 
        clip_imsize = clip_model.visual.input_resolution 
        cfg_imsize = cfg.image_size 
        assert cfg_imsize == clip_imsize, f"img size ({cfg_imsize}) must equal to clip ({clip_imsize})"

        # random initialization
        print("Initializing context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device) 
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) 

        print(f'Initial context: "{prompt_prefix}"') 
        print(f"Number of tokens: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors) 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        prompts = [prompt_prefix + " " + name for name in classnames] 

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 5, 77, 512
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 

        prefix = self.token_prefix 
        suffix = self.token_suffix 

        prompts = torch.cat(
            [prefix, ctx, suffix], dim=1,
        )
        return prompts 


class ProMEDR(nn.Module):
    def __init__(self, cfg, dict_class:dict, dict_views:dict, device):
        super(ProMEDR, self).__init__()
        self.cfg = cfg
        self.device = device
        self.dict_class = dict_class
        self.dict_views = dict_views
        self.view_num_tokens = len(self.dict_views)
        self.cls_num_tokens = len(self.dict_class)

        clip:CLIP = self.load_clip()
        self.conv1 = clip.visual.conv1
        self.myclip = clip
        width = self.conv1.out_channels
        self.feature_template = clip.visual.class_embedding
        self.feature_proj = clip.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip.visual.positional_embedding

        self.generator = VisionTransformer(224, 32, 768, self.cfg.generator_layer, 12, 512) 
        scale = width ** -0.5
        self.generator_proj = nn.Parameter(scale * torch.randn(width, 768)) # 768 to 768
        self.ln_pre = clip.visual.ln_pre
        self.transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        self.ori_text_encoder = clip.encode_text
        
        if self.cfg.tp_N_CTX != -1:
            if self.cfg.use_NTP == 1: # all text tuning
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_class.keys(), clip, device)
            else:
                self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_class.keys(), clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else:
            self.text_encoder = clip.encode_text 
        if self.cfg.VIEW_PROJECT > -1:
            sp_view_prompt_dim = self.cfg.VIEW_PROJECT
            self.sp_view_prompt_proj = nn.Linear(sp_view_prompt_dim, width)
            nn.init.kaiming_normal_(self.sp_view_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_view_prompt_dim = width
            self.sp_view_prompt_proj = nn.Identity()

        if self.cfg.CLS_PROJECT > -1:
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_view_prompt_dim))
        self.specific_view_prompts = nn.Parameter(torch.zeros(self.view_num_tokens, sp_view_prompt_dim)) 
        nn.init.uniform_(self.specific_view_prompts.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))
        self.specific_class_prompts = nn.Parameter(torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim)) 
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + width)) 
        self.ge_cls_prompt_template = nn.Parameter(torch.zeros(self.cfg.GP_CLS_NUM_TOKENS, width))
        nn.init.uniform_(self.ge_cls_prompt_template.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + width)) 
        self.ge_view_prompt_template = nn.Parameter(torch.zeros(self.cfg.GP_VIEW_NUM_TOKENS, width))
        nn.init.uniform_(self.ge_view_prompt_template.data, -val, val) 

        self.specific_head = nn.Linear(512, 5)
        self.mask_prob = cfg.mask_prob


    def incorporate_prompt(self, x, view_index, cls_index, stage, view_prompts=None, cls_prompts=None):
        B = x.shape[0] # batch size
        if stage == 1:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1) 
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B, -1).view(B, 1, -1), 
                self.sp_view_prompt_proj(self.specific_view_prompts[view_index]).view(B, 1, -1), 
                self.sp_cls_prompt_proj(self.specific_class_prompts[cls_index]).view(B, 1, -1), 
                x + self.clip_positional_embedding[1:]
            ), dim=1) 
            
        elif stage == 2:
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1)
            sp_view_prompts = self.sp_view_prompt_proj(self.specific_view_prompts) 
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts) 
            cls_mask = torch.rand(B, sp_cls_prompts.shape[0]).to(self.device)
            cls_prompt_mask = cls_mask < self.mask_prob
            cls_prompt_mask = cls_prompt_mask.unsqueeze(-1)

            view_mask = torch.rand(B, sp_view_prompts.shape[0]).to(self.device) 
            view_prompt_mask = view_mask < self.mask_prob 
            view_prompt_mask = view_prompt_mask.unsqueeze(-1)
        
            sp_cls_prompts = sp_cls_prompts.expand(B, -1, -1).masked_fill(cls_prompt_mask, 0) 
            sp_view_prompts = sp_view_prompts.expand(B, -1, -1).masked_fill(view_prompt_mask, 0) 

            x = torch.cat((
                self.ge_view_prompt_template.expand(B, -1, -1), 
                self.ge_cls_prompt_template.expand(B, -1, -1), 
                sp_view_prompts, 
                sp_cls_prompts, 
                x + self.generator.positional_embedding[1:]
            ), dim=1)
        
        elif stage == 3:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1) 
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1), 
                view_prompts.view(B, self.cfg.GP_VIEW_NUM_TOKENS, -1), 
                cls_prompts.view(B, self.cfg.GP_CLS_NUM_TOKENS, -1), 
                x + self.clip_positional_embedding[1:]
            ), dim=1) 
        
        elif stage == 4:
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1) 
            x = torch.cat(( 
                self.ge_view_prompt_template.expand(B, -1, -1), 
                self.ge_cls_prompt_template.expand(B, -1, -1), 
                self.sp_view_prompt_proj(self.specific_view_prompts).expand(B, -1, -1), 
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B, -1, -1), 
                x + self.generator.positional_embedding[1:]
            ), dim=1) 

        elif stage == 5: 
            x = self.generator.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1) 
            x = torch.cat(( 
                self.ge_view_prompt_template.expand(B,-1,-1), 
                self.ge_cls_prompt_template.expand(B,-1,-1), 
                self.sp_view_prompt_proj(self.specific_view_prompts).expand(B,-1,-1), 
                self.sp_cls_prompt_proj(self.specific_class_prompts).expand(B,-1,-1), 
                x + self.generator.positional_embedding[1:]
            ), dim=1) 
            
        elif stage == 6:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1) 
            x = torch.cat((
                (self.feature_template+self.clip_positional_embedding[0]).expand(B,-1).view(B, 1, -1), 
                view_prompts[torch.arange(B), view_index].view(B, 1, -1), 
                cls_prompts[torch.arange(B), cls_index].view(B, 1, -1), 
                x + self.clip_positional_embedding[1:]
            ), dim=1)
        
        return x


    def vit(self, x, out_token):
        if out_token == 1:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:,out_token,:])
            x = x @ self.feature_proj
        else :
            x = self.generator.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.generator.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.generator.ln_post(x[:,:out_token,:])
            x = x @ self.generator_proj
        return x
    

    def image_encoder(self, image, view_id, cls_id, stage):
        if stage == 1: 
            x = self.incorporate_prompt(image, view_id, cls_id, 1) 
            x = torch.nn.functional.dropout(x, p=0.5) 
            x = self.vit(x, 1) 
        elif stage == 2: 
            x = self.incorporate_prompt(image, view_id, cls_id, 2) 
            x = self.vit(x, self.cfg.GP_VIEW_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS) 
            ge_view_prompts = x[:, :self.cfg.GP_VIEW_NUM_TOKENS]
            ge_cls_prompts = x[:, self.cfg.GP_VIEW_NUM_TOKENS:]
            x = self.incorporate_prompt(image, view_id, cls_id, 3, ge_view_prompts, ge_cls_prompts) 
            x = self.vit(x, 1) 
        elif stage == 4:
            x = self.incorporate_prompt(image, view_id, cls_id, 4) 
            x = self.vit(x, self.cfg.GP_VIEW_NUM_TOKENS+self.cfg.GP_CLS_NUM_TOKENS) 
            ge_view_prompts = x[:, :self.cfg.GP_VIEW_NUM_TOKENS]
            ge_cls_prompts = x[:, self.cfg.GP_VIEW_NUM_TOKENS:]
            x = self.incorporate_prompt(image, view_id, cls_id, 3, ge_view_prompts, ge_cls_prompts) 
            x = self.vit(x, 1) 

        if stage == 2:
            ori_view_prompts = self.sp_view_prompt_proj(self.specific_view_prompts)
            ori_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)
            return x, ge_view_prompts, ge_cls_prompts, ori_view_prompts, ori_cls_prompts
        else:
            return x
    

    def forward(self, image, view_id, cls_id, stage):
        class_name = ['no diabetic retinopathy', 'mild diabetic retinopathy', 
                      'moderate diabetic retinopathy', 
                      'severe diabetic retinopathy', 'proliferative diabetic retinopathy']
        if stage == 2: 
            image_features, ge_v, ge_c, ori_v, ori_c = self.image_encoder(image, view_id, cls_id, stage) 
        else:
            image_features = self.image_encoder(image, view_id, cls_id, stage) 

        if self.cfg.tp_N_CTX != -1: 
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else:
            text_template = torch.cat([clip.tokenize(f"a photo of {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)

        pred_logits = self.specific_head(image_features) 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ori_text_template = torch.cat([clip.tokenize(f"a photo of {c}") for c in class_name]).to(self.device)
        ori_text_features = self.ori_text_encoder(ori_text_template)
        ori_text_features = ori_text_features / ori_text_features.norm(dim=-1, keepdim=True)
        if stage == 2:
            ori_v = ori_v.unsqueeze(0).repeat(ge_v.size(0), 1, 1)
            ori_c = ori_c.unsqueeze(0).repeat(ge_c.size(0), 1, 1)
            return image_features, text_features, ori_text_features, pred_logits, ge_v, ge_c, ori_v, ori_c 
        else:
            return image_features, text_features, ori_text_features, pred_logits 
    

    def predict(self, image, stage):
        cls_id = None
        view_id = None
        image_features = self.image_encoder(image, view_id, cls_id, stage) # batch, 512
        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else:
            class_name = self.dict_class.keys()
            text_template = torch.cat([clip.tokenize(f"a photo of {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features
    

    def get_feas(self, image, view_id, cls_id):
        class_name = ['no diabetic retinopathy', 'mild diabetic retinopathy', 
                      'moderate diabetic retinopathy', 
                      'severe diabetic retinopathy', 'proliferative diabetic retinopathy']
        
        image_features = self.image_encoder(image, view_id, cls_id, 4) 
        
        if self.cfg.tp_N_CTX != -1: 
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else:
            text_template = torch.cat([clip.tokenize(f"a photo of {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)

        pred_logits = self.specific_head(image_features) 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features, pred_logits 


    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print('-'*80)
        print(f"******** load CLIP backbone : {backbone_name} ********")
        print('-'*80)
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        try:
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)
        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)