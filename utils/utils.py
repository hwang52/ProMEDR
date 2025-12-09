import os
import warnings
import numpy as np
import math
import torch
import torch.optim as optim
from torch.autograd import Function

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return 

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        reversed_scaled_grad = torch.neg(ctx.lambda_*grad_output.clone())
        return reversed_scaled_grad, None

def grad_reverse(x, LAMBDA):
    return GradReverse.apply(x, LAMBDA)


def numeric_classes(tags_classes, dict_tags):
    num_classes = np.array([dict_tags.get(t) for t in tags_classes])
    return num_classes


def create_dict_texts(texts):
    d = {l: i for i, l in enumerate(texts)}
    return d


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, directory, save_name, last_chkpt):  
    checkpoint_file = os.path.join(directory, save_name+'.pth')
    torch.save(state, checkpoint_file)


def soft_sup_con_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device=None):
    """Compute loss for model. 
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [bsz, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), base_temperature) 
    loss = torch.nn.functional.cross_entropy(features_dot_softlabels, hard_labels) 
    predict = torch.argmax(features_dot_softlabels, 1) 
    correct = (predict == hard_labels).sum().item()

    return loss, correct, predict


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count