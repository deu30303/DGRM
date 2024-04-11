import torch
import numpy as np

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None
        
def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)
  
    
def random_seq(src):
    #adding [SEP] to unify the format of samples for NSP
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    for i in range(batch_size):
        cur = src[i]
        try:
            first_pad = (cur.tolist()).index(0)
        except:
            first_pad = 256
        cur = cur[1:first_pad].tolist()
        cur = random_string(cur)
        padding = [0] * (length - len(cur))
        dst.append(torch.tensor([101] + cur + padding))
    return torch.stack(dst)

 
def random_string(str):
    #randomly split positive samples into two halves and add [SEP] between them
    try:
        str.remove(102)
    except:
        pass
    try:
        str.remove(102)
    except:
        pass
    len1 = len(str)
    if len1 == 1:
        cut = 1
        str = str[:cut] + [102] + str[cut:] + [102]
    elif len1 == 0:
        pass
    else:
        cut = np.random.randint(1, len1)
        str = str[:cut] + [102] + str[cut:] + [102]
    return str

def change_string(str):
    #creating negative samples for NSP by randomly splitting positive samples
    #and swapping two halves
    try:
        str.remove(102)
    except:
        pass
    try:
        str.remove(102)
    except:
        pass
    len1 = len(str)
    if len1 == 1:
        cut = 1
        str = str[:cut] + [102] + str[cut:] + [102]
    elif len1 == 0:
        pass
    else:
        cut = np.random.randint(1, len1)
        str = str[:cut] + [102] + str[cut:] + [102]
    return str

def get_permutation_batch(src, src_mask):
    #create negative samples for Next Sentence Prediction
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    dst_mask = []
    lbl = []
    for i in range(batch_size):
        cur = src[i]
        mask = src_mask[i].tolist() + [0]
        first_pad = (cur.tolist() + [0]).index(0)
        cur = cur[1:first_pad].tolist()
        cur = change_string(cur)
        lbl.append(1)
        padding = [0] * (length - len(cur) - 1)
        dst.append(torch.tensor([101] + cur + padding))
        dst_mask.append(torch.tensor(mask))
        
    return torch.stack(dst), torch.stack(dst_mask), torch.tensor(lbl)
