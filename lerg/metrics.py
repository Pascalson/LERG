import torch
import numpy as np
import random
import pdb

from scipy.stats import skew
from collections import Counter

def get_expl(x, expl, ratio=0.2, remain_masks=False):
    if expl is None:
        x_entities = [tok for tok in x if random.random() < ratio] if not remain_masks else [tok if random.random() < ratio else "__" for tok in x ]
    else:
        k = int(len(x) * ratio // 1)
        topk = torch.topk(expl, k) if not remain_masks else torch.topk(expl, max(k,1))
        x_entities = [tok for ind, tok in enumerate(x) if ind in topk.indices] if not remain_masks else [tok if ind in topk.indices else "__" for ind, tok in enumerate(x)]
    if remain_masks:
        merged = [x_entities[0]]
        for tok in x_entities[1:]:
            if tok == "__" and merged[-1] == "__":
                continue
            else:
                merged.append(tok)
        x_entities = merged
    return x_entities

def remove_expl(x, expl, ratio=0.2):
    """
    remove given explanation from x
    if None explanation is given, randomly remove
    """
    if expl is None:
        x_re = [tok for tok in x if random.random() >= ratio]
    else:
        k = int(len(x) * ratio // 1)
        topk = torch.topk(expl, k)
        x_re = [tok for ind, tok in enumerate(x) if ind not in topk.indices]
    return x_re

def get_ppl(probs, y, y_inds=None):
    if y_inds is None:
        ent = np.sum(np.log(p[yi]) for p, yi in zip(probs[0], y)) / len(y)
    else:
        ent = np.sum(np.log(p[yi]) for i, (p, yi) in enumerate(zip(probs[0], y)) if i in y_inds) / len(y_inds)
    return np.exp(-ent), ent

def ppl_c_add(expl, x, y, model_f, ratio=0.2):
    """
    additive perplexity changes
    """
    x_add = get_expl(x, expl, ratio=ratio)
    y_probs_add, y_inds = model_f([x_add], label=y, is_x_tokenized=True, is_y_tokenized=True)
    ppl_add, ent_add = get_ppl(y_probs_add.cpu(), y_inds)
    return ent_add, ppl_add, x_add

def ppl_c(expl, x, y, model_f, ratio=0.2):
    """
    perplexity changes
    """
    x_re = remove_expl(x, expl, ratio=ratio)

    y_probs, y_inds = model_f([x], label=y, is_x_tokenized=True, is_y_tokenized=True)
    y_probs_re, _ = model_f([x_re], label=y, is_x_tokenized=True, is_y_tokenized=True)

    ppl, ent = get_ppl(y_probs.cpu(), y_inds)
    ppl_re, ent_re = get_ppl(y_probs_re.cpu(), y_inds)

    entc = ent_re - ent
    return entc, x_re, ppl, ppl_re
