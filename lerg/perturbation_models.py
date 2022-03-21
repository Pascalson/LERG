import torch
import warnings
import math
import random
import numpy as np
import pdb

import scipy as sp
import sklearn
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def binomial_coef_dist(n):
    dist = [math.comb(n, i+1) for i in range(n//2)]
    total = sum(dist)
    dist = [density / total for density in dist]
    return dist, total

class BasicPM():
    def __init__(self):
        pass
    def perturb_inputs(self, x, num=1):
        """
        argument:
            x: the tokenized input sentence
        return:
            x_set: the perturbed xs
            z_set: the simplified features of x_set, {0,1}^|x|, tensor
        """
        if num != 1:
            warnings.warn("BasicPM will always set argument num == 1")
        return [x],torch.tensor([[1.0 for tok in x]])


class RandomPM(BasicPM):
    """
    randomly choose tokens to be replaced with sub_t
    """
    def __init__(self, sub_t="", denoising=False):
        super().__init__()
        self.sub_t = sub_t
        self.denoising = denoising
        if self.denoising:
            self.sub_t = '<mask>'
            self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to('cuda')

    def _select_repl_num(self, dist_scale):
        """
        select the number of tokens to be replaces #repl_num, following binomial distribution
        """
        pos = random.random()
        for i in range(len(dist_scale)):
            if pos < dist_scale[i]:
                break
        return i+1

    def _denoise_x_set(self, x_set):
        inputs = self.bart_tokenizer(x_set, max_length=256, return_tensors='pt', padding=True).to('cuda')
        summary_ids = self.bart_model.generate(
            inputs['input_ids'],
            top_k=10, top_p=0.9, temperature=0.9, max_length=256,
            early_stopping=True, num_return_sequences=1
        )
        lines = [self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        x_set = lines
        return x_set

    def perturb_inputs(self, x, num=1000, with_i=None):
        """
        allow half tokens at most can be replaced
        """
        dist, num_comb = binomial_coef_dist(len(x))
        dist_scale = [sum(dist[:i+1]) for i in range(len(dist))]
        num = num if num < num_comb*4 else num_comb*4

        """
        choose tokens to be replaced with sub_t
        """
        if with_i is None:
            x_set, z_set = [], []
        else:
            x_set, x_set_with_i = [], []
            weights = []
        for _ in range(num):
            repl_num = self._select_repl_num(dist_scale)
            x_set.append(list(x))
            if with_i is None:
                z_set.append(np.ones((len(x),)))
                repl_list = random.sample(list(range(len(x))), repl_num)
                for t in repl_list:
                    x_set[-1][t] = self.sub_t
                    z_set[-1][t] = 0.
            else:
                x_set_with_i.append(list(x))
                indices_to_repl = list(range(len(x)))
                indices_to_repl.remove(with_i)
                repl_list = random.sample(indices_to_repl, repl_num)
                for t in repl_list:
                    x_set[-1][t] = self.sub_t
                    x_set_with_i[-1][t] = self.sub_t
                x_set[-1][with_i] = self.sub_t
                weights.append(1/(dist[repl_num-1]*len(x)))
        if self.denoising:
            x_set = self._denoise_x_set([' '.join(x) for x in x_set])
        if with_i is None:
            return x_set, torch.tensor(z_set, dtype=torch.float32), x
        else:
            return x_set, x_set_with_i, weights, x


class LIMERandomPM(RandomPM):
    def perturb_inputs(self, x, num=1000):
        """
        allow half tokens at most can be replaced
        choose tokens to be replaced with sub_t
        """
        dist, num_comb = binomial_coef_dist(len(x))
        num = num if num < num_comb*4 else num_comb*4

        x_set, z_set = [], []

        sample = np.random.randint(1,len(x)//2+1,num-1)
        
        x_set.append(list(x))
        z_set.append(np.ones((len(x),)))
        for i in range(num-1):
            repl_num = sample[i]
            x_set.append(list(x))
            z_set.append(np.ones((len(x),)))
            repl_list = random.sample(list(range(len(x))), repl_num)
            for t in repl_list:
                x_set[-1][t] = self.sub_t
                z_set[-1][t] = 0.

        if self.denoising:
            x_set = self._denoise_x_set([' '.join(x) for x in x_set])
        return x_set, torch.tensor(z_set, dtype=torch.float32), x
