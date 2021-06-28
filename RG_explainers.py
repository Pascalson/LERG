import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import Ridge
import numpy as np
import random

class Explainer():
    """
    The base class for various explainers
    arguments:
        model_f: the tested model function, who generates y given x
            require the outputs in the form (probabilities of output sequence, y's tokens ids), having the same length
        x: the input
        y: the generated sequence
    return:
        phi_set: correspond to the weight of each x_i
    """
    def __init__(self, model_f, x, y):
        self.phi_map = {}
        self.model_f = model_f
        self.x = x
        self.y = y
        
    def get_prob(self, probs, y):
        y_probs = [[p[yi] for p, yi in zip(prob, y)] for prob in probs]
        return y_probs


class LERG(Explainer):
    """
    The base class for all Local Explanation methods for Response Generation
    """
    def __init__(self, model_f, x, y, perturb_f, tokenizer, max_iters=50, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_f, x, y)
        self.perturb_inputs = perturb_f
        self.max_iters = max_iters
        self.tokenizer = tokenizer
        self.device = device

    def combine_sequence(self, phi_sets):
        """
        phi_sets.shape: (output_dim) x (input_dim)
        """
        return torch.sum(phi_sets,dim=0)

    def map_to_interactions(self, phi_sets):
        phi_map = {}
        for yi in range(phi_sets.shape[0]):
            for xi in range(phi_sets.shape[1]):
                phi_map[(xi,yi)] = phi_sets[yi,xi]
        return phi_map

    def save_exp(self, save_path):
        if self.phi_set is not None:
            torch.save([self.phi_set, self.phi_map, self.components, self.y], save_path)
        else:
            raise ValueError("run get_local_exp() first")


class LERG_LIME(LERG):
    """
    LERG by LIME
    """
    def __init__(self, model_f, x, y, perturb_f, tokenizer, max_iters=50, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_f, x, y, perturb_f, tokenizer, max_iters, device=device)
        self.batchsize = 64

    def get_local_exp(self):
        self.x = self.tokenizer.tokenize(self.x)
        self.y = self.tokenizer.tokenize(self.y)

        x_set, z_set, self.components = self.perturb_inputs(self.x)

        y_probs = []
        for i in range(len(x_set)//self.batchsize + 1 if len(x_set)%self.batchsize > 0 else 0):
            probs,y = self.model_f(x_set[i*self.batchsize:(i+1)*self.batchsize], label=self.y, is_x_tokenized=True, is_y_tokenized=True)
            y_probs_batch = self.get_prob(probs, y)
            y_probs_batch = torch.tensor(y_probs_batch)
            y_probs.append(y_probs_batch)
        y_probs = torch.cat(y_probs,dim=0)

        D = pairwise_distances(z_set,z_set[0].view(1,-1),metric='cosine')
        kernel_width = 25# as LIME's original implementation
        weights = torch.tensor(np.sqrt(np.exp(-(D ** 2) / kernel_width ** 2)), requires_grad=False).to(self.device)

        self.expl_model = nn.Linear(z_set.shape[1],len(y),bias=False).to(self.device)
        self.optimizer = torch.optim.SGD(self.expl_model.parameters(), lr=5e-1)

        for i in range(self.max_iters):
            for z_batch, y_probs_batch, w_batch in zip(torch.split(z_set, self.batchsize), torch.split(y_probs, self.batchsize), torch.split(weights,self.batchsize)):
                z_batch = z_batch.to(self.device)
                y_probs_batch = y_probs_batch.to(self.device)
                preds = self.expl_model(z_batch)# the original version for classifier
                loss = torch.mean(w_batch * (preds - y_probs_batch) ** 2)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        with torch.no_grad():
            phi_sets = self.expl_model.weight
            self.phi_set = self.combine_sequence(phi_sets)
            self.phi_map = self.map_to_interactions(phi_sets)

        return self.phi_set, self.phi_map, self.components, self.y


class LERG_R(LERG):
    """
    LERG use ratio probability
    """
    def __init__(self, model_f, x, y, perturb_f, tokenizer, max_iters=50, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_f, x, y, perturb_f, tokenizer, max_iters, device=device)
        self.batchsize = 64

    def get_local_exp(self):
        self.x = self.tokenizer.tokenize(self.x)
        self.y = self.tokenizer.tokenize(self.y)
        x_set, z_set, self.components = self.perturb_inputs(self.x)

        gold_probs,y = self.model_f([self.x], label=self.y, is_x_tokenized=True, is_y_tokenized=True)
        gold_probs = self.get_prob(gold_probs, y)
        gold_probs = torch.tensor(gold_probs)
        gold_probs = gold_probs[0]

        probs,y = self.model_f(x_set, label=self.y, is_x_tokenized=True, is_y_tokenized=True)
        y_probs = self.get_prob(probs, y)
        y_probs = torch.tensor(y_probs)
        y_probs /= gold_probs

        self.expl_model = nn.Linear(z_set.shape[1],len(y),bias=False).to(self.device)
        self.optimizer = torch.optim.SGD(self.expl_model.parameters(), lr=5e-1)

        for i in range(self.max_iters):
            for z_batch, y_probs_batch in zip(torch.split(z_set, self.batchsize), torch.split(y_probs, self.batchsize)):
                z_batch = z_batch.to(self.device)
                y_probs_batch = y_probs_batch.to(self.device)
                preds = self.expl_model(z_batch)
                loss = F.mse_loss(preds,y_probs_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        with torch.no_grad():
            phi_sets = self.expl_model.weight
            self.phi_set = self.combine_sequence(phi_sets)
            self.phi_map = self.map_to_interactions(phi_sets)

        return self.phi_set, self.phi_map, self.components, self.y


class LERG_SHAP(LERG):
    """
    LERG use SampleShapley (original)
    """
    def __init__(self, model_f, x, y, perturb_f, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_f, x, y, perturb_f, tokenizer, max_iters=0, device=device)

    def get_local_exp(self):
        self.x = self.tokenizer.tokenize(self.x)
        self.y = self.tokenizer.tokenize(self.y)

        phi_sets = []
        for i in range(len(self.x)):
            x_set, x_set_with_i, weights, self.components = \
                self.perturb_inputs(self.x, num=500//len(self.x), with_i=i)# results in total 1000 samples as LERG_LIME

            probs,y = self.model_f(x_set, label=self.y, is_x_tokenized=True, is_y_tokenized=True)
            y_probs = self.get_prob(probs, y)
            y_probs = torch.tensor(y_probs)

            probs, _ = self.model_f(x_set_with_i, label=self.y, is_x_tokenized=True, is_y_tokenized=True)
            y_probs_with_i = self.get_prob(probs, y)
            y_probs_with_i = torch.tensor(y_probs_with_i)
            weights = torch.tensor(weights).view(-1,1)
            phi_sets.append(torch.mean((y_probs_with_i - y_probs)*weights, dim=0))

        phi_sets = torch.stack(phi_sets).transpose(0,1)
        self.phi_set = self.combine_sequence(phi_sets)
        self.phi_map = self.map_to_interactions(phi_sets)

        return self.phi_set, self.phi_map, self.components, self.y


class LERG_SHAP_log(LERG):
    """
    LERG use Shapley value with sample mean (Logarithm)
    """
    def __init__(self, model_f, x, y, perturb_f, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_f, x, y, perturb_f, tokenizer, max_iters=0, device=device)

    def get_local_exp(self):
        self.x = self.tokenizer.tokenize(self.x)
        self.y = self.tokenizer.tokenize(self.y)

        phi_sets = []
        for i in range(len(self.x)):
            x_set, x_set_with_i, weights, self.components = \
                self.perturb_inputs(self.x, num=500//len(self.x), with_i=i)# results in total 1000 samples as LERG_LIME

            probs,y = self.model_f(x_set, label=self.y, is_x_tokenized=True, is_y_tokenized=True)
            y_probs = self.get_prob(probs, y)
            y_probs = torch.tensor(y_probs)

            probs, _ = self.model_f(x_set_with_i, label=self.y, is_x_tokenized=True, is_y_tokenized=True)
            y_probs_with_i = self.get_prob(probs, y)
            y_probs_with_i = torch.tensor(y_probs_with_i)
            phi_sets.append(torch.mean((torch.log(y_probs_with_i) - torch.log(y_probs)), dim=0))

        phi_sets = torch.stack(phi_sets).transpose(0,1)
        self.phi_set = self.combine_sequence(phi_sets)
        self.phi_map = self.map_to_interactions(phi_sets)

        return self.phi_set, self.phi_map, self.components, self.y
