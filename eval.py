from perturbation_models import BasicPM, RandomPM, RandomPhrasePM
from target_models import GPT

from metrics import ppl_c, ppl_c_add
from interaction_utils import plot_interactions

import tqdm
import sys
import json
import torch
import numpy as np
import os
from datetime import datetime

def read_data(data_path):
    with open(data_path,"r") as fin:
        raw_data = json.load(fin)
    data = [(line["history"][-1], line["gt_response"]) for line in raw_data["test"]]
    return data

pplc_r_ratios = [0.1,0.2,0.3,0.4,0.5]
ppl_a_ratios = [0.5,0.6,0.7,0.8,0.9]

def evaluate_exp(tokenizer, model_f, denoise_f, data_path):
    data = read_data(data_path)
    avg_pplc = [0 for _ in pplc_r_ratios]
    avg_pplc_add = [0 for _ in ppl_a_ratios]
    
    def count_stats(phi_set, phi_map, x_components, y_components, model_f):
        for i, r in enumerate(pplc_r_ratios):
            entc, x_re, _, _ = ppl_c(phi_set, x_components, y_components, model_f, ratio=r)
            avg_pplc[i] += entc
        for i, r in enumerate(ppl_a_ratios):
            ent_add, *_ = ppl_c_add(phi_set, x_components, y_components, model_f, ratio=r)
            avg_pplc_add[i] += ent_add

    example_id = 0
    count = 0
    if sys.argv[3] == "True":
        if not os.path.exists("plots/{}/".format(sys.argv[1])):
            os.mkdir("plots/{}".format(sys.argv[1]))
    if sys.argv[1] == "attn" or sys.argv[1] == "none" or sys.argv[1] == "grad":
        for x, y in tqdm.tqdm(data):
            if len(tokenizer.tokenize(x)) <= 30 and len(tokenizer.tokenize(y)) <= 30:
                if sys.argv[1] != "none":
                    phi_set, phi_map, x_components, y_components = model_f([x],y,output_type=sys.argv[1])
                    if sys.argv[3] == "True":
                        plot_interactions(phi_map,x_components,y_components,save_path='plots/{}/{}_{}.png'.format(sys.argv[1], example_id, sys.argv[2]))
                else:
                    phi_set, phi_map = None, None
                    x_components = tokenizer.tokenize(x)
                    y_components = tokenizer.tokenize(y)
                count_stats(phi_set, phi_map, x_components, y_components, model_f)
                count += 1
            example_id += 1
    else:
        for x, y in tqdm.tqdm(data):
            exp_path = 'exp/{}_{}_{}.exp'.format(sys.argv[1], example_id, sys.argv[2])
            if os.path.exists(exp_path):
                phi_set, phi_map, x_components, y_components = torch.load(exp_path)
                if sys.argv[3] == "True":
                    plot_interactions(phi_map,x_components,y_components,save_path='plots/{}/{}_{}.png'.format(sys.argv[1], example_id, sys.argv[2]))
                count_stats(phi_set, phi_map, x_components, y_components, model_f)
                count += 1
            example_id += 1
    print(count)
    print("PPLC_R:{}".format([np.exp(-pplc_r/count) for pplc_r in avg_pplc]))
    print("PPL_A:{}".format([np.exp(-pplc_a/count) for pplc_a in avg_pplc_add]))


if __name__ == "__main__":
    model = GPT(model_dir=sys.argv[4])
    evaluate_exp(model.tokenizer, model.forward, sys.argv[5])
