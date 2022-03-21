from target_models import GPT
from lerg.metrics import ppl_c, ppl_c_add
from lerg.visualize import plot_interactions

import tqdm
import sys
import json
import torch
import numpy as np
import os
from datetime import datetime
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--explain_method",type=str,required=True,
    help="Choose from 'LERG_S', 'LERG_L', 'SHAP', 'LIME', 'attn', 'grad', 'none'(random)")
parser.add_argument("--time_stamp",type=str,required=True,
    help="None for 'attn','grad','none'(random); for others, the time stamp in format '%m%d%Y_%H%M%S' of the saved explanations after runing 'explain.py'")
parser.add_argument("--model_dir",type=str,required=True,
    help="Directory of the trained target model")
parser.add_argument("--data_path",type=str,required=True,
    help="Path of the data for explaining the target model on")
parser.add_argument("--plot",action='store_true',
    help="If true, plot the interactions (maps) for all data points")
args = parser.parse_args()

def read_data(data_path):
    with open(data_path,"r") as fin:
        raw_data = json.load(fin)
    data = [(line["history"][-1], line["gt_response"]) for line in raw_data["test"]]
    return data

pplc_r_ratios = [0.1,0.2,0.3,0.4,0.5]
ppl_a_ratios = [0.5,0.6,0.7,0.8,0.9]

def evaluate_exp(tokenizer, model_f, data_path):
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
    if args.plot:
        if not os.path.exists("plots/{}/".format(args.explain_method)):
            os.mkdir("plots/{}".format(args.explain_method))
    if args.explain_method == "attn" or args.explain_method == "none" or args.explain_method == "grad":
        for x, y in tqdm.tqdm(data):
            if len(tokenizer.tokenize(x)) <= 30 and len(tokenizer.tokenize(y)) <= 30:
                if args.explain_method != "none":
                    phi_set, phi_map, x_components, y_components = model_f([x],y,output_type=args.explain_method)
                    if args.plot:
                        plot_interactions(phi_map,x_components,y_components,save_path='plots/{}/{}_{}.png'.format(args.explain_method, example_id, args.time_stamp))
                else:
                    phi_set, phi_map = None, None
                    x_components = tokenizer.tokenize(x)
                    y_components = tokenizer.tokenize(y)
                count_stats(phi_set, phi_map, x_components, y_components, model_f)
                count += 1
            example_id += 1
    else:
        for x, y in tqdm.tqdm(data):
            exp_path = 'exp/{}_{}_{}.exp'.format(args.explain_method, example_id, args.time_stamp)
            if os.path.exists(exp_path):
                phi_set, phi_map, x_components, y_components = torch.load(exp_path)
                if args.plot:
                    plot_interactions(phi_map,x_components,y_components,save_path='plots/{}/{}_{}.png'.format(args.explain_method, example_id, args.time_stamp))
                count_stats(phi_set, phi_map, x_components, y_components, model_f)
                count += 1
            example_id += 1
    print(count)
    print("PPLC_R:{}".format([np.exp(-pplc_r/count) for pplc_r in avg_pplc]))
    print("PPL_A:{}".format([np.exp(-pplc_a/count) for pplc_a in avg_pplc_add]))


if __name__ == "__main__":
    model = GPT(model_dir=args.model_dir)
    evaluate_exp(model.tokenizer, model.forward, args.data_path)
