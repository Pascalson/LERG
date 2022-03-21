from lerg.perturbation_models import RandomPM, LIMERandomPM
from lerg.RG_explainers import LERG_LIME, LERG_R, LERG_SHAP, LERG_SHAP_log
from target_models import GPT

import torch
import tqdm
import sys
import json
import os
from datetime import datetime
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--explain_method",type=str,required=True,
    help="Choose from 'LERG_S', 'LERG_L', 'SHAP', 'LIME'")
parser.add_argument("--model_dir",type=str,required=True,
    help="Directory of the trained target model")
parser.add_argument("--data_path",type=str,required=True,
    help="Path of the data for explaining the target model on")
args = parser.parse_args()

def read_data(data_path):
    with open(data_path,"r") as fin:
        raw_data = json.load(fin)
    data = [(line["history"][-1], line["gt_response"]) for line in raw_data["test"]]
    return data

def explain_dataset(explainer, model_f, tokenizer, data_path):
    if isinstance(explainer, tuple):
        PM, LERG = explainer
        perturb_f = PM.perturb_inputs
    else:
        LERG = explainer
        perturb_f = None
    data = read_data(data_path)
    avg_pplc = 0
    example_id = 0
    now = datetime.now()
    nowstr = now.strftime("%m%d%Y_%H%M%S")
    if not os.path.exists("exp/"):
        os.mkdir("exp")
    for x, y in tqdm.tqdm(data):
        # experiment on sentences with length less than 30, such that can get explanation using 8G GPU
        if len(tokenizer.tokenize(x)) <= 30 and len(tokenizer.tokenize(y)) <= 30:
            local_exp = LERG(model_f, x, y, perturb_f, tokenizer)
            phi_set, phi_map, x_components, y_components = local_exp.get_local_exp()
            save_path = 'exp/{}_{}_{}.exp'.format(args.explain_method, example_id, nowstr)
            local_exp.save_exp(save_path)
        example_id += 1

if __name__ == "__main__":
    PM = RandomPM()
    if args.explain_method == "LIME":
        PM = LIMERandomPM()
        explainer = (PM, LERG_LIME)
    elif args.explain_method == "LERG_L":
        PM = LIMERandomPM()
        explainer = (PM, LERG_LIME_R)
    elif args.explain_method == "SHAP":
        explainer = (PM, LERG_SHAP)
    elif args.explain_method == "LERG_S":
        explainer = (PM, LERG_SHAP_log)
    else:
        raise ValueError("select an explainer from \{'LIME', 'SHAP', 'LERG_L', 'LERG_S'\}, currently is {}".format(args.explain_method))
    model = GPT(model_dir=args.model_dir)
    explain_dataset(explainer, model.forward, model.tokenizer, args.data_path)
