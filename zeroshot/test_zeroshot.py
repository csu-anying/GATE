import os
import random
import yaml as yaml
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

from models.Graph_text_model import ECG_Graph_Text_with_co
from zeroshot_eval import zeroshot_eval
from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]  # 去掉 'module.'
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device_id = 'cuda:0'

config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

model = ECG_Graph_Text_with_co(config['network'])
# model = ECG_Graph_Text(config['network'])
ckpt = '../checkpoints/pretrain/GATE_2500_with_co_Clinical_ModernBERT_node_edge_mask_bestZeroShotAll_ckpt.pth'
ckpt = torch.load(f'{ckpt}', map_location='cpu')
ckpt = remove_module_prefix(ckpt)
model.load_state_dict(ckpt)
model = model.to(device_id)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

args_zeroshot_eval = config['zeroshot']

avg_f1, avg_acc, avg_auc = 0, 0, 0
for set_name in args_zeroshot_eval['test_sets'].keys():
    f1, acc, auc, _, _, _, res_dict = \
        zeroshot_eval(model=model,
                      set_name=set_name,
                      device=device_id,
                      args_zeroshot_eval=args_zeroshot_eval)

    avg_f1 += f1
    avg_acc += acc
    avg_auc += auc

avg_f1 = avg_f1 / len(args_zeroshot_eval['test_sets'].keys())
avg_acc = avg_acc / len(args_zeroshot_eval['test_sets'].keys())
avg_auc = avg_auc / len(args_zeroshot_eval['test_sets'].keys())
