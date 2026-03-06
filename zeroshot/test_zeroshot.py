import os
import random
import yaml
from tqdm import tqdm
import numpy as np
import torch

import sys
sys.path.append("../utils")
import utils_builder
from zeroshot_val import zeroshot_eval

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device_id = 'cuda'

config = yaml.load(open("zeroshot_config.yaml", "r"), Loader=yaml.FullLoader)

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

# Build FOCAL model and load checkpoint
model = utils_builder.FOCAL(config['network'])
ckpt_path = 'your_ckpt_path'   # path to focal_*_best_ckpt.pth or focal_*_final_total.pth
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt)
model = model.to(device_id)
model = torch.nn.DataParallel(model)

args_zeroshot_eval = config['zeroshot']

avg_auc = 0.0
for set_name in args_zeroshot_eval['test_sets'].keys():
    f1, acc, auc, _, _, _, res_dict = zeroshot_eval(
        model=model,
        set_name=set_name,
        device=device_id,
        args_zeroshot_eval=args_zeroshot_eval,
    )
    avg_auc += auc

avg_auc /= len(args_zeroshot_eval['test_sets'].keys())
print(f'\nAverage AUROC across all test sets: {avg_auc:.2f}')
