"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""

import math
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np
from tqdm import tqdm as tk
from datetime import datetime


import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding, GraphAttentionEmbeddingExp
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from modules import my_utils


# ==========
# ========== 
# ==========


def load_model(model, model_path):
    # print('MODEL_PATH', model_path)
    checkpoint = torch.load(model_path)
    for model_name, model_ in model.items():
        model_.load_state_dict(checkpoint[model_name])
    return model


def load_neighbors_before_test():
    model['memory'].reset_state()  
    neighbor_loader.reset_state()  

    for batch in tk(train_loader):
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neighbor_loader.insert(src, pos_dst)


def tensor_reset():
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()  
    neighbor_loader.reset_state()  

@torch.no_grad()
def test(loader, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []
    perf_list_hits = []

    for pos_batch in tk(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }

            res = evaluator.eval(input_dict)

            perf_list.append(evaluator.eval(input_dict)[metric])
            perf_list_hits.append(evaluator.eval(input_dict)[f'hits@{HITS_K}'])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())
    perf_metrics_hits = float(torch.tensor(perf_list_hits).mean())

    return perf_metrics, perf_metrics_hits


def log_model_details(model):
    print('-'*100)
    print(f'Model: {model}')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print('-'*100)


def log_model_dict_details(model_dict):
    """
    Given a dictionary of PyTorch models, returns a dictionary
    mapping model names to (total_params, trainable_params).
    """
    param_counts = {}
    total_all = 0
    trainable_all = 0
    print('Model parameters: (Total, Trainable) ')
    for name, model in model_dict.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[name] = (total, trainable)
        total_all += total
        trainable_all += trainable
    param_counts["Model"] = (total_all, trainable_all)
    print(param_counts)
    # return param_counts

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()
DATA = "tgbl-wiki"

# ========== set parameters...
args, _ = get_args()
args.num_epoch = 200
args.patience = 25
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
HITS_K = 10
# ========== set parameters...

# Dataset to use from ['tgbl-wiki', 'tgbl-review', 'tgbl-coin', 'tgbl-comment', 'tgbl-flight']
DATA = 'tgbl-wiki'

# Experiment name for reference and logging from ['TGN', 'GM', 'LETE']
EXP_NAME = 'TGN'

# TGNN backbone used
MODEL_NAME = 'TGN'

# Use of kernel (default is Laplacian)
USE_KERNEL = True

# Time encoding methods from ['TGN', 'GM', 'LETE']
TIME_ENC_METHOD = EXP_NAME
# ==========
args.time_enc_method = TIME_ENC_METHOD
args.use_kernel = USE_KERNEL
# ==========
print("INFO: Arguments:", args)

if args.use_kernel:
    EXP_NAME = EXP_NAME + '_LAP'


# ==========
STD_TIME_DICT = {
    'tgbl-wiki': 80111.0 ,
    'tgbl-review': 19224004.0, 
    'tgbl-coin':756107.0,
    'tgbl-comment':1055184.0,
    'tgbl-flight':1852987.0
}
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

# print(f'K_VALUE: {HITS_K}')
print(f'EXP_NAME: {EXP_NAME}')

evaluator = Evaluator(name=DATA, k_value=HITS_K)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results/results_pretrained_base/'
if not osp.exists(results_path):
    os.makedirs(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{EXP_NAME}_{MODEL_NAME}_{DATA}_results.json'

val_mrr_runs = []
test_mrr_runs = []

val_mrr_runs_hits = []
test_mrr_runs_hits = []

for run_idx in range(NUM_RUNS):
# for run_idx in range(4,5):

    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)


    # neighborhood sampler
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

    # define the model end-to-end
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
        time_enc_method=TIME_ENC_METHOD
    ).to(device)

    if args.use_kernel:
        gnn = GraphAttentionEmbeddingExp(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
            std_time = STD_TIME_DICT[args.data]
        ).to(device)
    else:
        gnn = GraphAttentionEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)


    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

    model = {'memory': memory,
            'gnn': gnn,
            'link_pred': link_pred}

    print('*'*100)
    # print(model)
    log_model_dict_details(model)
    # log_model_details(model['memory'])
    # log_model_details(model['gnn'])
    # log_model_details(model['link_pred'])
    print('*'*100)
    # exit()

    load_neighbors_before_test()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
   
    # loading the validation negative samples
    dataset.load_val_ns()
    tensor_reset()

    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models_pretrained/{EXP_NAME}'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'

    model_path = save_model_dir + '/' + save_model_id + '.pth'
    model = load_model(model, model_path)

    # loading the test negative samples
    dataset.load_test_ns()

    start_test = timeit.default_timer()
    perf_metric_val_mrr, perf_metric_val_hits = test(val_loader, neg_sampler, split_mode="val")
    perf_metric_test_mrr, perf_metric_test_hits = test(test_loader, neg_sampler, split_mode="test")

    val_mrr_runs.append(perf_metric_val_mrr)
    test_mrr_runs.append(perf_metric_test_mrr)

    val_mrr_runs_hits.append(perf_metric_val_hits)
    test_mrr_runs_hits.append(perf_metric_test_hits)

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    # print(f"\tVal: {metric}: {perf_metric_val_mrr: .4f}, Hits@k: {perf_metric_val_hits: .4f}")
    # print(f"\tTest: {metric}: {perf_metric_test_mrr: .4f}, Hits@k: {perf_metric_test_hits: .4f}")
    print(f"\tVal: {metric}: {perf_metric_val_mrr: .4f}")
    print(f"\tTest: {metric}: {perf_metric_test_mrr: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
    logging_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  f'val {metric}': perf_metric_val_mrr,
                  f'val val_hits': perf_metric_val_hits,
                  f'test {metric}': perf_metric_test_mrr,
                  f'test test_hits': perf_metric_test_hits,
                  'val_test_time': test_time,
                  'logging_time': logging_time
                  }, 
    results_filename)


    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

for i in range(NUM_RUNS):
    print('*'*100)
    print(f'run_idx: {i}')
    print(f'val_mrr: {val_mrr_runs[i]:.4f}')
    print(f'test_mrr: {test_mrr_runs[i]:.4f}')
    # print(f'val_hits_{HITS_K}: {val_mrr_runs_hits[i]:.4f}')
    # print(f'test_hits_{HITS_K}: {test_mrr_runs_hits[i]:.4f}')

print('*'*100)
print(f'val_mrr_mean: {np.mean(val_mrr_runs):.4f}')
print(f'val_mrr_std: {np.std(val_mrr_runs):.4f}')
print(f'test_mrr_mean: {np.mean(test_mrr_runs):.4f}')
print(f'test_mrr_std: {np.std(test_mrr_runs):.4f}')
print('*'*100)

# print(f'{np.mean(val_mrr_runs_hits):.4f}, {np.std(val_mrr_runs_hits):.4f}')
# print(f'{np.mean(test_mrr_runs_hits):.4f}, {np.std(test_mrr_runs_hits):.4f}')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")

print('val')

for i in range(NUM_RUNS):
    print(f'{val_mrr_runs[i]:.4f}')

print('test')

for i in range(NUM_RUNS):
    print(f'{test_mrr_runs[i]:.4f}')
