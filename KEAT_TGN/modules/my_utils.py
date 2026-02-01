import numpy as np 
from tqdm import tqdm as tk


def check_batch_wise_repetition_ratio(loader):
    perf_list = []
    batches = 0
    avg_rep_ratio = 0.0
    for pos_batch in tk(loader):
        repeat = {}
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )
        
        for src in pos_src:
            val = src.item()
            if(val in repeat):      
                repeat[val]+=1
            else:
                repeat[val] = 1
            # print(repeat[src])

        
        for dst in pos_dst:
            val = dst.item()
            if(val in repeat):      
                repeat[val]+=1
            else:
                repeat[val] = 1
        
        repeated = 0.0
        tot = 0.0
        for key in repeat.keys():
            tot+=1
            if repeat[key] > 1:
                repeated+=1

        # print(repeated/tot)
        avg_rep_ratio+= (repeated/tot)
        batches+=1

    return avg_rep_ratio/batches


def get_batch_repeat_ratio(loader):

    node_count_list = []

    for batch in tk(loader):
        node_count = {}
        src = batch.src 
        dst = batch.dst 

        for i in range(len(src)):

            src_node = src[i].item()
            dst_node = dst[i].item()

            if src_node in node_count:
                node_count[src_node] += 1
            else:
                node_count[src_node] = 1

            if dst_node in node_count:
                node_count[dst_node] += 1
            else:
                node_count[dst_node] = 1

        repeat_node_count = [1 for v in node_count.values() if v > 1]
        # val_sum = sum(node_count.values())
        # repeat_nodes_sum = sum(repeat_node_count)

        # Compute average if there are any such values
        if repeat_node_count:
            batch_average = sum(repeat_node_count) / (len(src)+len(dst))
        else:
            batch_average = 0 

        node_count_list.append(batch_average)

    return np.mean(node_count_list)


def get_inter_time_stats(data_in):
    arr_times = [x.t[0].item() for x in data_in]
    time_diff = np.asarray(arr_times[1:]) - np.asarray(arr_times[:-1])
    print(f'LEN: {len(arr_times)}')
    print(f'MIN time_diff: {min(time_diff)}, MAX time_diff: {max(time_diff)}')
    print(f'MEAN: {np.mean(time_diff):.4f}, STD: {np.std(time_diff):.4f}')


def get_avg_inter_edge_time(data_in):
    node_time_dict = {}
    for x in data_in:
        src = x.src.item()
        dst = x.dst.item()
        t = x.t.item()
        
        if src in node_time_dict:
            node_time_dict[src].append(t)
        else:
            node_time_dict[src] = [t]

        if dst in node_time_dict:
            node_time_dict[dst].append(t)
        else:
            node_time_dict[dst] = [t]

    mean_et_l = []
    std_et_l = []
    max_val = 0
    
    for key in node_time_dict:
        t_list = node_time_dict[key]
        if len(t_list) > 1:
            t_diff_curr = np.asarray(t_list[1:]) - np.asarray(t_list[:-1])
            mean_et_l.append(np.mean(t_diff_curr))
            std_et_l.append(np.std(t_diff_curr))
            if max_val < max(t_diff_curr):
                max_val = max(t_diff_curr)

    print(f'MAX: {max_val}')    
    print(f'MEAN: {np.mean(mean_et_l):.4f}, STD: {np.mean(std_et_l):.4f}')
    return mean_et_l, std_et_l