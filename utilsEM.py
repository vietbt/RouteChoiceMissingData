import collections
import numpy as np
import torch
import random

from numpy import ma
import scipy.sparse as sparse
from scipy import special as misc


def get_links_from_observed_1based_data(seq_smat, nlink):
    starts = seq_smat[:,1].toarray().squeeze() # init_links
    n_observed_obs = seq_smat.shape[0]
    links = [None] * n_observed_obs
    successors_no_dummy = [None] * n_observed_obs
    sizes = np.zeros(n_observed_obs).astype(int)
    ends = []
    for i in range(n_observed_obs):
        links[i] = seq_smat.data[seq_smat.indptr[i]+1:seq_smat.indptr[i+1]-3].astype(int)
        successors_no_dummy[i] = seq_smat.data[seq_smat.indptr[i]+2:seq_smat.indptr[i+1]-2].astype(int)
        sizes[i] = len(links[i])
        if seq_smat.data[ seq_smat.indptr[i+1]-3 ] != seq_smat.data[ seq_smat.indptr[i+1]-2]:
            ends.append(seq_smat.data[seq_smat.indptr[i+1]-3])
        else:
            ends.append(nlink+1)
    ends = np.array(ends)
    links = np.concatenate(links)
    successors_no_dummy = np.concatenate(successors_no_dummy)
    dummies = seq_smat[:,0].toarray().squeeze()
    indptr = np.zeros(len(sizes)+1).astype(int)
    indptr[1:] = np.cumsum(sizes)
    starts = (starts-1).astype(int)
    ends = (ends-1).astype(int)
    links = (links-1).astype(int)
    successors_no_dummy = (successors_no_dummy-1).astype(int)
    dummies = (dummies-1).astype(int)
    return starts, ends, links, successors_no_dummy, dummies, indptr


def BFS(slink, elink, adjlist, max_depth=30, mark_visited=True, seed=None, explore_len=2):
    if seed is not None:
        random.seed(seed)
    shortest_path_only = False
    visited = set([slink])
    all_success_paths = []
    bfs_list = collections.deque([(slink, collections.deque([slink]), 0)]) 
    while True:
        if len(bfs_list) == 0:
            break
        s, path, depth = bfs_list.popleft()
        n_connect_s = int(adjlist[s,0])
        if n_connect_s == 0:
            continue 
        for j in random.sample(list(adjlist[s,1:(n_connect_s+1)]), n_connect_s):
            path_j = path.copy()
            path_j.append(j)
            if j == elink:
                all_success_paths.append(path_j)
                if shortest_path_only:
                    break
            else:
                if not mark_visited or j not in visited:
                    bfs_list.append((j, path_j, depth+1))
                if mark_visited:
                    visited.add(j)
        if depth > max_depth-explore_len and len(all_success_paths) == 0:
                shortest_path_only = True
                mark_visited = True
        if depth > max_depth and not shortest_path_only:
            break
    if len(all_success_paths) == 0:
        raise Exception("cannot find paths {}->{}!".format(slink, elink))
    for i in range(len(all_success_paths)):
        all_success_paths[i].appendleft(len(all_success_paths[i])) # sequence length
        all_success_paths[i] = np.array(all_success_paths[i])
    return all_success_paths, shortest_path_only

def get_adjlist(incidence_smat_no_dummy):
    nlink = incidence_smat_no_dummy.shape[0]
    n_connect = np.array(incidence_smat_no_dummy.sum(axis=1)).reshape(-1,).astype(int)
    n_max = np.max(n_connect)
    adjlist = np.zeros([nlink, n_max+1]).astype(int) 
    adjlist[:,0] = n_connect
    for i in range(nlink):
        start = incidence_smat_no_dummy.indptr[i]
        end = incidence_smat_no_dummy.indptr[i+1]
        n_connect_i = n_connect[i]
        adjlist[i,1:(n_connect_i+1)] = incidence_smat_no_dummy.indices[start:end]
    return adjlist

def padding_zeros_list_of_1d_nparrays(arrays, n_padding_col=0):
    maxlen = max(len(a) for a in arrays)
    arr = np.zeros((len(arrays), maxlen+n_padding_col))
    for i,row in enumerate(arrays):
        arr[i,:len(row)] += row
    return arr

def padding_zeros_list_of_2d_nparrays(arrays, n_padding_col=0):
    maxlen = max(a.shape[1] for a in arrays)
    nrow = np.sum([a.shape[0] for a in arrays])
    arr = np.zeros((nrow, maxlen+n_padding_col))
    start = 0
    for mat in arrays:
        end = start+mat.shape[0]
        arr[start:end,:mat.shape[1]] += mat
        start = end
    return arr

def compute_probs_from_seq_1based_smat(seq_0based_mat, policy, is_able_to_connect_missing_segs, sample_path_lens):
    n = seq_0based_mat.shape[0]
    if not is_able_to_connect_missing_segs:
        return np.zeros(n)
    lprobs = np.zeros(n)
    for i in range(n):
        l = sample_path_lens[i]
        links = seq_0based_mat[i,1:l]
        successors = seq_0based_mat[i,2:(l+1)]
        probs_i = np.array(policy[links, successors])#.toarray()
        lprobs_i = ma.log(probs_i).filled(-1e200)
        lprobs[i] = lprobs_i.sum()
    lprobs = lprobs - misc.logsumexp(lprobs)
    probs = np.exp(lprobs)
    if np.abs(np.sum(probs)-1.0) > 1e-5:
        probs = np.zeros_like(probs)
    return probs

def generate_samples_for_missing_data_subset(missing_seg, adjlist, max_depth, mark_visited, seed):
    dummy = missing_seg[0]
    slink = missing_seg[1]
    elink = missing_seg[2]
    dest = missing_seg[3]
    paths, shortest_path_only = BFS(slink-1, elink-1, adjlist, max_depth, mark_visited, seed=seed)
    paths = [p + 1 for p in paths]
    path_arr = padding_zeros_list_of_1d_nparrays(paths, n_padding_col=2)
    path_lens = path_arr[:,0].astype(int) - 1
    path_arr[np.array(range(path_arr.shape[0])), path_lens+1] = dest
    path_arr[np.array(range(path_arr.shape[0])), path_lens+2] = dummy
    path_arr[:,0] = dummy
    return path_arr, shortest_path_only, path_lens