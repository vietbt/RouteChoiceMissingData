import os
import pickle
import numpy as np
import scipy.sparse as sparse
import torch
from solve import Solve
from tqdm import tqdm
from joblib import Memory
memory = Memory('__pycache__', verbose=0)

@memory.cache
def generate_data(observation_smat, missing_prob, consecutive, seed):
    nobs = observation_smat.shape[0]
    data = observation_smat.data
    ptr = observation_smat.indptr
    output = []
    missing_seg = []
    for i in range(nobs):
        seed_i = None if seed is None else seed + i
        tlen = ptr[i+1] - ptr[i] - 2
        if tlen < 2:
            continue
        full_trajectory = data[ptr[i]:ptr[i+1]]
        observed_seq_i, missing_seg_i = get_random_missing_obs_idxs_from_a_trajectory(
            full_trajectory, tlen, missing_prob, consecutive, seed_i)
        missing_seg.extend(missing_seg_i)
        if len(observed_seq_i) == 0:
            continue
        observed_seq_i_data = np.concatenate(observed_seq_i)
        observed_seq_i_len = np.array(list((map(len, observed_seq_i))))
        output.append((observed_seq_i_data, observed_seq_i_len, missing_seg))
        missing_seg = []
    return output

@memory.cache
def load_data(path="dataset"):
    incidence = sparse.load_npz(f'{path}/incidence.npz')
    lefturn = sparse.load_npz(f'{path}/lefturn.npz')
    uturn = sparse.load_npz(f'{path}/uturn.npz')
    travel_time = sparse.load_npz(f'{path}/traveltime.npz')
    observation = sparse.load_npz(f'{path}/observation.npz')
    nlink = incidence.shape[0]
    link_size = load_link_size_data(f'{path}/LS', nlink)
    return observation, incidence, lefturn, uturn, travel_time, link_size

def convert_to_sparse_mat(data, square=False, dim=0):
    ndims = [int(np.max(data[:, 0])), int(np.max(data[:, 1]))]
    if square and ndims[0] != ndims[1]:
        if ndims[1-dim] > ndims[dim]:
            m = sparse.csr_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)), ndims)
            ndims[1-dim] = ndims[dim]
            m = m[:ndims[0], :ndims[1]]
        else:
            ndims[1-dim] = ndims[dim]
            m = sparse.csr_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)), ndims)
    else:
        m = sparse.csr_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)), ndims)
    return m

def append_zero_to_csr_matrix(csr_mat, n_zero_col=1, n_zero_row=1):
    indptr = np.concatenate([csr_mat.indptr, [csr_mat.indptr[-1]] * n_zero_row])
    zero_appended_csr_mat = sparse.csr_matrix((csr_mat.data, csr_mat.indices, indptr), shape=(csr_mat.shape[0] + n_zero_row, csr_mat.shape[1] + n_zero_col))
    return zero_appended_csr_mat

def load_link_size(filename, nlink):
    data = np.loadtxt(filename)
    smat = convert_to_sparse_mat(data, True, 0)
    row, col = smat.shape
    smat = append_zero_to_csr_matrix(smat, nlink-row, nlink-col)
    return smat

@memory.cache
def load_link_size_data(link_size_folder, nlink):
    all_LS = []
    for filename in os.listdir(link_size_folder):
        if filename.endswith(".txt"):
            path = f"{link_size_folder}/{filename}"
            smat = load_link_size(path, nlink)
            all_LS.append(smat)
    return all_LS

@memory.cache(ignore=['LS', 'OL_indices'])
def load_beta_ls(LS, OL_indices, nlink):
    beta_LS = []
    for ls in tqdm(LS, "load_beta_ls"):
        ls = scipy_sparse_to_tensor(ls, [nlink, nlink]).to_dense()
        ls = ls[OL_indices[1], OL_indices[2]]
        beta_LS.append(ls)
    beta_LS = torch.stack(beta_LS)
    return beta_LS

def get_random_missing_obs_idxs_from_a_trajectory(full_trajectory, tlen, missing_prob, consecutive=False, seed=None):
    all_observed_seq, all_missing_seg = get_random_observed_missing_idxs(tlen, missing_prob, shift=1, consecutive=consecutive, seed=seed)
    dummy = full_trajectory[0]
    dest = full_trajectory[tlen]
    for i, observed_seq in enumerate(all_observed_seq):
        observed_seq = np.concatenate([[dummy], full_trajectory[observed_seq], [dest], [dummy]])
        all_observed_seq[i] = observed_seq
    for i, missing_seg in enumerate(all_missing_seg):
        missing_seg = np.concatenate([[dummy], full_trajectory[missing_seg], [dest], [dummy]])
        all_missing_seg[i] = missing_seg
    return all_observed_seq, all_missing_seg


def get_random_observed_missing_idxs(tlen, missing_prob, shift=0, consecutive=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    bernoulli_seq = np.concatenate([[1.0], np.random.rand(tlen-2), [1.0]])
    observed_missing_seq = np.zeros_like(bernoulli_seq)
    observed_missing_seq[bernoulli_seq >= missing_prob] = 1.0
    if consecutive:
        n_missing = int(len(observed_missing_seq) - observed_missing_seq.sum())
        start_missing_idx = np.random.randint(1, len(observed_missing_seq) - n_missing)
        observed_missing_seq = np.ones_like(observed_missing_seq)
        observed_missing_seq[start_missing_idx:start_missing_idx+n_missing] = 0.0
    all_observed_seq = []
    all_missing_seg = []
    observed_seq = [0 + shift]
    for i in range(1, len(observed_missing_seq)):
        if observed_missing_seq[i]:
            shifted_i = i + shift
            if shifted_i - observed_seq[-1] == 1:
                observed_seq.append(shifted_i)
            else:
                if len(observed_seq) > 1:
                    all_observed_seq.append(np.array(observed_seq))
                all_missing_seg.append(np.array([observed_seq[-1], shifted_i]))
                observed_seq = [shifted_i]
    if len(observed_seq) > 1:
        all_observed_seq.append(np.array(observed_seq))
    return all_observed_seq, all_missing_seg

def get_csr_from_data_length(data, lengths):
    ncol = np.max(lengths)
    nrow = len(lengths)
    indices = np.concatenate([np.array(range(l)) for l in lengths])
    indptr = np.cumsum(np.concatenate([[0], lengths]))
    csr_mat = sparse.csr_matrix((data, indices, indptr), shape=(nrow, ncol))
    return csr_mat

def form_B(nlink, dummy_incidence_smat):
    n_distinct_dest = dummy_incidence_smat.shape[1]
    new_indptr = np.concatenate([dummy_incidence_smat.indptr, [dummy_incidence_smat.indptr[-1]]])
    t = sparse.csr_matrix((dummy_incidence_smat.data, dummy_incidence_smat.indices, new_indptr), shape=(nlink + 1, n_distinct_dest)).toarray()
    b = np.zeros([nlink + 1, n_distinct_dest])
    b[nlink, :] = 1.0
    B = t + b
    B = torch.FloatTensor(B).unsqueeze(0)
    return B

def form_sparse_M(ids, ireward_data, mu, nlink):
    ireward_data = ireward_data / mu[ids[1]]
    M_data = torch.exp(ireward_data).cpu()
    sparse_M_shape = (1, nlink + 1, nlink + 1)
    sparse_M = torch.sparse_coo_tensor(ids, M_data, sparse_M_shape)
    ids = torch.stack([ids[0], ids[2], ids[1]])
    # sparse_M_shape = (nlink + 1, nlink + 1)
    sparse_M_transpose = torch.sparse_coo_tensor(ids, M_data, sparse_M_shape)
    return sparse_M, sparse_M_transpose

def form_M(ids, ireward_data, mu, nlink):
    ireward_data = ireward_data / mu[ids[1]]
    M_data = torch.exp(ireward_data)
    M_shape = (nlink + 1, nlink + 1)
    M = torch.zeros(M_shape, device=M_data.device)
    M[ids[1], ids[2]] = M_data
    return M

def scipy_sparse_to_tensor(data, size=None) -> torch.FloatTensor:
    data = data.tocoo()
    row = data.row.tolist()
    col = data.col.tolist()
    
    if size is not None:
        indices = np.stack([row, col])
        data = torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(data.data),
            size
        )
    else:
        indices = np.stack([np.zeros_like(row), row, col])
        data = torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(data.data)
        )
    return data

def get_Z(M, B, I):
    if not M.is_sparse:
        A = I.to_dense()[0,...].to(M.device) - M
        B = B.squeeze(0).double().to(M.device)
        Z = torch.linalg.solve(A.double(), B)
    else:
        A = I - M
        Z = Solve.apply(A, B)
        Z = torch.squeeze(Z, 0)
    return Z

def get_V(Z, mu):
    tmp = torch.log(Z)
    Z = torch.where(torch.isfinite(tmp), tmp, torch.ones_like(tmp)).type(mu.dtype)
    V = torch.unsqueeze(mu, -1) * Z
    return V

def get_loglikelihood(nlink, V, ireward_smat, mu, mu_cpu, init_links, end_links, links, sucessors_no_dummy, dummy_links, path_probs=None, link_probs=None):
    v = ireward_smat[links, sucessors_no_dummy] / mu[links]
    V_start = V[init_links, dummy_links - nlink] / mu_cpu[init_links]
    V_end = V[end_links, dummy_links - nlink]/ mu_cpu[end_links]

    if path_probs is not None and link_probs is not None:
        path_probs = torch.tensor(path_probs, dtype=V.dtype, device=V.device)
        link_probs = torch.tensor(link_probs, dtype=v.dtype, device=v.device)
        v = v * link_probs
        V_start = V_start * path_probs
        V_end = V_end * path_probs

    V_start = V_start.type(v.dtype)
    V_end = V_end.type(v.dtype)
    loglikelihood = torch.sum(v) - torch.sum(V_start) + torch.sum(V_end)
    return loglikelihood

def compute_Q(M, i, db):
    Q = M * torch.sparse_coo_tensor(i, db, M.size(), dtype=M.dtype)
    return Q

def compute_P(I, Q, D):
    if Q is None:
        return
    P = Solve.apply(I - Q, D).type(Q.dtype)
    return P

def compute_all_Q(M, Z):
    i = M._indices()
    d = torch.ones(M._nnz(), device=M.device, dtype=Z.dtype)
    d = d * Z[:, i[1, :]]
    d = d / Z[:, i[2, :]]
    all_Q = [compute_Q(M, i, db) for db in tqdm(d, 'compute_Q', leave=False)]
    return all_Q

def compute_all_P(I, all_Q, all_D):
    missing_P_transposes = [compute_P(I, Q, D) for Q, D in tqdm(zip(all_Q, all_D), 'compute_P', leave=False)]
    return missing_P_transposes

def get_log_path(logdir, method, seed, missing_prob, train_mu, use_LS, use_LS_for_beta):
    os.makedirs(logdir, exist_ok=True)
    logdir = os.path.join(logdir, f"{method}")
    os.makedirs(logdir, exist_ok=True)
    logdir = os.path.join(logdir, f"seed_{seed}")
    os.makedirs(logdir, exist_ok=True)
    logdir = os.path.join(logdir, f"prob_{missing_prob}")
    os.makedirs(logdir, exist_ok=True)
    logdir = os.path.join(logdir, "with_mu" if train_mu else "without_mu")
    os.makedirs(logdir, exist_ok=True)
    if use_LS:
        if use_LS_for_beta:
            logdir = os.path.join(logdir, "with_LS_beta")
        else:
            logdir = os.path.join(logdir, "with_LS")
    else:
        logdir = os.path.join(logdir, "without_LS")
    return logdir