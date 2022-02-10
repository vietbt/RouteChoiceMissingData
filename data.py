import torch
from utils import get_csr_from_data_length
import numpy as np
import scipy.sparse as sparse

from utilsEM import generate_samples_for_missing_data_subset, get_links_from_observed_1based_data
import numpy as np
from tqdm import tqdm
from utilsEM import compute_probs_from_seq_1based_smat, get_adjlist, get_links_from_observed_1based_data, padding_zeros_list_of_2d_nparrays
from multiprocessing import Pool
from functools import partial

class DataLoader:

    def __init__(self, dataset, OL):
        self.nlink = OL.shape[0]
        self.dummy_incidence_smat = OL[:, self.nlink:]
        all_obs_seq_data = []
        all_obs_seq_len = []
        all_missing_seg = []
        for obs_data, obs_len, miss in dataset:
            all_obs_seq_data.append(obs_data)
            all_obs_seq_len.append(obs_len)
            all_missing_seg.extend(miss)
        self.process_obs(all_obs_seq_data, all_obs_seq_len)
        self.process_missing(all_missing_seg)

    def process_obs(self, all_obs_seq_data, all_obs_seq_len):
        observed_smat = get_csr_from_data_length(
            np.concatenate(all_obs_seq_data).astype(int), 
            np.concatenate(all_obs_seq_len).astype(int)
        )
        observed_seq_smat = observed_smat.copy()
        observed_seq_smat.data = observed_seq_smat.data - 1
        self.obs_starts = observed_seq_smat[:,1].toarray().squeeze()
        self.n_observed_obs = observed_seq_smat.shape[0]
        observed_links = [None] * self.n_observed_obs
        observed_successors_no_dummy = [None] * self.n_observed_obs
        observed_ends = []
        obs_ptr = observed_seq_smat.indptr
        obs_data = observed_seq_smat.data
        for i in range(self.n_observed_obs):
            observed_links[i] = obs_data[obs_ptr[i]+1:obs_ptr[i+1]-3].astype(int)
            observed_successors_no_dummy[i] = obs_data[obs_ptr[i]+2:obs_ptr[i+1]-2].astype(int)
            if obs_data[obs_ptr[i+1]-3] != obs_data[obs_ptr[i+1]-2]:
                observed_ends.append(obs_data[obs_ptr[i+1]-3])
            else:
                observed_ends.append(self.nlink)
        self.obs_ends = np.array(observed_ends)
        self.obs_links = np.concatenate(observed_links)
        self.obs_successors_no_dummy = np.concatenate(observed_successors_no_dummy)
        self.obs_dummies = observed_seq_smat[:,0].toarray().squeeze()

    def process_missing(self, all_missing_seg):
        if all_missing_seg is None or len(all_missing_seg) == 0:
            self.n_missing_segs = 0
            return
        missing_mat = np.stack(all_missing_seg).astype(int)
        self.network_dummies = np.array(range(self.dummy_incidence_smat.shape[1])) + self.nlink
        missing_seg_mat = missing_mat.copy()
        missing_seg_mat = missing_seg_mat - 1
        n_missing_segs = missing_seg_mat.shape[0]
        self.n_missing_segs = n_missing_segs
        missing_starts = missing_seg_mat[:,1]
        self.missing_ends = missing_seg_mat[:,2]
        missing_dummies = missing_seg_mat[:,0]
        j = 0
        missing_all_dummy_idxs = [0]
        for i,d in enumerate(self.network_dummies):
            j = missing_all_dummy_idxs[-1]
            while j < len(missing_dummies) and missing_dummies[j] == d:
                j += 1
            missing_all_dummy_idxs.append(j)
        self.missing_dummy_indptr = np.array(missing_all_dummy_idxs)
        data = np.ones(2*n_missing_segs)
        indices = np.concatenate(list(zip(missing_starts, self.nlink * np.ones(n_missing_segs).astype(int))))
        indptr = 2 * np.array(range(n_missing_segs+1)).astype(int)
        D_concat = sparse.csc_matrix((data, indices, indptr), shape=(self.nlink+1, n_missing_segs))
        with torch.no_grad():
            self.all_D = []
            for i in range(len(self.network_dummies)):
                D_sidx = self.missing_dummy_indptr[i]
                D_eidx = self.missing_dummy_indptr[i+1]
                D = D_concat[:,D_sidx:D_eidx].tocsc()
                D = torch.FloatTensor(D.toarray()).unsqueeze(0)
                self.all_D.append(D)


class DataLoaderEM:

    def __init__(self, dataset, OL, max_depth=5):
        self.nlink = OL.shape[0]
        self.dummy_incidence_smat = OL[:, self.nlink:]
        all_obs_seq_data = []
        all_obs_seq_len = []
        all_missing_seg = []
        for obs_data, obs_len, miss in dataset:
            all_obs_seq_data.append(obs_data)
            all_obs_seq_len.append(obs_len)
            all_missing_seg.extend(miss)
        self.full_starts = None
        self.max_depth = max_depth
        self.mark_visited = False
        self.use_shortest_path = False
        self.seed = 0

        self.process_obs(all_obs_seq_data, all_obs_seq_len)
        self.process_missing(all_missing_seg)
        self.adjlist = get_adjlist(OL[:, :self.nlink])

    def process_obs(self, all_obs_seq_data, all_obs_seq_len):
        observed_smat = get_csr_from_data_length(
            np.concatenate(all_obs_seq_data).astype(int), 
            np.concatenate(all_obs_seq_len).astype(int)
        )
        self.observed_seq_1based_mat = observed_smat.toarray()
        self.observed_seq_1based_smat = observed_smat
        observed_smat = observed_smat.copy()
        observed_smat.data = observed_smat.data - 1
        self.network_dummies = np.array(range(self.dummy_incidence_smat.shape[1])) + self.nlink
        self.n_observed_obs = observed_smat.shape[0]

    def process_missing(self, all_missing_seg):
        if all_missing_seg is None or len(all_missing_seg) == 0:
            self.n_missing_segs = 0
            self.full_starts, self.full_ends, self.full_links,\
                self.full_successors_no_dummy, self.full_dummies, self.full_indptr\
                = get_links_from_observed_1based_data(self.observed_seq_1based_smat, self.nlink)
            self.observed_link_idx = 0
            self.observed_path_idx = 0
            self.full_path_probs = np.ones(self.n_observed_obs)
            self.link_probs = np.ones(len(self.full_links))
        else:
            missing_mat = np.stack(all_missing_seg).astype(int)
            self.missing_seg_1based_mat = missing_mat
            self.n_missing_segs = missing_mat.shape[0]
            self.missing_dummies = missing_mat[:,0] - 1
    
    def resample_path_for_missing_segments(self):
        self.is_able_to_connect_missing_segs = np.ones(self.n_missing_segs).astype(int)
        sample_1based_paths, self.paths_1based_smat, sample_path_count, sample_path_lens = self.generate_samples_for_missing_data(self.missing_seg_1based_mat)
        self.n_samples = sample_1based_paths.shape[0]
        self.sample_path_lens = sample_path_lens
        self.generate_full_observations(sample_1based_paths)
        self.n_full_obs = self.full_obs_1based_smat.shape[0]
        self.cumulative_sample_count = np.zeros(len(sample_path_count)+1).astype(int)
        self.cumulative_sample_count[1:] = np.cumsum(sample_path_count)
        self.paths_mat = self.paths_1based_smat.copy()
        self.paths_mat.data = self.paths_mat.data + 1
        self.paths_mat = self.paths_mat.toarray().astype(int)
        self.paths_mat[self.paths_mat == 0] = self.nlink + 2
        self.paths_mat -= 2
        self.paths_mat[self.paths_mat > self.nlink] = self.nlink
        self.full_starts, self.full_ends, self.full_links, self.full_successors_no_dummy, self.full_dummies, self.full_indptr \
            = get_links_from_observed_1based_data(self.full_obs_1based_smat, self.nlink)
    
    def generate_full_observations(self, sample_1based_paths):
        if len(sample_1based_paths) == 0:
            self.full_obs_smat = self.observed_seq_1based_smat
        else:
            self.full_obs_1based_smat = sparse.csr_matrix(padding_zeros_list_of_2d_nparrays([sample_1based_paths, self.observed_seq_1based_mat], n_padding_col=0))
        return self.full_obs_1based_smat

    def generate_samples_for_missing_data(self, missing_seg_1based_mat):
        sample_1based_paths = []
        n_sample_paths = []
        path_lens = []
        all_shortest_path_only = []
        with Pool(24) as p:
            func = partial(generate_samples_for_missing_data_subset, adjlist=self.adjlist, max_depth=self.max_depth, mark_visited=self.mark_visited, seed=self.seed)
            self.seed += 1
            for path_arr, shortest_path_only, path_len in p.map(func, missing_seg_1based_mat):
                sample_1based_paths.append(path_arr)
                n_sample_paths.append(len(path_arr))
                path_lens.append(path_len)
                all_shortest_path_only.append(not shortest_path_only)
        path_lens = np.concatenate(path_lens)
        if not self.use_shortest_path:
            self.is_able_to_connect_missing_segs = np.array(all_shortest_path_only)
        sample_1based_paths = padding_zeros_list_of_2d_nparrays(sample_1based_paths, n_padding_col=0).astype(int)
        sample_paths_1based_smat = sparse.csr_matrix(sample_1based_paths)
        return sample_1based_paths, sample_paths_1based_smat, n_sample_paths, path_lens

    def reload_path_probs(self, Ps):
        all_sample_missing_paths_probs = []
        path_prob_args = []
        for i in tqdm(range(len(self.cumulative_sample_count)-1), leave=False, desc='reload_path_probs'):
            s = self.cumulative_sample_count[i]
            e = self.cumulative_sample_count[i+1]
            dummy = self.paths_1based_smat[s,0] - 1
            policy:torch.FloatTensor = Ps[dummy-self.nlink]
            indices = policy._indices().detach().cpu().numpy()
            data = policy._values().detach().cpu().numpy()
            indices = [indices[2], indices[1]]
            shape = [policy.shape[2], policy.shape[1]]
            policy = sparse.csr_matrix((data, indices), shape=shape)
            path_prob_args.append((
                self.paths_mat[s:e,:],
                policy,
                self.is_able_to_connect_missing_segs[i],
                self.sample_path_lens[s:e]))
        
        with Pool(24) as p:
            for path_probs in p.starmap(compute_probs_from_seq_1based_smat, path_prob_args):
                all_sample_missing_paths_probs.append(path_probs)

        all_sample_missing_paths_probs = np.concatenate(all_sample_missing_paths_probs)
        self.full_path_probs = np.ones(self.n_full_obs)
        self.full_path_probs[:self.n_samples] = all_sample_missing_paths_probs
        self.link_probs = np.ones(len(self.full_links))
        self.observed_link_idx = self.full_indptr[self.n_samples]
        self.observed_path_idx = self.n_samples
        for i in range(1, len(self.full_indptr)):
            self.link_probs[self.full_indptr[i-1]:self.full_indptr[i]] = self.full_path_probs[i-1]
        
    
class DataLoaderMaxEnt(DataLoaderEM):

    def __init__(self, dataset, OL):
        self.nlink = OL.shape[0]
        self.dummy_incidence_smat = OL[:, self.nlink:]
        all_obs_seq_data = []
        all_obs_seq_len = []
        all_missing_seg = []
        for obs_data, obs_len, miss in dataset:
            all_obs_seq_data.append(obs_data)
            all_obs_seq_len.append(obs_len)
            all_missing_seg.extend(miss)
        self.process_obs(all_obs_seq_data, all_obs_seq_len)
        self.process_missing(None)
        self.adjlist = get_adjlist(OL[:, :self.nlink])
        