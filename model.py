import torch
from data import DataLoader, DataLoaderEM, DataLoaderMaxEnt
from routechoice import RouteChoiceBase
from utils import compute_all_P, compute_all_Q, get_loglikelihood


class RouteChoice(RouteChoiceBase):

    def __init__(self, init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta=False, regularizer=False, seed=None):
        super().__init__(init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta, regularizer, seed)
    
    def forward_missing(self, M_T, Z_T, data: DataLoader):
        neglog_likelihood = 0
        self.all_Q = compute_all_Q(M_T, Z_T)
        missing_P_transposes = compute_all_P(self.I, self.all_Q, data.all_D)
        for i in range(len(data.network_dummies)):
            D_sidx = data.missing_dummy_indptr[i]
            D_eidx = data.missing_dummy_indptr[i+1]
            missing_P_transpose = missing_P_transposes[i]
            if missing_P_transpose is None:
                continue
            missing_P_transpose = torch.reshape(missing_P_transpose, (self.nlink+1, -1))
            missing_seg_ends = data.missing_ends[D_sidx:D_eidx]
            dest_idxs = list(range(D_eidx - D_sidx))
            probs = missing_P_transpose[missing_seg_ends, dest_idxs]
            probs[probs<=0] = 0
            probs[probs>1] = 0
            probs = probs[probs>0]
            if len(probs) == 0:
                continue
            probs = torch.log(probs)
            NLL = -torch.sum(probs)
            neglog_likelihood += NLL
        return neglog_likelihood

    def init_data(self, data: DataLoader):
        self.data = data

    def forward(self, use_missing=False):
        results = {} 
        if self.data.n_missing_segs == 0:
            use_missing = False
        ireward_smat_no_dummy, mu, mu_cpu, V, M_T, Z_T = self.forward_obs()
        obs_loss = - get_loglikelihood(
            self.nlink, V, ireward_smat_no_dummy, mu, mu_cpu,
            self.data.obs_starts, self.data.obs_ends, self.data.obs_links,
            self.data.obs_successors_no_dummy, self.data.obs_dummies)
        loss = obs_loss
        scale_factor = self.data.n_observed_obs
        results.update(dict(obs_loss=obs_loss))
        if use_missing:
            missing_loss = self.forward_missing(M_T, Z_T, self.data)
            loss += missing_loss
            scale_factor += self.data.n_observed_obs
            results.update(dict(missing_loss=missing_loss))
        if self.regularizer:
            reg_loss = self.forward_reg()
            loss += reg_loss
            results.update(dict(reg_loss=reg_loss))
        loss /= scale_factor
        loss = loss.to(self.device)
        results.update(dict(loss=loss))
        return results
    

class RouteChoiceEM(RouteChoiceBase):

    def __init__(self, init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta=False, regularizer=False, seed=None):
        super().__init__(init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta, regularizer, seed)
        
    def init_data(self, data: DataLoaderEM):
        data.seed = self.seed
        _, _, _, _, M_T, Z_T = self.forward_obs()
        all_Q = compute_all_Q(M_T, Z_T)
        data.resample_path_for_missing_segments()
        data.reload_path_probs(all_Q)
        self.data = data

    def forward(self, use_missing=False):
        results = {}
        if self.data.n_missing_segs == 0:
            use_missing = False
        ireward_smat_no_dummy, mu, mu_cpu, V, M_T, Z_T = self.forward_obs()
        
        if use_missing:
            all_Q = compute_all_Q(M_T, Z_T)
            self.data.resample_path_for_missing_segments()
            self.data.reload_path_probs(all_Q)

        if use_missing:
            start_path_idx = self.data.observed_path_idx
            start_link_idx = self.data.observed_link_idx
        else:
            start_path_idx = 0
            start_link_idx = 0

        obs_loss = - get_loglikelihood(
            self.nlink, V, ireward_smat_no_dummy, mu, mu_cpu,
            self.data.full_starts[start_path_idx:],
            self.data.full_ends[start_path_idx:],
            self.data.full_links[start_link_idx:],
            self.data.full_successors_no_dummy[start_link_idx:],
            self.data.full_dummies[start_path_idx:],
            self.data.full_path_probs[start_path_idx:],
            self.data.link_probs[start_link_idx:])
        loss = obs_loss
        scale_factor = self.data.n_observed_obs
        results.update(dict(obs_loss=obs_loss))

        if self.regularizer:
            reg_loss = self.forward_reg()
            loss += reg_loss
            results.update(dict(reg_loss=reg_loss))
        
        if use_missing:
            scale_factor += self.data.n_missing_segs
        loss /= scale_factor
        loss = loss.to(self.device)
        results.update(dict(loss=loss))
        return results
    

class RouteChoiceMaxEnt(RouteChoiceBase):

    def __init__(self, init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta=False, regularizer=False, seed=None):
        super().__init__(init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta, regularizer, seed)
        
    def init_data(self, data: DataLoaderMaxEnt):
        self.data = data

    def forward(self, use_missing=False):
        results = {}
        if use_missing:
            return 
        ireward_smat_no_dummy, mu, mu_cpu, V, _, _ = self.forward_obs()

        obs_loss = - get_loglikelihood(
            self.nlink, V, ireward_smat_no_dummy, mu, mu_cpu,
            self.data.full_starts,
            self.data.full_ends,
            self.data.full_links,
            self.data.full_successors_no_dummy,
            self.data.full_dummies)
        
        loss = obs_loss
        scale_factor = self.data.n_observed_obs
        results.update(dict(obs_loss=obs_loss))

        if self.regularizer:
            reg_loss = self.forward_reg()
            loss += reg_loss
            results.update(dict(reg_loss=reg_loss))

        loss /= scale_factor
        loss = loss.to(self.device)
        results.update(dict(loss=loss))
        return results
    