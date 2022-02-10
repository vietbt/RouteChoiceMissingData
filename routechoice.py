from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DataLoader
from utils import form_M, load_beta_ls, scipy_sparse_to_tensor, form_B, form_sparse_M, get_V, get_Z, get_loglikelihood
import numpy as np
from tqdm import tqdm

class RouteChoiceBase(nn.Module):

    def __init__(self, init_beta, init_scale, OL, LT, UT, TT, LS, use_LS_for_beta=False, regularizer=False, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.beta = nn.Linear(5 if use_LS_for_beta else 4, 1, bias=False)
        self.scale = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            if init_beta is not None:
                if use_LS_for_beta:
                    init_beta = init_beta + [0.0]
                self.beta.weight.copy_(torch.FloatTensor(init_beta))
            if init_scale is not None:
                self.scale.weight.copy_(torch.FloatTensor(init_scale))
        self.nlink = OL.shape[0]
        self.B = form_B(self.nlink, OL[:, self.nlink:])
        OL = OL[:, :self.nlink]
        self.OL_indices = scipy_sparse_to_tensor(OL)._indices()
        if use_LS_for_beta:
            self.beta_LS = load_beta_ls(LS, self.OL_indices, self.nlink)
            self.beta_ls_decoder = nn.Sequential(
                nn.Linear(len(self.beta_LS), 1),
                nn.ReLU()
            )
            
        self.beta_feat_data = torch.FloatTensor(np.stack([f.data for f in [OL, LT, UT, TT]])).T
        self.OL = torch.FloatTensor(np.sum(OL, 0))
        self.TT = torch.FloatTensor(np.sum(TT, 0)/np.sum(TT!=0, 0))
        self.LS = torch.FloatTensor(np.stack([np.sum(ls, 0) for ls in LS])).T
        self.ls_decoder = nn.Sequential(
            nn.Linear(len(LS), len(LS)//2),
            nn.ReLU(),
            nn.Linear(len(LS)//2, 1),
            nn.ReLU()
        )
        self.I = torch.eye(self.nlink+1).unsqueeze(0).to_sparse()
        self.regularizer = regularizer
        self.regularizer_mean = -3.0
        self.regularizer_std = 1.0
        self.use_LS = False
        self.use_LS_for_beta = use_LS_for_beta
        self._use_LS_for_beta = use_LS_for_beta
        self.train_mu = True
    
    def to(self, device):
        self.beta_feat_data = self.beta_feat_data.to(device)
        self.OL = self.OL.to(device)
        self.TT = self.TT.to(device)
        self.LS = self.LS.to(device)
        self.OL_indices_cuda = self.OL_indices[1:].to(device)
        self.device = self.OL.device
        if self._use_LS_for_beta:
            try:
                self.beta_LS = self.beta_LS.to(device)
            except:
                pass
        super().to(device)

    def forward_obs(self):
        torch.set_grad_enabled(self.train_mu)
        if self.use_LS:
            LS = self.ls_decoder(self.LS).T
        else:
            LS = torch.zeros_like(self.TT)
        mu_feat_data = torch.stack([self.OL, self.TT, LS]).squeeze(1)
        mu_feat_data = F.pad(mu_feat_data, [0, 1], value=1).T
        mu = torch.exp(self.scale(mu_feat_data).squeeze(-1))
        torch.set_grad_enabled(True)
        if self._use_LS_for_beta:
            if self.use_LS_for_beta:
                LS = self.beta_ls_decoder(self.beta_LS.T)
            else:
                LS = torch.zeros((self.OL_indices[1].shape[0], 1), device=self.device)
            beta_feat_data = torch.cat((self.beta_feat_data, LS), -1)
        else:
            beta_feat_data = self.beta_feat_data
        ireward_data = self.beta(beta_feat_data).squeeze(-1)
        M, M_T = form_sparse_M(self.OL_indices, ireward_data, mu, self.nlink)
        Z = get_Z(M, self.B, self.I)
        mu_cpu = mu.cpu().type(Z.dtype)
        V = get_V(Z, mu_cpu)
        ireward_smat_no_dummy = torch.sparse_coo_tensor(self.OL_indices_cuda, ireward_data, device=self.device).to_dense()
        return ireward_smat_no_dummy, mu, mu_cpu, V, M_T, Z.T
    
    def nll_func_jacobian(self, beta_scale):
        pos = 5 if self.use_LS_for_beta else 4
        beta = beta_scale[None,:pos]
        scale = beta_scale[None, pos:]
        scale = F.pad(scale, [0, 3-scale.shape[1]])
        with torch.no_grad():
            beta_feat_data = self.beta_feat_data
            LS = torch.zeros_like(self.TT)
            if self.use_LS:
                LS = self.ls_decoder(self.LS).T
                if self.use_LS_for_beta:
                    LS_beta = self.beta_ls_decoder(self.beta_LS.T)
                    beta_feat_data = torch.cat((self.beta_feat_data, LS_beta), -1)
            mu_feat_data = torch.stack([self.OL, self.TT, LS]).squeeze(1)
            mu_feat_data = F.pad(mu_feat_data, [0, 1], value=1).T
        mu = torch.exp(mu_feat_data @ scale.T)
        ireward_data = beta_feat_data @ beta.T
        mu = mu.squeeze(-1)
        ireward_data = ireward_data.squeeze(-1)
        M, M_T = form_sparse_M(self.OL_indices, ireward_data, mu, self.nlink)
        Z = get_Z(M, self.B, self.I)
        mu_cpu = mu.cpu().type(Z.dtype)
        V = get_V(Z, mu_cpu)
        ireward_smat = torch.sparse_coo_tensor(self.OL_indices_cuda, ireward_data, device=self.device).to_dense()
        obs_nll = - get_loglikelihood(
            self.nlink, V, ireward_smat, mu, mu_cpu,
            self.data.obs_starts, self.data.obs_ends, self.data.obs_links,
            self.data.obs_successors_no_dummy, self.data.obs_dummies)
        return obs_nll

    def nll_func_hessian(self, beta_scale):
        pos = 5 if self.use_LS_for_beta else 4
        beta = beta_scale[None,:pos]
        scale = beta_scale[None, pos:]
        scale = F.pad(scale, [0, 3-scale.shape[1]])
        with torch.no_grad():
            beta_feat_data = self.beta_feat_data
            LS = torch.zeros_like(self.TT)
            if self.use_LS:
                LS = self.ls_decoder(self.LS).T
                if self.use_LS_for_beta:
                    LS_beta = self.beta_ls_decoder(self.beta_LS.T)
                    beta_feat_data = torch.cat((self.beta_feat_data, LS_beta), -1)
            mu_feat_data = torch.stack([self.OL, self.TT, LS]).squeeze(1)
            mu_feat_data = F.pad(mu_feat_data, [0, 1], value=1).T
        mu = torch.exp(mu_feat_data @ scale.T)
        ireward_data = beta_feat_data @ beta.T
        mu = mu.squeeze(-1)
        ireward_data = ireward_data.squeeze(-1)
        M = form_M(self.OL_indices, ireward_data, mu, self.nlink)
        Z = get_Z(M, self.B, self.I)
        V = get_V(Z, mu)
        ireward_smat = torch.zeros((self.nlink, self.nlink), device=self.device, dtype=ireward_data.dtype)
        ids = self.OL_indices_cuda
        ireward_smat[ids[0], ids[1]] = ireward_data
        obs_nll = - get_loglikelihood(
            self.nlink, V, ireward_smat, mu, mu,
            self.data.obs_starts, self.data.obs_ends, self.data.obs_links,
            self.data.obs_successors_no_dummy, self.data.obs_dummies)
        return obs_nll
    
    def forward_reg(self):
        reg = torch.sum((self.beta - self.regularizer_mean) * (self.beta - self.regularizer_mean) ) / self.regularizer_std
        return reg
    
    def evaluate(self, data: DataLoader):
        with torch.no_grad():
            ireward_smat_no_dummy, mu, mu_cpu, V, _, _ = self.forward_obs()
            obs_nll = - get_loglikelihood(
                self.nlink, V, ireward_smat_no_dummy, mu, mu_cpu,
                data.obs_starts, data.obs_ends, data.obs_links,
                data.obs_successors_no_dummy, data.obs_dummies)
            return obs_nll