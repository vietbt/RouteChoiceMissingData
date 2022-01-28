from functools import partial
import os
import torch
from lbfgs import LBFGS
from model import RouteChoice, RouteChoiceEM, RouteChoiceMaxEnt
from data import DataLoader, DataLoaderEM, DataLoaderMaxEnt
from utils import generate_data, get_log_path, load_data
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
import time

try:
    from apex import amp
except:
    amp = None


class Trainer:

    METHODS = {
        'composition': (RouteChoice, DataLoader),
        'em5': (RouteChoiceEM, partial(DataLoaderEM, max_depth=5)),
        'em8': (RouteChoiceEM, partial(DataLoaderEM, max_depth=8)),
        'maxent': (RouteChoiceMaxEnt, DataLoaderMaxEnt),
    }

    def __init__(self, init_beta, init_scale, missing_prob, method, use_LS, use_LS_for_beta, seed=0, consecutive=True, train_mu=True, data_folder="dataset", device='cuda', logdir="logs", evaluate=False):
        if not evaluate:
            print("Setting up ...")
            pprint(locals(), indent=4)
        obs, self.OL, LT, UT, TT, LS = load_data(data_folder)
        dataset = generate_data(obs, missing_prob, consecutive, seed)
        all_dataset = generate_data(obs, 0.0, consecutive, seed)
        self.beta = init_beta
        self.scale = init_scale
        self.device = device
        self.best_score = None
        self.step = 0
        
        self.logdir = get_log_path(logdir, method, seed, missing_prob, train_mu, use_LS, use_LS_for_beta)
        if not evaluate:
            self.writer = SummaryWriter(self.logdir)

        Model, Data = self.METHODS[method]
        
        self.model = Model(self.beta, self.scale, self.OL, LT, UT, TT, LS, use_LS_for_beta, seed=seed)
        self.model.to(device)
        self.train_mu = train_mu
        self.use_LS = use_LS
        self.use_LS_for_beta = use_LS_for_beta

        if not evaluate:
            self.lbfgs = LBFGS(self.model.parameters(), lr=0.1)
            self.lbfgs2 = LBFGS(self.model.parameters(), lr=0.1)
            self.lbfgs3 = LBFGS(self.model.parameters(), lr=1)
            self.lbfgs4 = LBFGS(self.model.parameters(), lr=1, max_iter=50)
            if amp is not None and device != 'cpu':
                self.model, [self.lbfgs, self.lbfgs2, self.lbfgs3, self.lbfgs4] \
                    = amp.initialize(self.model, [self.lbfgs, self.lbfgs2, self.lbfgs3, self.lbfgs4], opt_level='O1', verbosity=0)
        elif amp is not None and device != 'cpu':
            self.model = amp.initialize(self.model, opt_level='O1', verbosity=0)

        self.data = Data(dataset, self.OL)
        self.all_data = DataLoader(all_dataset, self.OL)
        self.all_dataset = all_dataset

    def save_model(self, file_name="model.pt"):
        path = os.path.join(self.logdir, file_name)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, file_name="model.pt"):
        path = os.path.join(self.logdir, file_name)
        self.model.load_state_dict(torch.load(path))
    
    def train(self, train_mu=None, use_LS=None, use_LS_for_beta=None, use_missing=False, max_steps=1, max_early_stop_steps=8, optim='lbfgs'):
        print("Training ...")
        pprint(locals(), indent=4)
        self.writer.add_text('log', str(locals()), self.step)
        self.writer.add_text('nll', str(locals()), self.step)
        # start_step = self.step if optim=='lbfgs3' else 0
        
        self.best_score = None
        self.early_stop_steps = 0

        if not use_missing:
            self.model.init_data(self.data)
        
        optimizer = getattr(self, optim)
        
        if use_LS is None:
            use_LS = self.use_LS
        if use_LS_for_beta is None:
            use_LS_for_beta = self.use_LS_for_beta
        if train_mu is None:
            train_mu = self.train_mu

        self.model.use_LS = use_LS
        self.model.use_LS_for_beta = use_LS_for_beta
        self.model.train_mu = train_mu
        
        for _ in range(max_steps):
            def closure():
                optimizer.zero_grad()
                self.step += 1
                
                outputs = self.model(use_missing=use_missing)
                loss = outputs['loss']

                assert torch.isfinite(loss) and loss > 0
                if amp is not None and self.device != 'cpu':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    scaled_loss = loss

                
                beta = self.model.beta.weight.detach().cpu().numpy()[0]
                scale = self.model.scale.weight.detach().cpu().numpy()[0]
                lr = optimizer.param_groups[0]["lr"]
                beta_str = f"{self.step}: LL = {-loss.item():.6f} @ beta = {beta} & scale = {scale} & lr = {lr:.6f}"
                nll = self.evaluate()
                # title = "_miss" if use_missing else "_no_miss"
                # title += "_ls" if self.use_LS else "_no_ls"
                # title += "_" + optim

                nll_str = f"Step {self.step}: loss = {-loss.item():6f} & NLL = {nll:.6f}/{self.best_score:.6f} & lr = {lr:.6f}"
                print(f"{nll_str} & stop = {self.early_stop_steps}/{max_early_stop_steps}")
                if self.early_stop_steps >= max_early_stop_steps:
                    return

                self.writer.add_scalar(f'settings/use_missing', use_missing, self.step)
                self.writer.add_scalar(f'settings/use_LS', use_LS, self.step)
                self.writer.add_scalar(f'settings/train_mu', train_mu, self.step)
                self.writer.add_scalar(f'likelihood/nll', nll, self.step)
                self.writer.add_scalar(f'likelihood/best_nll', self.best_score, self.step)
                for name, value in outputs.items():
                    self.writer.add_scalar(f'likelihood/{name}', value, self.step)
                self.writer.add_scalar(f'learning_rate', lr, self.step)
                self.writer.add_scalar(f'beta/OL', beta[0], self.step)
                self.writer.add_scalar(f'beta/LT', beta[1], self.step)
                self.writer.add_scalar(f'beta/UT', beta[2], self.step)
                self.writer.add_scalar(f'beta/TT', beta[3], self.step)
                if self.use_LS_for_beta:
                    self.writer.add_scalar(f'beta/LS', beta[4], self.step)
                self.writer.add_scalar(f'scale/OL', scale[0], self.step)
                self.writer.add_scalar(f'scale/TT', scale[1], self.step)
                self.writer.add_scalar(f'scale/LS', scale[2], self.step)
                self.writer.add_text('log', beta_str, self.step)
                self.writer.add_text('nll', nll_str, self.step)
                return scaled_loss
            try:
                if not optim.startswith('lbfgs'):
                    closure()
                    optimizer.step()
                    if self.early_stop_steps >= max_early_stop_steps:
                        break
                else:
                    optimizer.step(closure)
                    break
            except:
                # raise
                time.sleep(0.1)
        torch.cuda.empty_cache()
        self.load_model()

    def evaluate(self):
        loss = self.model.evaluate(self.all_data)
        assert torch.isfinite(loss) and loss > 0
        nll = loss.item()
        if self.best_score is None or self.best_score > nll:
            self.best_score = nll
            self.early_stop_steps = 0
            self.save_model()
        else:
            self.early_stop_steps += 1
        return nll