import os
from tensorboard.backend.event_processing import event_accumulator
import torch
from torch import autograd
from tqdm import tqdm
from data import DataLoader

from trainer import Trainer

import time
from torch.autograd.functional import hessian, jacobian


def read_tensorboard(path, use_missing=True):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    miss_data = ea.Scalars('settings/use_missing')
    miss_data = [miss.value for miss in miss_data]
    miss_pos = miss_data.index(1.0) - 1
    data = ea.Scalars('likelihood/best_nll')
    
    
    if use_missing:
        nll = data[-1].value
        train_time = data[-1].wall_time - data[0].wall_time
        train_step_time = train_time / len(data)
        return nll, train_time, train_step_time
    else:
        prev_nll = data[miss_pos].value
        train_time = data[miss_pos].wall_time - data[0].wall_time
        train_step_time = train_time / miss_pos
        return prev_nll, train_time, train_step_time

def load_model(parts, model_pt, logdir):
    method, seed, prob, mu, LS = parts[1:]
    seed = int(seed.split("_")[-1])
    prob = float(prob.split("_")[-1])
    mu = mu == 'with_mu'
    use_LS_for_beta = LS == 'with_LS_beta'
    LS = use_LS_for_beta or LS == 'with_LS'
    trainer = Trainer(None, None, prob, method, LS, use_LS_for_beta, seed, train_mu=mu, device='cpu', evaluate=True, logdir=logdir)
    trainer.load_model(model_pt)
    trainer.model.use_LS = trainer.use_LS
    trainer.model.train_mu = trainer.train_mu
    return trainer

if __name__=="__main__":
    logdir = 'logs_6'
    use_missing = False
    k = 0
    with open("log_9.txt", 'w') as f:
        for dirpath, _, filenames in os.walk(logdir):
            # if "seed_0" not in dirpath:
            #     continue
            if "composition" not in dirpath:
                continue
            # if "with_LS_beta" not in dirpath:
            #     continue
            
            if len(filenames) > 0:
                print("dirpath:", dirpath)
                event_name = [file for file in filenames if file.startswith("events.out")][0]
                model_pt = [file for file in filenames if file.startswith("model.pt")][0]
                parts = os.path.normpath(dirpath).split(os.path.sep)
                trainer = load_model(parts, model_pt, logdir)
                

                if False:
                    t0 = time.time()
                    trainer.model.init_data(trainer.data)
                    loss = trainer.model(use_missing=True)['loss']
                    execute_time = time.time() - t0
                else:
                    execute_time = 0

                beta = trainer.model.beta.weight
                scale = trainer.model.scale.weight

                if False:
                    n = beta.shape[-1]
                    if trainer.train_mu:
                        n += scale.shape[-1] - 1
                        if trainer.use_LS:
                            n += 1
                    beta_scale = torch.cat((beta, scale), -1)
                    beta_scale = beta_scale[0, :n]

                    J = []
                    for obs in tqdm(trainer.all_dataset, "Computing Jacobian", leave=False):
                        data = DataLoader([obs], trainer.OL)
                        trainer.model.data = data
                        J_obs = jacobian(trainer.model.nll_func_jacobian, beta_scale).cpu().unsqueeze(0)
                        J_obs = J_obs.T @ J_obs
                        J.append(J_obs)
                    N = len(J)
                    J = torch.stack(J)
                    BHHH = torch.mean(J, 0)

                    trainer.model.data = trainer.all_data
                    trainer.model.to('cpu')
                    H = hessian(trainer.model.nll_func_hessian, beta_scale.cpu(), vectorize=True) / N
                    inv_H = torch.inverse(H)
                    cov = (inv_H @ BHHH @ inv_H) / N
                    diag = torch.diag(cov, 0)
                    diag = torch.clamp_min(diag, 0)
                    std_error = torch.sqrt(diag)
                else:
                    std_error = []

                event_path = os.path.join(dirpath, event_name)
                info = list(read_tensorboard(event_path, use_missing=use_missing))
                info.append(execute_time)
                info.extend([x.item() for x in beta[0]])
                info.extend([x.item() for x in scale[0]])
                info.extend([x.item() for x in std_error])
                print("std_error:", std_error)

                if not use_missing:
                    parts[1] += "_no_miss"

                f.write("\t".join(parts + [str(x) for x in info]) + "\n")
                
