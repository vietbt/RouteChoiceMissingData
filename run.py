import os
from tqdm import tqdm

if __name__ == "__main__":
    all_cmd = []
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for train_mu in [True, False]:
            for use_LS in [True, False]:
                for use_LS_for_beta in [True, False]:
                    for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        for method in ['composition', 'em5', 'em8']:
                            cmd = f"python main.py --seed {seed} --prob {prob} --method {method} --logdir logs"
                            if train_mu:
                                cmd = f"{cmd} --train_mu"
                                if use_LS:
                                    cmd = f"{cmd} --use_LS"
                                    if use_LS_for_beta:
                                        cmd = f"{cmd} --use_LS_for_beta"
                            if prob == 0.0 and method in ['em5', 'em8']:
                                continue
                            if cmd in all_cmd:
                                continue
                            all_cmd.append(cmd)
    for cmd in tqdm(all_cmd):
        print(cmd)
        os.system(cmd)
    