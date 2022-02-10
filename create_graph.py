from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tsmoothie.smoother import LowessSmoother

def read_str(s, mode=int):
    try:
        return mode(s)
    except:
        if mode == int:
            return read_str(s, float)
        else:
            return s

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size'   : 22
})

def load_log(file_name, keys=None):
    data = defaultdict(list)
    with open(file_name, "r") as f:
        for line in f:
            parts = line.split()
            if keys is not None:
                if any(key not in line for key in keys):
                    continue
            parts[3] = parts[3][-3:]
            parts = [read_str(part) for part in parts]
            parts[6] = - parts[6]
            data[parts[1]].append(parts[3:])
    return data

def plot(data, pos, ylabel, graph_name, deg=5):
    colors = iter(['#4c72b0', '#dd8453', '#55a868', '#c44f51', '#41e1b9', '#4169e1', '#e1b941', '#e14169', '#b9e141', '#6941e1'])
    markers = iter(['o', 's', 'P', 'X'])
    plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.set_facecolor('#eaeaf2')
    plt.grid(color='w', linestyle=':')
    for method, parts in data.items():
        probs = [part[0] for part in parts]
        probs = sorted(set(probs))
        values = defaultdict(list)
        for part in parts:
            values[part[0]].append(part[pos])
        values = [values[prob] for prob in probs]
        uppers = [max(value) for value in values]
        lowers = [min(value) for value in values]
        
        values = [np.mean(value) for value in values]

        value_model = np.poly1d(np.polyfit(probs, values, deg))
        upper_model = np.poly1d(np.polyfit(probs, uppers, deg))
        lower_model = np.poly1d(np.polyfit(probs, lowers, deg))

        # values = LowessSmoother(0.4, 1).smooth(values).smooth_data[0]
        values = value_model(probs)

        color = next(colors)
        marker = next(markers)
        method = algos[method]
        plt.scatter(probs, values, color=color, marker=marker)
        probs = np.linspace(min(probs), max(probs), 100)
        values = value_model(probs)
        plt.plot(probs, values, color=color, linewidth=3)
        plt.plot([], [], color=color, linewidth=3, label=method, marker=marker)

        uppers = upper_model(probs)
        lowers = lower_model(probs)

        # print("probs:", probs)
        # print("uppers:", uppers)

        # uppers = LowessSmoother(0.4, 1).smooth(uppers).smooth_data[0]
        # lowers = LowessSmoother(0.4, 1).smooth(lowers).smooth_data[0]
        
        plt.fill_between(probs, lowers, uppers, color=color, alpha=0.15, linewidth=0)
    plt.legend(loc='best')
    plt.xlabel('Missing Probability')
    plt.title(ylabel, pad=20)
    plt.savefig(f"{graph_name}.pdf",  bbox_inches="tight")
    plt.cla()

if __name__ == "__main__":
    algos = {
        'composition': 'Composition',
        'em5': 'EM-BFS-5',
        'em8': 'EM-BFS-8',
    }

    data = load_log("log.txt")
    plot(data, 6, 'Execution Time of Computing Log Likelihood (s)', 'graph_exec_time')
    
    data = load_log("log_2.txt", ['without_mu', 'without_LS'])
    plot(data, 4, 'Training Time (s)', 'graph_training_time', 3)

    data = load_log("log_4.txt", ['with_mu', 'with_LS'])
    plot(data, 3, 'Log Likelihood (mu + LS)', 'graph_nll_mu_ls')

    data = load_log("log_4.txt", ['with_mu', 'without_LS'])
    plot(data, 3, 'Log Likelihood (mu + w/o LS)', 'graph_nll_mu_without_ls')

    data = load_log("log_4.txt", ['without_mu', 'without_LS'])
    plot(data, 3, 'Log Likelihood (w/o mu + w/o LS)', 'graph_nll_without_mu_without_LS')

    data = load_log("log_5.txt", ['with_LS_beta'])
    plot(data, 3, 'Log Likelihood (mu + LS beta)', 'graph_nll_mu_ls_beta')
