from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


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

def plot(init_data, data, pos, ylabel, graph_name, deg=5):
    colors = iter(['#4c72b0', '#dd8453', '#55a868', '#c44f51', '#41e1b9', '#4169e1', '#e1b941', '#e14169', '#b9e141', '#6941e1'])
    markers = iter(['o', 's', 'P', 'X'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#eaeaf2')
    plt.grid(color='w', linestyle=':')

    if init_data is not None:
        init_value = init_data[pos]
        plt.plot((0.1, 0.9), (init_value, init_value), linewidth=3, color=next(colors), label=algos["init"])
    else:
        next(colors)

    for method, parts in data.items():
        if method == "em8":
            continue
        probs = [part[0] for part in parts]
        probs = sorted(set(probs))
        values = defaultdict(list)
        for part in parts:
            if init_data is None:
                values[part[0]].append(part[pos])
            else:
                values[part[0]].append(min(part[pos], init_value))
        values = [values[prob] for prob in probs]
        uppers = [max(value) for value in values]
        lowers = [min(value) for value in values]
        
        values = [np.mean(value) for value in values]

        value_model = np.poly1d(np.polyfit(probs, values, deg))
        upper_model = np.poly1d(np.polyfit(probs, uppers, deg))
        lower_model = np.poly1d(np.polyfit(probs, lowers, deg))

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

        plt.fill_between(probs, lowers, uppers, color=color, alpha=0.15, linewidth=0)
    
    legend = plt.legend(loc='best')
    plt.xlabel('Missing Probability')
    plt.ylabel(ylabel)
    legend.remove()

    plt.savefig(f"graph_{graph_name}.pdf",  bbox_inches="tight")
    export_legend(fig, ax, legend, f"graph_{graph_name}_legend.pdf")
    plt.cla()

def export_legend(fig, ax, legend, legend_figpath):
    fig.canvas.draw()
    legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width*1.2, legend_bbox.height))
    legend_squared = legend_ax.legend(
        *ax.get_legend_handles_labels(), 
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=False,
        fancybox=None,
        shadow=False,
        mode='expand',
    )
    legend_ax.axis('off')
    legend_fig.savefig(
        legend_figpath,
        bbox_inches='tight',
        bbox_extra_artists=[legend_squared],
    )


if __name__ == "__main__":
    algos = {
        'init': 'Composition with No-Missing Dataset',
        'composition': 'Composition with Missing Dataset',
        'composition_no_miss': 'Connected Segments of Missing Dataset',
        'em5': 'EM-BFS',
        'em8': 'EM-BFS',
    }

    init_data = load_log("log_no_missing.txt")
    init_data = list(init_data.values())[0]
    
    data = load_log("log_exec_time.txt")
    plot(None, data, 6, 'Exec. Time of Computing LL (s)', 'exec_time')
    
    data = load_log("log_training_time.txt", ['without_LS'])
    plot(None, data, 4, 'Training Time (s)', 'training_time', 3)

    data = load_log("log_nll.txt", ['without_mu', 'without_LS'])
    data.update(load_log("log_nll_maxent.txt", ['without_mu', 'without_LS']))
    plot(init_data[0], data, 3, 'Log Likelihood', 'nll_without_mu_without_LS')

    data = load_log("log_nll.txt", ['with_mu', 'without_LS'])
    data.update(load_log("log_nll_maxent.txt", ['with_mu', 'without_LS']))
    plot(init_data[1], data, 3, 'Log Likelihood', 'nll_mu_without_ls')

    data = load_log("log_nll.txt", ['with_mu', 'with_LS'])
    data.update(load_log("log_nll_maxent.txt", ['with_mu', 'with_LS']))
    plot(init_data[2], data, 3, 'Log Likelihood', 'nll_mu_ls')

    data = load_log("log_nll_beta.txt", ['with_mu', 'with_LS_beta'])
    data.update(load_log("log_nll_beta_maxent.txt", ['with_mu', 'with_LS_beta']))
    plot(init_data[3], data, 3, 'Log Likelihood', 'nll_mu_ls_beta')
