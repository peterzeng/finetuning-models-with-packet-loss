import json
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os

colors = ['#9edfd8','#81bab3','#357d8d','#072635', 'black']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_steps_vs_nodes(data, dataset_name, save_path, accuracy_threshold=0.75):
    # plt.rcParams.update({'font.size': 16})  # Set base font size
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"Steps Required to Reach {accuracy_threshold*100}% Accuracy - {dataset_name}", fontsize=18, pad=15)
    ax.set_xlabel("Node Count", fontsize=16)
    ax.set_ylabel("Steps to Accuracy", fontsize=16)
    x = [2, 4, 6]
    ax.set_xticks(x)
    ax.set_xticklabels(["2", "4", "6"], fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.plot(x, data["0"], label="0%", marker='o', color=colors[0])
    ax.plot(x, data["0.001"], label="0.1%", marker='o', color=colors[1])
    ax.plot(x, data["0.005"], label="0.5%", marker='o', color=colors[2])
    ax.plot(x, data["0.01"], label="1.0%", marker='o', color=colors[3])
    ax.legend(title="Packet Loss Rate", title_fontsize=16, fontsize=16)
    ax.grid()
    plt.savefig(save_path)
    plt.close()

def plot_steps_vs_loss_rate(data, dataset_name, save_path):
    # plt.rcParams.update({'font.size': 16})  # Set base font size
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"Steps to Accuracy When Trained on 4 nodes- {dataset_name}", fontsize=18, pad=15)
    ax.set_xlabel("Packet Loss Rate (%)", fontsize=16)
    ax.set_ylabel("Steps", fontsize=16)
    x = [0.0, 0.1, 0.5, 1.0]
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    accuracies = list(data.keys())
    for i in range(len(accuracies)):
        acc = accuracies[i]
        ax.plot(x, data[acc], label=acc, marker='o', color=colors[i])
    ax.legend(title="Accuracy", title_fontsize=16, fontsize=16)
    ax.grid()

    plt.savefig(save_path)
    plt.close()

def main():

    target_accuracy_map = {
        'winogrande': 0.8,
        'piqa': 0.7,
        'hellaswag': 0.7,
        'mnli': 0.75
    }
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Load results
    with open("results/tta-node_count.json", "r") as f:
        tta_node_count = json.load(f)
    
    with open("results/tta-loss_rate.json", "r") as f:
        tta_loss_rate = json.load(f)
    # Create plots for each dataset

    for dataset_name in tta_node_count.keys():
        # Create both types of plots for each dataset
        pure_ds_name = dataset_name.split('-')[0]

        plot_steps_vs_nodes(
            tta_node_count[dataset_name],
            dataset_name,
            f"plots/{dataset_name}_steps_vs_nodes.png",
            accuracy_threshold=target_accuracy_map[pure_ds_name]
        )
        
    for dataset_name in tta_loss_rate.keys():
        plot_steps_vs_loss_rate(
            tta_loss_rate[dataset_name],
            dataset_name,
            f"plots/{dataset_name}_steps_vs_loss_rate.png"
        )

if __name__ == "__main__":
    main() 