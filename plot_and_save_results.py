import json
import matplotlib.pyplot as plt
import os

def plot_steps_vs_nodes(data, dataset_name, save_path, accuracy_threshold=0.75):
    plt.rcParams.update({'font.size': 14})  # Set base font size
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"Steps Required to Reach {accuracy_threshold*100}% Accuracy - {dataset_name}", fontsize=14)
    ax.set_xlabel("Node Count", fontsize=14)
    ax.set_ylabel("Steps to Accuracy", fontsize=14)
    x = [2, 4, 6]
    ax.set_xticks(x)
    ax.set_xticklabels(["2", "4", "6"], fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.plot(x, data["0"], label="0%", marker='o', color='blue')
    ax.plot(x, data["0.001"], label="0.1%", marker='o', color='orange')
    ax.plot(x, data["0.005"], label="0.5%", marker='o', color='green')
    ax.plot(x, data["0.01"], label="1.0%", marker='o', color='red')
    ax.legend(title="Packet Loss Rate", title_fontsize=15, fontsize=15)
    ax.grid()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_steps_vs_loss_rate(data, dataset_name, save_path):
    plt.rcParams.update({'font.size': 14})  # Set base font size
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"Steps to Accuracy When Trained on 4 nodes- {dataset_name}", fontsize=14)
    ax.set_xlabel("Packet Loss Rate (%)", fontsize=14)
    ax.set_ylabel("Steps", fontsize=14)
    x = [0.0, 0.1, 0.5, 1.0]
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    accuracies = list(data.keys())
    for i in range(len(accuracies)):
        acc = accuracies[i]
        ax.plot(x, data[acc], label=acc, marker='o')
    ax.legend(title="Accuracy", title_fontsize=15, fontsize=15)
    ax.grid()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
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
        plot_steps_vs_nodes(
            tta_node_count[dataset_name],
            dataset_name,
            f"plots/{dataset_name}_steps_vs_nodes.png"
        )
        
    for dataset_name in tta_loss_rate.keys():
        plot_steps_vs_loss_rate(
            tta_loss_rate[dataset_name],
            dataset_name,
            f"plots/{dataset_name}_steps_vs_loss_rate.png"
        )

if __name__ == "__main__":
    main() 