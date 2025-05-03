import json
import matplotlib.pyplot as plt
import os

def plot_steps_vs_nodes(data, dataset_name, save_path, accuracy_threshold=0.75):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"Steps Required to Reach {accuracy_threshold*100}% Accuracy - {dataset_name}")
    ax.set_xlabel("Node Count")
    ax.set_ylabel("Steps to Accuracy")
    x = [2, 4, 6]
    ax.set_xticks(x)
    ax.set_xticklabels(["2", "4", "6"])
    ax.plot(x, data["0"], label="0%", marker='o', color='blue')
    ax.plot(x, data["0.001"], label="0.1%", marker='o', color='orange')
    ax.plot(x, data["0.005"], label="0.5%", marker='o', color='green')
    ax.plot(x, data["0.01"], label="1.0%", marker='o', color='red')
    ax.legend(title="Packet Loss Rate")
    ax.grid()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_steps_vs_loss_rate(data, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"Steps vs Loss Rate - {dataset_name}")
    ax.set_xlabel("Packet Loss Rate (%)")
    ax.set_ylabel("Steps")
    x = [0.0, 0.1, 0.5, 1.0]
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    
    # For each node count (2, 4, 6 nodes - indices 0, 1, 2)
    for node_idx, node_count in enumerate([2, 4, 6]):
        steps = []
        for loss_rate in ["0", "0.001", "0.005", "0.01"]:
            steps.append(data[loss_rate][node_idx])
        ax.plot(x, steps, label=f"{node_count} nodes", marker='o')
    
    ax.legend(title="Node Count")
    ax.grid()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Load results
    with open("results/results.json", "r") as f:
        results = json.load(f)
    
    # Create plots for each dataset
    for dataset_name in results.keys():
        # Create both types of plots for each dataset
        plot_steps_vs_nodes(
            results[dataset_name],
            dataset_name,
            f"plots/{dataset_name}_steps_vs_nodes.png"
        )
        
        plot_steps_vs_loss_rate(
            results[dataset_name],
            dataset_name,
            f"plots/{dataset_name}_steps_vs_loss_rate.png"
        )

if __name__ == "__main__":
    main() 