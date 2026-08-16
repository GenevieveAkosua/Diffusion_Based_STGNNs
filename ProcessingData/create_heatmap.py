import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_adjacency(file_path="adj.npy", output_file="adjacency_heatmap.png"):
    print(f"Loading adjacency matrix from {file_path}...")
    
    try:
        # Load the adjacency matrix
        adj = np.load(file_path)
        num_nodes = adj.shape[0]
        
        print(f"Matrix shape: {adj.shape}")
        
        # Calculate the number of neighbors per node (node degree)
        # Since self-loops were set to 0.0, any value > 0 is a valid edge
        node_degrees = np.sum(adj > 0, axis=1)
        
        # Initialize a figure with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        
        # ---------------------------------------------------------
        # Plot 1: The Adjacency Heatmap
        # ---------------------------------------------------------
        sns.heatmap(
            adj, 
            cmap="viridis", 
            ax=ax1, 
            square=True,
            cbar_kws={'shrink': 0.8, 'label': 'Edge Weight (Gaussian Similarity)'}
        )
        ax1.set_title("Adjacency Matrix Heatmap", fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel("Target Node (Station ID)", fontsize=12)
        ax1.set_ylabel("Source Node (Station ID)", fontsize=12)
        
        # ---------------------------------------------------------
        # Plot 2: Node Degree Distribution
        # ---------------------------------------------------------
        # Create a bar chart of how many edges each node has
        ax2.bar(range(num_nodes), node_degrees, color="#3498db", edgecolor="#2980b9")
        
        # Highlight isolated nodes in red
        isolated_nodes = np.where(node_degrees == 0)[0]
        if len(isolated_nodes) > 0:
            ax2.bar(isolated_nodes, node_degrees[isolated_nodes], color="#e74c3c")
            print(f"WARNING: Found {len(isolated_nodes)} isolated nodes! IDs: {isolated_nodes}")
            
        # Draw a line showing the average degree
        avg_degree = np.mean(node_degrees)
        ax2.axhline(avg_degree, color="#e74c3c", linestyle="--", alpha=0.8, label=f"Average Degree: {avg_degree:.1f}")
        
        ax2.set_title("Node Degree Distribution (Number of Neighbors)", fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel("Station ID", fontsize=12)
        ax2.set_ylabel("Number of Edges", fontsize=12)
        ax2.set_xticks(range(0, num_nodes, 2)) # Adjust tick frequency
        ax2.legend()
        ax2.grid(axis='y', linestyle=':', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Success! Visualization saved to '{output_file}'.")

        is_symmetric = np.allclose(adj, adj.T)
        print(f"Symmetric: {is_symmetric}")

        if not is_symmetric:
            diff = np.abs(adj - adj.T)
            print(f"Max asymmetry: {diff.max():.4f}")
            # find which entries differ
            rows, cols = np.where(diff > 1e-6)
            print(f"Number of asymmetric entries: {len(rows)}")
            print(list(zip(rows[:10], cols[:10])))  # peek at a few
        
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Ensure the matrix has been built.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    visualize_adjacency()
