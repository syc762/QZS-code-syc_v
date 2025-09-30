import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_multiple_graphs(graphs: List[Dict[str, Any]], figsize: tuple = (10, 6)):
    """
    Plots multiple graphs in one plot with specified parameters.

    Args:
        graphs (List[Dict[str, Any]]): A list of dictionaries, each containing parameters for a single graph.
        figsize (tuple): Size of the figure.

    Raises:
        ValueError: If required keys are missing in the graph dictionaries.
    """
    try:
        plt.figure(figsize=figsize)
        for graph in graphs:
            if not all(key in graph for key in ['x', 'y', 'label']):
                raise ValueError("Each graph dictionary must contain 'x', 'y', and 'label' keys.")
            x = graph['x']
            y = graph['y']
            label = graph['label']
            plt.plot(x, y, label=label, **graph.get('kwargs', {}))
        
        plt.xlabel(graphs[0].get('xlabel', ''))
        plt.ylabel(graphs[0].get('ylabel', ''))
        plt.title(graphs[0].get('title', ''))
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Graphs plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    example_graphs = [
        {'x': [1, 2, 3], 'y': [4, 5, 6], 'label': 'Graph 1'},
        {'x': [1, 2, 3], 'y': [6, 5, 4], 'label': 'Graph 2', 'kwargs': {'linestyle': '--'}}
    ]
    plot_multiple_graphs(example_graphs)
