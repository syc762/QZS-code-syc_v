import pytest
from plotter import plot_multiple_graphs

def test_plot_multiple_graphs_valid_input():
    graphs = [
        {'x': [1, 2, 3], 'y': [4, 5, 6], 'label': 'Graph 1'},
        {'x': [1, 2, 3], 'y': [6, 5, 4], 'label': 'Graph 2', 'kwargs': {'linestyle': '--'}}
    ]
    try:
        plot_multiple_graphs(graphs)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

def test_plot_multiple_graphs_missing_keys():
    graphs = [{'x': [1, 2, 3], 'y': [4, 5, 6]}]  # Missing 'label'
    with pytest.raises(ValueError):
        plot_multiple_graphs(graphs)

if __name__ == "__main__":
    pytest.main()
