import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import functions as func



def create_plotly_animation(filepath, output_dir, Vs_displacement, window_size=10):
    """
    Create an animated plot using Plotly and save as an HTML file
    
    Args:
        filepath (str): Absolute path to the CSV file
        output_dir (str): Absolute path to the output directory for the HTML file
    """
    
    # Get the original filename without extension
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Read the data
    force_voltage = []
    displacement_voltage = []
    with open(filepath, 'r') as f:
        next(f)  # Skip header
        for line in f:
            f, d = line.strip().split(',')
            force_voltage.append(float(f))
            displacement_voltage.append(float(d))
    
    # Convert to numpy arrays for easier manipulation and then convert to physical units
    force, displacement = func.convert_to_physical_units(force_voltage, displacement_voltage, Vs_displacement)
    
    # Average every {window_size} points
    force_averaged = func.average_reduce(force, window_size)
    displacement_averaged = func.average_reduce(displacement, window_size)
    
    # Create frames for animation
    frames = []
    for i in range(len(displacement_averaged)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=displacement_averaged[:i+1],
                        y=force_averaged[:i+1],
                        mode='markers+lines',
                        line=dict(color='blue'),
                        marker=dict(
                            size=8,
                            color='blue',
                        ),
                        hovertemplate='Displacement: %{x:.3f}<br>Force: %{y:.3f}<extra></extra>'
                    )
                ]
            )
        )
    
    # Create the figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[displacement_averaged[0]],
                y=[force_averaged[0]],
                mode='markers+lines',
                line=dict(color='blue'),
                marker=dict(
                    size=8,
                    color='blue',
                ),
                hovertemplate='Displacement: %{x:.3f}<br>Force: %{y:.3f}<extra></extra>'
            )
        ],
        frames=frames
    )
    
    # Update layout
    
    fig.update_layout(
        title=f'Force vs. Displacement Time-lapse ({window_size}-point moving average)',
        xaxis_title='Displacement [mm]',
        yaxis_title='Force [N]',
        showlegend=False,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        # Add a slider
        sliders=[{
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 50, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": str(k),
                    "method": "animate"
                } for k, f in enumerate(frames)
            ]
        }]
    )

    # Save as HTML file
    fig.write_html(os.path.join(output_dir, "{base_filename}_FvsD.html"), auto_play=False)


def process_directory(input_dir, output_dir, Vs_disp_voltage, window_size):
    """
    Process all CSV files in the input directory
    
    Args:
        input_dir (str): Absolute path to input directory containing CSV files
        output_dir (str): Absolute path to output directory for animations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    successful = 0
    failed = 0
    
    for filepath in csv_files:
        if create_plotly_animation(filepath, output_dir, Vs_disp_voltage, window_size):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Output directory: {output_dir}")



if __name__ == "__main__":
    
    # Directory paths
    input_directory = r"C:\Users\students\Desktop\SoyeonChoi\QZS\FvsD_August1\20250807_bestYet2Hz"
    output_directory = r"C:\Users\students\Desktop\SoyeonChoi\QZS\FvsD_August1\20250807_bestYet2Hz"

    # Supply voltage to the linear displacement sensor
    Vs_displacement_voltage = 0.2
    window_size=50
    
    process_directory(input_directory, output_directory, Vs_displacement_voltage, window_size)
    
    