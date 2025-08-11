import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import os
import glob
import functions as func

def create_mplotlib_animation(filepath, output_dir, Vs_displacement, window_size=10):
    """
    Creates animation from CSV file and saves with matching filename
    
    Args:
        filepath (str): Absolute path to the CSV file
        output_dir (str): Absolute path to the output directory
    """
    # Get the original filename without extension
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Set the backend
    matplotlib.use('Agg')

    try:
        # Read the data
        force = []
        displacement = []
        with open(filepath, 'r') as f:
            next(f)  # Skip header
            for line in f:
                f, d = line.strip().split(',')
                force.append(float(f))
                displacement.append(float(d))

        # Convert to numpy arrays
        force, displacement = func.convert_to_physical_units(force, displacement, Vs_displacement)

        # Average the data
        force_averaged = []
        displacement_averaged = []

        for i in range(0, len(force), window_size):
            force_window = force[i:i+window_size]
            displacement_window = displacement[i:i+window_size]
            
            force_averaged.append(np.mean(force_window))
            displacement_averaged.append(np.mean(displacement_window))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(min(force_averaged) - 0.02, max(force_averaged) + 0.02)
        ax.set_xlim(min(displacement_averaged) - 0.02, max(displacement_averaged) + 0.02)
        ax.set_ylabel('Force [N]')
        ax.set_xlabel('Displacement [mm]')
        ax.set_title(f'Force vs. Displacement: {base_filename}')
        ax.grid(True)

        # Initialize empty scatter plot
        scatter = ax.scatter([], [], c='blue', alpha=0.8, s=50)

        # Animation function
        def animate(frame):
            scatter.set_offsets(np.c_[displacement_averaged[:frame+1], force_averaged[:frame+1]])
            return [scatter]

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(force_averaged),
            interval=100,
            blit=True,
            repeat=False
        )

        # Save animation with matching filename
        output_path = os.path.join(output_dir, f'{base_filename}_animation.gif')
        anim.save(output_path, writer='pillow')
        plt.close()
        
        print(f"Successfully processed: {base_filename}")
        return True

    except Exception as e:
        print(f"Error processing {base_filename}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, Vs_displacement_voltage, window_size):
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
        if create_mplotlib_animation(filepath, output_dir, Vs_displacement_voltage, window_size):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Output directory: {output_dir}")

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual paths
    input_directory = r"C:\Users\students\Desktop\SoyeonChoi\QZS\FvsD_August1\20250807_bestYet2Hz"
    output_directory = r"C:\Users\students\Desktop\SoyeonChoi\QZS\FvsD_August1\20250807_bestYet2Hz"
    
    # Supply voltage to the linear displacement sensor
    Vs_displacement_voltage = 0.2
    window_size=50
    
    process_directory(input_directory, output_directory, Vs_displacement_voltage, window_size)