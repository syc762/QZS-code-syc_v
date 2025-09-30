import numpy as np

def average_reduce(array, factor):
    """
    Reduce array size by averaging over windows of size factor
    
    Parameters:
    array: numpy array or list of numbers
    factor: size of the averaging window
    
    Returns:
    reduced_array: list of averaged values
    """
    # Convert to numpy array if it isn't already
    data = np.asarray(array)
    
    # Calculate the number of complete chunks
    n_chunks = len(data) // factor
    
    # Reshape the data into chunks and compute means
    if n_chunks > 0:
        # Handle the complete chunks
        shaped_data = data[:n_chunks*factor].reshape(n_chunks, factor)
        means = np.mean(shaped_data, axis=1)
        
        # Handle any remaining data
        if len(data) > n_chunks*factor:
            last_chunk = data[n_chunks*factor:]
            means = np.append(means, np.mean(last_chunk))
            
        return means.tolist()
    else:
        # If the array is smaller than the factor
        return [np.mean(data)]
    
def convert_to_physical_units(force_voltage, displacement_voltage, Vs_displacement):
    force_voltage = np.array(force_voltage)
    displacement_voltage = np.array(displacement_voltage)
    
    # Convert to physical units
    force = 9.81 * (1.2506 * force_voltage - 0.6525)
    displacement = -38.1/Vs_displacement * displacement_voltage
    displacement = displacement - np.min(displacement)
    
    return force, displacement