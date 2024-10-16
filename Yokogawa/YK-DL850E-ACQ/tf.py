import pyvisa
import re
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import textwrap
from tqdm import tqdm
import time
import sys
import pint
import os
import pandas as pd
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
from simpleplot import get_filename


# Globals
ureg = pint.UnitRegistry()


# Function Definitions:

# extracts number in scientific notation from string
def extract_number(string):
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    match = re.search(pattern, string)
    if match:
        number_str = match.group()
        try:
            number = float(number_str)
            return number
        except ValueError:
            pass
    return None


# Converts number to string +Hz
def convert_to_si_prefix(value):
    # Convert the value to the appropriate unit
    quantity = value * ureg.hertz

    # Get the SI prefix and value in that prefix
    prefix, scaled_value = quantity.to_compact()

    # Create the string representation
    string_value = f"{scaled_value:g}{prefix}Hz"

    return string_value


# converts time SI string to float
def convert_to_seconds(time_string):
    try:
        int_time = ureg(time_string)
        return int_time.to('second').magnitude
    except pint.errors.UndefinedUnitError:
        raise ValueError('Invalid unit')
    except pint.errors.DimensionalityError:
        raise ValueError('Invalid time string')


# Saves data to a .csv file
def save_data_to_csv(data, label, save_dir):                                                                                           
    # Create a temporary directory to save the file
    # temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(save_dir, exist_ok=True)

    # Generate a timestamp for the file name
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    file_name = f"{timestamp}_transfer_data_{label}.csv"

    # Create a dictionary to hold the data arrays
    data_dict = {}
    for i, arr in enumerate(data):
        if i == 0:
            tag ="frequency"
        else:
            tag = "channel" + str(i)

        var_name = tag # + f"_variable_{i+1} "
        data_dict[var_name] = arr

    # Create a DataFrame using the data dictionary
    df = pd.DataFrame(data_dict)
    # df["normalized"] = df.iloc[:,[2]]/df.iloc[:,[1]] # Do not use iloc
    df["normalized"] = df['channel2']/df['channel1']
    
    # Save the DataFrame to a CSV file in the temporary directory

    save_file_path = os.path.join(save_dir, file_name)
    
    try:
        df.to_csv(save_file_path, index=False)
    except Exception as e:
        print(e)
        print("File did not save, but will plot the data")
        return df
    
    print(f"File saved to: {save_file_path}")
    return df

    # Open a file dialogue for the user to choose the save location
    '''
    try:
        Tk().withdraw()
        file_path = asksaveasfilename(
            defaultextension=".csv",
            initialdir=os.path.expanduser("~\\Documents"),
            initialfile=file_name,
            filetypes=[("CSV files", "*.csv")]
        )

        if file_path:
            # Copy the temporary file to the chosen save location
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            os.replace(temp_file_path, file_path)
            print(f"File saved to: {file_path}")
        else:
            print("File save operation cancelled.")
    except Exception as e:
        print(f"An error occurred while saving the file: {str(e)}")
    '''
    return df


def damping_func(t, A, l, w, p):
    return A * np.exp(-1 * l * t) * np.cos(w * t - p)


def split_even_odd(lst):
    def __pick(lst, remainder):
        return [lst[i] for i in range(len(lst)) if i%2==remainder]
    
    odd_list = __pick(lst, 1)
    even_list = __pick(lst, 0)

    return even_list, odd_list


"""
if 'frequency domain' or 'resonance' in self.mode:
                    data['t_acc'] = (9.81 / 10) * t_data / self.amp_gain # Q. Where does this 9.81 value come from?
                    if 'frequency domain' in self.mode:
                        # freq, psd_acc = sc.signal.periodogram(data['t_acc'], fs=sampling_rate)
                        freq, psd_acc = sc.signal.welch(data['t_acc'],fs=sampling_rate,nperseg=sampling_rate,window='blackman',noverlap=0)
                        
                        freq = freq[1:-1]
                        psd_acc = psd_acc[1:-1]

                        data['f'] = freq
                        data['psd_acc'] = psd_acc
                        data['psd_pos'] = psd_acc / freq ** 2
                self.channel_data[channel] = data
"""





def plot_arrays(time, timeData, freq, psdData, plotType):
    if 'resonance' in plotType:
        x = time
        y = timeData

        popt, pcov = sc.optimize.curve_fit(damping_func, x, y)
        [A, l, w, p] = popt
        zeta = l / np.sqrt(l ** 2 + w ** 2)
        delta = 2 * 3.1416 * zeta / np.sqrt(1 - zeta ** 2)
        f = w / (2 * 3.1416)
        plt.plot(x, y)
        plt.plot(x, damping_func(x, A, l, w, p))
        titlestring = 'Resonance Measurement of QZS Flexure Component\n'
        #titlestring += r'$y = A \exp{-\lambda t} \cos{(\omega t - \varphi})$' + '\n'
        titlestring += rf'$A = {A:.2e}$' ', ' \
                       rf'$\lambda = {l:.2e}$' ', ' \
                       rf'$\omega = {w:.2e}$' ', ' \
                       rf'$\varphi = {p:.2e}$' '\n' \
                       rf'$f_n = {f:.2e}$' ', ' ', ' \
                       rf'$\delta = {delta:.2e}$'
        plt.title(titlestring)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration ($m/s^2$)')
        plt.legend(['data', 'fit'])
        plt.show()

    if 'timeandfreq' in plotType:
        xlim = [0, 6E2]
        ylim = []

        i_xlim = np.argmax(freq > 1E3)
        #ylim.append(1E-1 * np.min(psdData[0:i_xlim]))
        #ylim.append(1E1 * np.max(psdData[0:i_xlim]))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(time, timeData)
        ax1.set_title('Time Domain Signal')
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('voltage (V)')

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel(r'Position PSD $m/\sqrt{Hz}$')
        ax2.set_title('Power Spectral Density')
        ax2.set_xlim(xlim[0], xlim[1])
        # ax2.set_ylim(ylim[0], ylim[1])
        ax2.semilogy(freq, psdData)

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.4)
        plt.show()



class tf:
    def __init__(self):
        ##### *!* Channels
        self.channel = [1,3] ### original: str(1) ### 
        # Note: Since January 2024, channel 2 & 3 [2,3] used for transfer function
        # Channel 1,2,3 used for 'X vs Y' or Force vs. Displacement curve
        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.agilentAddress = 'USB0::0x0957::0x0407::MY44026553::INSTR'
        self.chunkSize = int(1E5)

    def open_instruments(self):
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
       
        if self.yokogawaAddress in resources:
            self.yk = rm.open_resource(self.yokogawaAddress)
        else:
            print('Failed to find Yokogawa!')
            sys.exit()
        if self.agilentAddress in resources:
            self.ag = rm.open_resource(self.agilentAddress)
        else:
            print('Failed to find Agilent!')
            sys.exit()

    def __initialize_yokogawa(self, sample_rate='10k', time_div='2s'):

        self.yk.write(':STOP')
        self.yk.write(':CALIBRATE:MODE OFF')

        # General Timebase settings
        self.yk.write(':TIMebase:TDIV ' + '1s')
        self.yk.write(':TIMebase:SRATe' + sample_rate)
        self.yk.write(':TIMebase:TDIV ' + time_div)
    
        ##### *!* 
        # Waveform settings 
        for c in self.channel:
            self.yk.write(':WAVeform:TRACE ' + str(c))
        result = self.yk.query('WAVEFORM:RECord? MINimum')
        min_record = int(extract_number(result))
        self.yk.write(':WAVeform:RECord ' + str(min_record))
        print("min_record: " + str(min_record))
        self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')
        self.yk.write(':WAVeform:FORMat WORD')


    def __initialize_agilent(self, voltage='10.0', shape='SINusoid', offset='0.0'):
        self.ag.write('*RST')
        self.ag.write('FUNCtion ' + shape)
        self.ag.write('VOLTage ' + voltage)
        self.ag.write('VOLTage:OFFSet ' + offset)
        self.ag.write('FREQuency 1')

    def initialize_instruments(self, sample_rate='10kHz', time_div='2s', voltage='10.0', shape='SINusoid', offset='0.0'):
        self.__sample_rate = sample_rate
        self.__time_div = time_div
        self.__initialize_agilent(voltage=voltage, shape=shape, offset=offset)
        self.__initialize_yokogawa(sample_rate=sample_rate, time_div=time_div)


    def __find_peak(self, c, frequency_of_interest, time_div, bin_size=0.5, t_y_p_e='customized'):


        def __cap_data(self, c, type=t_y_p_e):

            # print("Obtaining data for channel " + str(c))
            self.yk.write(':WAVeform:TRACE ' + str(c))

            #####
            # Download data from scope, convert it to an acceleration signal
            result = self.yk.query(':WAVeform:SRATe?')  # Get sampling rate
            sampling_rate = extract_number(result)

            result = self.yk.query('WAVEFORM:LENGth?')  # Get waveform length
            waveform_length = int(extract_number(result))

            n = int(np.floor(waveform_length / self.chunkSize))

            # Currently is collecting bit data from one channel
            bit_data = []

            for i in range(n + 1):
                m = min(waveform_length, (i + 1) * self.chunkSize) - 1
                self.yk.write("WAVEFORM:START {};:WAVEFORM:END {}".format(i * self.chunkSize, m))
                buff = self.yk.query_binary_values('WAVEFORM:SEND?', datatype='h', container=list)
                bit_data.extend(buff)

            result = self.yk.query('WAVEFORM:OFFSET?')
            waveform_offset = extract_number(result)

            result = self.yk.query(':WAVeform:RANGe?')
            waveform_range = extract_number(result)

            voltage_data = waveform_range * np.array(
                bit_data) * 10 / 24000 + waveform_offset  # formula from the communication manual

            acceleration_data = 9.81 / 10 * voltage_data / 100
            time_data = np.array(range(len(acceleration_data))) / sampling_rate

            
            # Calculate the position power spectral density using welch or periodogram:
            if type == 'welch':

                freq, psd_acc = sc.signal.welch(acceleration_data, fs=sampling_rate, nperseg=sampling_rate, window='blackman',noverlap=0)       


            else: # periodogram

                freq, psd_acc = sc.signal.periodogram(acceleration_data,
                                                      fs=sampling_rate
                                                      )
            
            freq = freq[1:-1]
            psd_acc = psd_acc[1:-1]
            psd_pos = psd_acc / freq ** 2
            psd_data = psd_pos
            
            ### Plot the fourier spectrum for each channel ###


            # Find the max value within bin range_width, centered on frequency_of_interest
            indices = np.where((freq >= frequency_of_interest - bin_size / 2) & (freq <= frequency_of_interest + bin_size / 2))
            psd_data_range = psd_data[indices]
            try:
                max_value = np.max(psd_data_range)
            except Exception as e:
                print(e)
                return 0
            
            return max_value ## Lengthy acceleration data and time data. Take the time data and compute v_o*t + 1/2*a*t^2 for displacement 

        max_value_list = __cap_data(self, c,  t_y_p_e)    # [__cap_data(self, c) for c in self.channel]
        
        return max_value_list #, psd_data


    def __measurement_cycle(self, frequency, num_iterations, time_div, bin_size=1):
        
        means = []

        self.ag.write('FREQuency ' + str(frequency))
        self.ag.write('OUTPut ON')
        time.sleep(0.1)
        
        
        # We change the TDIV to 1s to force a refresh and ensure we are getting the most current data
        self.yk.write(':TIMebase:TDIV ' + '1s')
        self.yk.write(':TIMebase:SRATE ' + self.__sample_rate)
        self.yk.write(':TIMebase:TDIV ' + time_div)

        
        ##### Yokogawa
        # Waveform capture sequence
        self.yk.write(':START')
        print(time_div)
        if convert_to_seconds(time_div) < 0.5:
            time.sleep(12 * convert_to_seconds(time_div))
        else:
            time.sleep(11 * convert_to_seconds(time_div))
        self.yk.write(':STOP')   

        ####### For each channel, need to __find_peak and obtain the measurement values
        for c in self.channel:
            # measurement_values = [self.__find_peak(c, frequency, time_div, bin_size=bin_size) for _ in tqdm(range(num_iterations))]
            measurement_values = [self.__find_peak(c, frequency, time_div, bin_size=bin_size, t_y_p_e='welch') for _ in tqdm(range(num_iterations))]
            
            # measurement_values, acceleration_data = split_even_odd(data)
            
            # Method to collect acceleration data here
            means.append(np.mean(measurement_values))
            # stds.append(np.std(measurement_values))
            
        self.ag.write('OUTPut OFF')

        return means


    ##### *!* Channels
    def measure(self, frequencies, iterations, time_div, bin_size=1):
        
        # Hard-coded, assuming two channel input
        results1 = []
        results2 = []
        # acc1 = []
        # acc2 = []
        # acc_norm = []
        # pos_norm = []

        # NO channel information applied here!
        for freq, it, t in tqdm(zip(frequencies, iterations, time_div)):

            mean = self.__measurement_cycle(freq, it, t, bin_size=bin_size)
            results1.append((mean[0]))
            results2.append((mean[1]))
            # acc1.append(acc[0])
            # acc2.append(acc[1])
            # acc_norm.append(acc[1]/acc[0])

            # pos = acc[1:-1]/freq**2
            # pos_norm.append(pos)

        return results1, results2 #, acc1, acc2, acc_norm, pos_norm


    def close_instruments(self):
        self.ag.write('OUTPut OFF')
        self.yk.write(':STOP')


def plot_tf_from_df(df, filename, save_dir): # Need to change which column it's referring to based on the old excel plots
    
    x = df.iloc[:,0].to_numpy()
    normalized = df.iloc[:,3].to_numpy()

    if 'channel1' in df.columns:
        V_out = df.iloc[:,1].to_numpy() 
        V_in = df.iloc[:,2].to_numpy()
    elif 'V_in' in df.columns:
        V_out = df['V_out'].to_numpy()
        V_in = df['V_in'].to_numpy()
    else:
        print(f"Error reading in columns, check the column names")
        return 1
    
    # Labels used in the past: label="$V_{in}$", "$V_{out}$"
    plt.plot(x, normalized, label ="$|V_{out}/V_{in}|$", c="darkorange", marker='.', linestyle='solid')
    plt.plot(x, V_in, label="$V_{in}$", c="cornflowerblue", marker='.', linestyle='dashed')
    plt.plot(x, V_out, label="$V_{out}$", c="mediumblue", marker='.', linestyle='dashed')
    #plt.plot(df["frequency"], df["channel2"])
    #plt.plot(df["frequency"], df["channel1"])
    #plt.plot(df["frequency"], df["normalized"])
    plt.loglog()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Vout/Vin")
    plt.legend()
    
    title = get_filename(filename)
    wrapped_title = textwrap.fill(title, width=52)
    plt.title(wrapped_title, fontsize=11, wrap=True) # Attenuation
        
    try:
        plt.savefig(os.path.join(save_dir, filename), transparent=True, bbox_inches='tight')
    except Exception as e:
        print(e)
    plt.close()


### Beginning of Main ###

# Sweeps frequences in the range 0, 200 with 20 steps in between
numPoints = 400
volt = ['4.0'] # Must be lower than 600mV if using the amplifier!!! Units in V, so 0.6V
# Run label
springType = "noAirlegs_noDampers_brokenFlexure_100x_ch1PSU2DOWN_ch2PSU1UP"






# Up to 50Hz it's ok. 1,67



if __name__ == "__main__":

    tf = tf()
    
    for v in volt:

        label = springType + "_tf_" + str(numPoints) + "points_" + v +"V"

        datestamp = datetime.now().strftime('%Y%m%d')
        save_folder = f"{datestamp}_{springType}"

        save_dir=os.path.join(os.path.expanduser("~\\Desktop\SoyeonChoi"), save_folder)

        frequency = np.logspace(0, 3, num=numPoints) 
        # Will take different frequency values 
        iterations = [1 if freq > 4 else 1 for freq in frequency]
        time_divisions = ['2s' if freq < 2 else '500ms' if freq < 10 else '200ms' for freq in frequency]



        
        tf.open_instruments()
        tf.initialize_instruments(voltage=v)
        all_transfer_data = tf.measure(frequency, iterations, time_divisions, bin_size=1)
        tf.close_instruments()

        # Assumes we are collecting from two channels only
        # print(all_transfer_data)
        means1 = all_transfer_data[0]
        means2 = all_transfer_data[1]

        ### Saves the data to a csv file

        df = save_data_to_csv([frequency, means1, means2], label, save_dir)
        # save_data_to_csv([frequency, means1, stds1], "test_10points_1channels")

        ### Directly use the data to obtain the transfer function
        plot_tf_from_df(df, label, save_dir)

        # Convert the acceleration data to position data


        # Using the built-in method
        # plot_tf(df, label)


        # First iteration had a board marker dropping in the trashcan sound