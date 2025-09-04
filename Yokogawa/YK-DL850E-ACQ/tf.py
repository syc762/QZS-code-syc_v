# Places marked #HERE# are the ones that need to be changed
# -*- coding: utf-8 -*-

import pyvisa
import logging
import re
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import textwrap
from tqdm import tqdm
import time
import math
import sys
import pint
import os
import pandas as pd
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
import psd as psd


# Globals
ureg = pint.UnitRegistry()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###### Function Definitions


def get_filename(targetFile, extension='.csv'):
    # Get the base filename with extension
    base_filename = os.path.basename(targetFile)
    
    # Remove the extension
    filename_without_extension = os.path.splitext(base_filename)[0]
    
    return filename_without_extension

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

def parse_unit_string(value_str):
    """
    Parses a string with SI units like '500ms', '2s', '10kHz', '2MHz' and converts it to a float in base units.
    - Time strings are converted to seconds (s)
    - Frequency strings are converted to hertz (Hz)

    Supports units: ms, s, Hz, kHz, MHz, GHz

    Returns:
        float: value in base SI units (seconds or hertz)

    Raises:
        ValueError: if the format is unrecognized or invalid
    """
    value_str = value_str.strip().lower()

    # Ordered by decreasing length to avoid 'hz' matching 'khz' or 'mhz' prematurely
    unit_multipliers = {
        'ghz': 1e9,
        'mhz': 1e6,
        'khz': 1e3,
        'hz': 1,
        'ms': 1e-3,
        's': 1,
        'k': 1e3,
        'm': 1e6,
    }

    for unit in sorted(unit_multipliers, key=len, reverse=True):
        if value_str.endswith(unit):
            number_part = value_str[: -len(unit)].strip()
            try:
                return float(number_part) * unit_multipliers[unit]
            except ValueError:
                raise ValueError(f"Invalid number in string: {value_str}")

    raise ValueError(f"Unrecognized unit format: {value_str}")


# returns the nearest valid record length when using the ACQ class for the Yokogawa scope
def round_acq_record_length(val):
    valid_steps = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000,
                   1000000, 2500000, 5000000, 10000000, 25000000, 50000000,
                   100000000, 250000000, 500000000, 1000000000, 2000000000]
    return min((x for x in valid_steps if x >= val), default=valid_steps[-1])

# returns the nearest valid sampling rate for the Yokogawa scope
def round_sample_rate(val):
    valid_rates = [
        5,
        10, 20, 50,
        1e2, 2e2, 5e2,
        1e3, 2e3, 5e3,
        1e4, 2e4, 5e4,
        1e5, 2e5, 5e5,
        1e6, 2e6, 5e6,
        10e6, 20e6, 50e6,
        100e6
    ]  # From 100 S/s to 100 MS/s
    return max((x for x in valid_rates if x <= val), default=valid_rates[0])


# Saves data to a .csv file
def save_data_to_csv(data, label, save_dir, timestamp, data_type):                                                                                           
    # Create a temporary directory to save the file
    # temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(save_dir, exist_ok=True)

    # Generate a timestamp for the file name
    file_name = f"{timestamp}_transfer_data_{label}_{data_type}.csv"

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
    
    # Check whether channel1 is input or output
    #df["normalized"] = df['channel1']/df['channel2']
    
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
        self.channel = [1,2] ### original: str(1) ###  [1,3]
        # Note: Since January 2024, channel 2 & 3 [2,3] used for transfer function
        # Channel 1,2,3 used for 'X vs Y' or Force vs. Displacement curve
        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.agilentAddress = 'USB0::0x0957::0x0407::MY44026553::INSTR'
        self.chunkSize = int(1E5)
        self.logger = logging.getLogger(__name__)


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

        # Remote control
        self.yk.write(':COMMunicate:REMote ON')
        self.logger.info(f"Remote control enabled")
        remote_status = self.yk.write(':COMMunicate:REMOTE?')
        self.logger.info(f"Remote status: {remote_status}")

        self.yk.write(':STOP')
        cal_mode = self.yk.write(':CALibrate:MODE?')
        self.logger.info(f"Current calibration mode: {cal_mode}")
        self.yk.write(':CALIBRATE:MODE OFF')
        cal_mode = self.yk.write(':CALibrate:MODE?')
        self.logger.info(f"Calibration mode has been set to: {cal_mode}")
        
        self.yk.write(':ACQuire:MODE NORMal')
        self.yk.write(':WAVEFORM:FORMAT WORD')
        self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')

        # General Timebase settingsf
        self.yk.write(':TIMebase:TDIV ' + '1s')
        self.yk.write(':TIMebase:SRATe' + sample_rate)
        q = self.yk.write(':TIMebase:SRATe?')
        self.logger.info(f"Sample rate set to: {q}")
        self.yk.write(':TIMebase:TDIV ' + time_div)
    
        ##### *!* 
        # Waveform settings 
        for c in self.channel:
            
            # single channel capture
            self.yk.write(':WAVeform:TRACE ' + str(c))
            # dual channel capture
            #self.yk.write(':CAPTure:CH'+str(c)+':DATA ON')  # Enable capture for CH1
            
            
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

    def initialize_instruments(self, sample_rate='10k', time_div='2s', voltage='10.0', shape='SINusoid', offset='0.0'):
        self.__sample_rate = sample_rate
        self.__time_div = time_div
        self.__initialize_agilent(voltage=voltage, shape=shape, offset=offset)
        self.__initialize_yokogawa(sample_rate=sample_rate, time_div=time_div)


    def __find_peak(self, c, numChanels, shapeType, frequency_of_interest, bin_size=0.5, t_y_p_e='welch', timestamp='0'):


        def __cap_data(self, c, numChannels, shapeType, type=t_y_p_e, timestamp=timestamp):

            if numChannels == 'dual':
                self.yk.write(':CAPTURE:CH{c}:DATA?')
                raw_ch = self.yk.read_raw()
                data_ch = np.frombuffer(raw_ch[16:], dtype=np.int16)  # Skip 16-byte header

            else:

                self.yk.write(':WAVeform:TRACE ' + str(c))

            

            #####
            # Download data from scope, convert it to an acceleration signal
            result = self.yk.query(':WAVeform:SRATe?')  # Get sampling rate
            sampling_rate = extract_number(result)
            self.logger.info(f"WAVeform sampling rate for channel {c}: {sampling_rate} Hz")
            # I want to write the :WAVeform:SRATe? to a file
            with open(os.path.join(save_dir, f"{frequency_of_interest}Hz_README.txt"), 'a') as f:
                f.write(f"WAVeform:SRATE for channel {c}: {sampling_rate} s\n")


            result = self.yk.query('WAVEFORM:LENGth?')  # Get waveform length
            waveform_length = int(extract_number(result))
            print("Waveform length: " + str(waveform_length))

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

             # Check that the waveform range is not so small / the bit-data values are not around 0
            print(f"Waveform Range: {waveform_range}, Offset: {waveform_offset}, Bit sample (first 10): {bit_data[:10]}")

            voltage_data = waveform_range * np.array(
                bit_data) * 10 / 24000 + waveform_offset  # formula from the communication manual

            print(f"Voltage data (first 10): {voltage_data[:10]}")
            
            acceleration_data = 9.81 / 10 * voltage_data / 100

            #  ### Save the raw acceleration data and plot the first 2 seconds of the waveform
            
            
            acc_df = pd.DataFrame(acceleration_data, columns=["acceleration (m/s^2)"])
            acc_df.to_csv(os.path.join(save_dir, f"ch{c}_acceleration_data_{frequency_of_interest}Hz.csv"), index=False)

            # Plot channel data assuming 10kSamples/s
            if frequency_of_interest > 10:
                end_time = 0.5
            else:
                end_time = 2.0
            psd.plot_acceleration_segment(acceleration_data, frequency_of_interest, sampling_rate, f"CH{c}", start_time=0.0, end_time=end_time, save_dir=save_dir)

            # optimal_nperserg = psd.optimal_nperseg(acceleration_data, sampling_rate, freq_of_interest, waveform_length,
                    # min_cycles=5, max_cycles=18, window='hamming', noverlap=None, verbose=False)
           
            # Calculate the position power spectral density using welch or periodogram:
            if type == 'welch':
                
                if shapeType == 'SINusoid':
                    self.logger.info(f"Using Welch method for PSD calculation with {shapeType} waveform")
                    freq, psd_acc = sc.signal.welch(
                        acceleration_data,
                        fs=sampling_rate,
                        window='hamming'
                    )
                    
                    
                elif shapeType == 'SQUare':
                    self.logger.info(f"Using Welch method for PSD calculation with {shapeType} waveform")
                    # nperseg = max(waveform_length // 100, 256)
                    
                    nperseg = sampling_rate # psd.optimal_nperseg_for_square(sampling_rate, len(acceleration_data), frequency_of_interest, max_k=100)
                    noverlap = int(min(nperseg // 2, nperseg - 1))
                    windowType = 'hamming'

                    with open(os.path.join(save_dir, f"{frequency_of_interest}Hz_README.txt"), 'a') as f:
                        f.write(f"Using nperseg: {nperseg}, noverlap: {noverlap} for {shapeType} waveform\n")
                    
                    self.logger.info(f"Using nperseg: {nperseg}, noverlap: {noverlap}, window:{windowType} for {shapeType} waveform")

                    freq, psd_acc = sc.signal.welch(
                        acceleration_data,
                        fs=sampling_rate,
                        window=windowType
                    )
                    print("check psd_acc length")
                    print(len(psd_acc))

                else:
                    self.logger.info(f"{shapeType} Not a registered waveform shape")
                    pass

            else: # periodogram

                freq, psd_acc = sc.signal.periodogram(acceleration_data,
                                                      fs=sampling_rate
                                                      )
            
            
            # Save freq, psd_data for each input signal
            raw_df = pd.DataFrame({'frequency': freq, 'psd_data': psd_acc})

            # NOTE: add the signal type to the filename
            raw_df.to_csv(os.path.join(save_dir, f"ch{c}_psd_input_{frequency_of_interest}Hz.csv"), index=False)

            ### Plot the PSD for each channel ###
            psd.plot_psd(c, freq, psd_acc, frequency_of_interest, save_dir)
            

            ### Analyze the PSD data ###
            freq = freq[1:-1]
            psd_acc = psd_acc[1:-1]
            psd_pos = psd_acc / freq ** 2
            psd_data = psd_pos
            psd_peak_index = np.argmax(psd_acc)


            # Find the max value within bin range_width, centered on frequency_of_interest
            indices = np.where((freq >= frequency_of_interest - bin_size / 2) & (freq <= frequency_of_interest + bin_size / 2))
            psd_data_range = psd_data[indices]
            
            try:
                max_value = np.max(psd_data_range)
                max_value = np.max(psd_data)
                max_Value_index = np.argmax(psd_data)
                max_Value_from_index = psd_data[max_Value_index]
            except Exception as e:
                print(e)
                return 0
            

        
            
            #self.logger.info(f"Max value for channel {c} at frequency {rounded_freq}Hz: {max_value}")
            #self.logger.info(f"Max value from index {max_Value_index} at frequency {rounded_freq}Hz: {max_Value_from_index}")
            return max_value ## Lengthy acceleration data and time data. Take the time data and compute v_o*t + 1/2*a*t^2 for displacement 


        # Beginning of the __find_peak function
        max_value_list = __cap_data(self, c, numChannels, shapeType, t_y_p_e)    # [__cap_data(self, c) for c in self.channel]
        
        return max_value_list #, psd_data


    def __measurement_cycle(self, numChannels, shapeType, frequency, num_iterations, bin_size=1, timestamp='0'):
        
        means = []

        self.ag.write('FREQuency ' + str(frequency))
        self.ag.write('OUTPut ON')
        time.sleep(1.0) # Wait for the ring-down to settle. Ideally, this will be dynamically adjusted based on the expected resonance of the flexure
        
        # We change the TDIV to 1s to force a refresh and ensure we are getting the most current data
        self.yk.write(':TIMebase:TDIV ' + '1s')

        # Set the sampling rate
        srate = str(parse_unit_string(self.__sample_rate))
        print(f"after parse_unit_string: {srate}")
        self.yk.write(':TIMebase:SRATe ' + srate) # 10kHz
        queried_sample_rate = extract_number(self.yk.query(':TIMebase:SRATe?'))
        self.logger.info(f"Actual sample rate - 1st: {queried_sample_rate}")
        
        # Set the record length
        record_length = 20 * int(parse_unit_string(self.__sample_rate))
        
        record_length = round_acq_record_length(record_length)
        self.logger.info(f"Setting the record length to : {record_length}")


        if numChannels == 'dual':
            self.yk.write(':CAPTure:RLENgth ' + str(record_length))
            confirmed_capture_len = extract_number(self.yk.query(':CAPTure:RLENgth?'))
        else:
            self.yk.write(':ACQuire:RLENgth ' + str(record_length)) # For single channel
            confirmed_capture_len = extract_number(self.yk.query(':ACQuire:RLENgth?'))
        
        # Check whether the record length is set correctly
        self.logger.info(f"Confirmed capture length: {confirmed_capture_len}, meant to set to {record_length}")

        # Waveform capture sequence
        # Check whether the time division is set correctly
        
        actual_time_div = extract_number(self.yk.query(':TIMebase:TDIV?'))
        # actual_time_div = extract_number(self.yk.query(':CAPTure:TDIV?'))
        self.logger.info(f"Queried TDIV: {actual_time_div}")

        queried_sample_rate = extract_number(self.yk.query(':TIMebase:SRATe?'))
        self.logger.info(f"Actual sample rate - 2nd: {queried_sample_rate}")
        
        # Small adjustment to acquisition time based on the record length and sample rate
        acquisition_time = float(record_length)/queried_sample_rate * 1.20
        self.logger.info(f"Will be acquiring for: {acquisition_time} seconds")

        # Write to a text file the final queried_sample_rate, the actual_time_div, and the record_length for the driving frequency
        with open(os.path.join(save_dir, f"{frequency}Hz_README.txt"), 'w') as f:
            f.write(f"Input waveform shape: {shapeType} \n")
            f.write(f"Frequency: {frequency} Hz\n")
            f.write(f"Sample Rate: {queried_sample_rate} Hz\n")
            f.write(f"Time Division: {actual_time_div} s\n")
            f.write(f"Record Length: {record_length}\n")
            f.write(f"Acquisition Time: {acquisition_time} s\n")

        self.yk.write(':START')
        time.sleep(acquisition_time) # 0.5s for 10kHz sampling rate, 1s for 20kHz sampling rate
        
        self.yk.write(':STOP')   


 
        # New: give scope time to write waveform to memory
        time.sleep(1)  # 1–2 seconds delay

        # New: poll until waveform is non-zero length (max wait ~5s)
        for _ in range(10):
            waveform_length = int(extract_number(self.yk.query('WAVEFORM:LENGth?')))
            if waveform_length > 0:
                break
            time.sleep(0.5)

        if waveform_length == 0:
            raise RuntimeError("No waveform data captured. Scope returned length 0.")

        self.logger.info(f"Freq: {frequency:.2f} Hz, TDIV: {actual_time_div}s, SRATE: {queried_sample_rate} Hz, RLEN: {record_length}, WLEN: {waveform_length}")

        ####### For each channel, need to __find_peak and obtain the measurement values
        for c in self.channel:
            # measurement_values = [self.__find_peak(c, frequency, time_div, bin_size=bin_size) for _ in tqdm(range(num_iterations))]
            measurement_values = [self.__find_peak(c, numChannels, shapeType, frequency, bin_size=bin_size, t_y_p_e='welch', timestamp=timestamp) for _ in tqdm(range(num_iterations))]
            
            # measurement_values, acceleration_data = split_even_odd(data)
            print(f"Measurement values for channel {c}: {measurement_values}")
            
            # Method to collect acceleration data here
            means.append(np.mean(measurement_values))
        
            
        self.ag.write('OUTPut OFF')

        return means


    ##### *!* Channels
    def measure(self, numChannels, shapeType, frequencies, iterations, bin_size=1, timestamp='0'):
        
        # seconds
        # Hard-coded, assuming two channel input
        results1 = []
        results2 = []
        # acc1 = []
        # acc2 = []
        # acc_norm = []
        # pos_norm = []

        # NO channel information applied here!
        for freq, it in tqdm(zip(frequencies, iterations)):

            print("beginning of measurement cycle")
            print(f"Frequency: {freq}, Iterations: {it}")

            mean = self.__measurement_cycle(numChannels, shapeType, freq, it, bin_size=bin_size, timestamp=timestamp)

            if len(mean) > 0:
                results1.append(mean[0])
            if len(mean) > 1:
                results2.append(mean[1])
            
            # acc1.append(acc[0])
            # acc2.append(acc[1])
            # acc_norm.append(acc[1]/acc[0])

            # pos = acc[1:-1]/freq**2
            # pos_norm.append(pos)

        return results1, results2 #, acc1, acc2, acc_norm, pos_norm  #HERE


    def close_instruments(self):
        self.ag.write('OUTPut OFF')
        self.yk.write(':STOP')
        self.yk.write(':CALibrate:MODE OFF')
        self.yk.write(':COMMunicate:REMote OFF')


def plot_tf_from_df(df, filename, save_dir, timestamp, label): # Need to change which column it's referring to based on the old excel plots
    
    x = df.iloc[:,0].to_numpy()
    # normalized = df.iloc[:,3].to_numpy() # HERE

    # Previous logic to plot the data
    """
    if 'channel1' in df.columns:
        V_out = df.iloc[:,1].to_numpy() 
        V_in = df.iloc[:,2].to_numpy()
    elif 'V_in' in df.columns:
        V_out = df['V_out'].to_numpy()
        V_in = df['V_in'].to_numpy()
    else:
        print(f"Error reading in columns, check the column names")
        return 1
    """

    V_signal = df.iloc[:,1].to_numpy()
    
    # Labels used in the past: label="$V_{in}$", "$V_{out}$"
    plt.plot(x, normalized, label ="$|V_{out}/V_{in}|$", c="darkorange", marker='.', linestyle='solid') # HERE
    plt.plot(x, V_signal, label="$V_{in}$", c="cornflowerblue", marker='.', linestyle='dashed')
    plt.plot(x, V_out, label="$V_{out}$", c="mediumblue", marker='.', linestyle='dashed')
    plt.plot(df["frequency"], df["channel2"])
    plt.plot(df["frequency"], df["channel1"])
    plt.plot(df["frequency"], df["normalized"])
    plt.loglog()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(label)
    plt.legend()
    
    title = get_filename(filename)
    wrapped_title = textwrap.fill(title, width=52)
    plt.title(wrapped_title, fontsize=11, wrap=True) # Attenuation
        
    try:
        plt.savefig(os.path.join(save_dir, timestamp + "_" + filename +".png"), transparent=True, bbox_inches='tight')
    except Exception as e:
        print(e)
    plt.close()


### Beginning of Main ###

# Sweeps frequencies in the range 0, 200 with 20 steps in between
numPoints = 38
start_freq = 18
end_freq = 70 #24Hz will be 0.5V
volt = ['2.0'] # AWG Voltage
numChannels = 'single' # either 'single' or 'dual'

shapeType = 'SINusoid'  #  '\\\\\\\\SQUare'
springType = "bestYet2Hz_0.9588kg_newSetup" # "bestYet2Hz_flexureOnly_0.5452kg" # _finer_vol67
# "noAirlegs_flexureNorm_copperPlate_sixPE016springs_2rot-2rot_7136_100x"
data_type="ch1top_ch2bot_x10"

"""
Enter the driving frequency range.
The start_freq and end_freq will be used to generate the frequency array:
np.logspace(np.log10(start_freq), np.log10(end_freq), num=numPoints)
"""


# Up to 50Hz it's ok. 1,67


if __name__ == "__main__":

    tf = tf()
    
    for v in volt:

        # Wait for 60s before running each test with a particular input signal of a particular amplitude
        
        label = springType + "_" + data_type + "_tf_" + v +"V"+ shapeType

        datestamp = datetime.now().strftime('%Y%m%d%H%M')
        timestamp = datetime.now().strftime('%H%M%S')
        save_folder = f"{datestamp}_{label}"

        # save_dir=os.path.join(os.path.expanduser("~\\Desktop\SoyeonChoi\QZS"), save_folder)
        save_dir = os.path.join(os.path.expanduser(r"Z:\Users\Soyeon\Sep"), save_folder)

        if shapeType == 'SQUare':
            baseFreq= [7] # 80Hz = use 2.5V
            frequency = baseFreq
        else: # shapeType == 'Sinusoid'
            
            '''
            Option1
            '''
            # # First 50 odd harmonics: 1st to 99th (step 2)
            # odd_multipliers = np.arange(1, 100, 2)

            # # Compute harmonics using outer product
            # harmonics_matrix = np.outer(baseFreq, odd_multipliers)

            # # Filter values ≤ 1000 Hz, flatten, sort, remove duplicates
            # all_odd_harmonics_capped = np.unique(harmonics_matrix[harmonics_matrix <= 1000])

            # # Optionally convert to a list
            # harmonics_list = all_odd_harmonics_capped.tolist()
            # print("The number of frequencies to sweep over: " + str(len(harmonics_list)))
            # # input("Press Enter to continue: ")

            # # Print as comma-separated values (rounded for clarity)
            # #print(", ".join(f"{h:.2f}" for h in harmonics_list))

            # frequency = harmonics_list

            '''
            Option2
            '''
            # frequency = np.logspace(np.log10(start_freq), np.log10(end_freq), num=numPoints) #np.log10(30), np.log10(26)
            frequency = np.linspace(start_freq, end_freq, num=numPoints)
        

        # Will take different frequency values 
        iterations = [1 if freq > 4 else 1 for freq in frequency]
        
        tf.open_instruments()
        

        """ Initialize the instruments with the specified parameters """
        """
        Note on the valid record lengths:
                  [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000,
                   1000000, 2500000, 5000000, 10000000, 25000000, 50000000,
                   100000000, 250000000, 500000000, 1000000000, 2000000000]
        """
        tf.initialize_instruments(sample_rate='10k', voltage=v, shape=shapeType)
        # Can I do a shape='PULSE' with 
        os.makedirs(save_dir, exist_ok=True)
        time.sleep(0.1)
        all_transfer_data = tf.measure(numChannels, shapeType, frequency, iterations, bin_size=1, timestamp=f"{timestamp}")

        tf.close_instruments()

        # Assumes we are collecting from two channels only
        # print(all_transfer_data)
        means1 = all_transfer_data[0]
        means2 = all_transfer_data[1]

        ### Saves the data to a csv file

        # Plot & Save data function should have the channel data type information
        df = save_data_to_csv([frequency, means1, means2], label, save_dir, datestamp, data_type)

        ### Directly use the data to obtain the transfer function
        # plot_tf_from_df(df, label, save_dir, timestamp, data_type)

        # Using the built-in method
        # plot_tf(df, label)                                        
        