import os
import sys
import logging
import pyvisa
import re
import numpy as np
import math
import plotly.graph_objects as go
import scipy as sc
from tqdm import tqdm
from datetime import datetime
from typing import Optional
import plotly.io as pio
import pandas as pd

#####
## Things to change
#####
# record_length = 1k
# sampling rate = 5 samples / sec
# acquisition mode = normal
# record time = 200s
# Timebase - 20s / div

sys.set_int_max_str_digits(400000) 

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


def average_reduce(array, factor):
    if isinstance(array, np.ndarray):
        array = array.tolist()  

    reduced_array = []
    i = 0
    while i < len(array):
        chunk = array[i:i + factor]
        if chunk:
            average = sum(chunk) / len(chunk)
            reduced_array.append(average)
        i += factor
    return reduced_array


def damping_func(t, A, l, w, p):
    return A * np.exp(-1 * l * t) * np.cos(w * t - p)


def get_devices():
    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        return resources
    except:
        return ['No devices found!']


class acq:
    def __init__(self, chunkSize = int(1E5), channels=[1,2], mode=['X vs Y'], amp_gain=1,
                 volt=None,
                 acquisition_time=None, sampling_rate=None, record_length=None,
                 noise_period_ms=41.67):
        """
        Initialize the acq Data collector with time control options.

        Args:
            
        """

        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.yk = None
        self.chunkSize = chunkSize
        self.channels = channels
        self.channel_data = {channel: None for channel in channels}
        self.mode = mode
        self.amp_gain = amp_gain
        self.prog = {
            'iteration': 1,
            'prog': 0
        }
        # Parameters that will vary by measurement
        self.volt = volt, # The total voltage applied to the displacement sensor (Needed to compute the distance traveled)
        self.acquisition_time = acquisition_time # Desired acquisition time in seconds
        self.sampling_rate = sampling_rate # Desired sampling rate in Hz
        self.record_length = record_length # Desired number of points to collect 
        self.noise_period_ms = noise_period_ms # Period of noise to filter out in milliseconds

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)


    def open_instruments(self) -> Optional[pyvisa.resources.MessageBasedResource]:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
            
        if self.yokogawaAddress not in resources:
            logging.info(f"Error: Yokogawa oscilloscope not found")
            logging.info("Available resources:", resources)
            return None

        try:
            self.yk = rm.open_resource(self.yokogawaAddress)
            logging.info(f"Successfully opened Yokogawa oscilloscope at {self.yokogawaAddress}")
            # return self.yokogawaInstrument
            
        except pyvisa.errors.VisaIOError as e:
            logging.info(f"Error opening Yokogawa oscilloscope: {e}")
            return None
    


    def _initialize_oscilloscope(self, sampling_rate='5000', time_div='1s', acquisition_time=20):

        # Initialize an instance of the acq object
        
        # Ensure a fresh restart
        self.yk.write(':STOP')
        self.yk.write(':WAVEFORM:FORMAT WORD')
        self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')
        
        # General Timebase settings
        self.yk.write(':TIMebase:TDIV ' + time_div)
        
        # @>-->----- Set the sampling rate
        self.yk.write(':TIMebase:SRATe' + sampling_rate)
        logging.info(f"Finished setting the desired sampling rate: {sampling_rate} Hz")
        print(f"Finished setting the desired time division: {time_div}" )
        self.sampling_rate = extract_number(sampling_rate)

        
        # When only acquisition time is given
        # Round to the nearest valid record length. p.5-30 in the communications manual
        print(f"The acquisition time is currently: {self.acquisition_time} ms")
        print(f"The record length is currently: {self.record_length}")
        if self.acquisition_time is None and self.record_length is None:
            print("Inside the if statement")
            required_points = int(acquisition_time * self.sampling_rate)
            valid_lengths = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000,
             5000000, 10000000, 25000000, 50000000, 100000000]
            closest_length = min(valid_lengths, key=lambda x: abs(x-required_points))
            self.yk.write(f':ACQuire:RLENgth {closest_length}')
            actual_acquisition_time = closest_length/self.sampling_rate
            self.acquisition_time = actual_acquisition_time
            

            print(f"Finished setting record length to {closest_length} points")
            print(f"Requested acquisition time: {acquisition_time}ms")
            print(f"Actual acquisition time: {actual_acquisition_time:.3f}ms")
        

        # Set up acquisition mode to averaging
        ### @>->---
        self.logger.info(f"Noise period in ms: {self.noise_period_ms}")

        self.logger.info(f"Sampling rate: {self.sampling_rate}")

        self.logger.info(f"Chunk size: {self.chunkSize}")


        samples_per_noise_period = int(self.noise_period_ms * self.sampling_rate/ self.chunkSize)
        print(samples_per_noise_period)
        average_count = 2 ** int(np.log2(samples_per_noise_period) + 0.5)
        average_count = max(2, min(65536, average_count))

        self.logger.info(f"Average count: {average_count}")

        self.yk.write(':ACQuire:MODE AVERage')
        self.yk.write(f':ACQuire:AVERage:COUNt {average_count}')
        self.yk.write(':ACQuire:AVERage:EWEight 16')

        
        # Initialize each of the channels
        for channel in self.channels:
            
            self.yk.write(':WAVeform:TRACE ' + str(channel))
            result = self.yk.query('WAVEFORM:RECord? MINimum')
            min_record = int(extract_number(result))
            self.yk.write(':WAVeform:RECord ' + str(min_record))
        
    
    def initialize_instruments(self, sampling_rate, time_div, acquisition_time):
        self._initialize_oscilloscope(sampling_rate, time_div, acquisition_time)


    # Verify that the sampling rate matches the expected value.
    def _verify_sampling_rate(self):
        
        actual_rate = extract_number(self.yk.query(':TIMebase:SRATe?'))
        
        if self.sampling_rate and actual_rate != self.sampling_rate:
            self.logger.warning(
                f"Sampling rate mismatch - Expected: {self.sampling_rate} Hz, "
                f"Actual: {actual_rate} Hz"
            )
        return actual_rate

    # Currently, the run() function contains initialization steps and yk.close()
    def run(self):

        # Check that the sampling rate is what was specified, and use the sampling rate in the oscilloscope if it is different
        sampling_rate = self._verify_sampling_rate()
        

        for channel in self.channels:

            self.logger.info(f"Processing channel {channel}")
            length = int(extract_number(self.yk.query(':WAVEFORM:LENGth?')))
            
            data = {}

            print("Defining n")
            n = int(np.floor(length / self.chunkSize))
            t_data = []


            for i in tqdm(range(n + 1)):
                
                m = min(length, (i + 1) * self.chunkSize) - 1

                print("Set the start and end data point in the waveform")
                
                self.yk.write(":WAVEFORM:START {};:WAVEFORM:END {}".format(i * self.chunkSize, m))


                print("query binary values")
                buff = self.yk.query_binary_values(':WAVEFORM:SEND?', datatype='h', container=list)

                t_data.extend(buff)
                self.prog['prog'] = (i + 1) / (n + 1)
                
                # Query the waveform offset
                result = self.yk.query(':WAVEFORM:OFFSET?')
                offset = extract_number(result)

                # Query the waveform range
                result = self.yk.query(':WAVeform:RANGe?')
                w_range = extract_number(result)

                print("THE waveform capture length is: " + str(self.yk.query(":WAVEFORM:CAPTure:LENGth?")))
                # Convert the raw waveform range data to t_data
                t_data = w_range * np.array(t_data, dtype=float) * 10.0 / 24000.0 + offset  # some random bullshit formula in the communication manual

                #if 'time domain' in self.mode or 'X vs Y' in self.mode or 'resonance' in self.mode:
                if any(mode in self.mode for mode in ['time domain', 'X vs Y', 'resonance']):
                    data['t_volt'] = t_data

                    # if 'time domain' in self.mode or 'resonance' in self.mode:
                    if any(mode in self.mode for mode in ['time domain', 'resonance']):
                        data['t'] = np.arange(len(t_data)) / sampling_rate

                #if 'frequency domain' in self.mode or 'resonance' in self.mode:
                if any(mode in self.mode for mode in ['frequency domain', 'resonance']):    
                    
                    data['t_acc'] = (9.81 / 10) * t_data / self.amp_gain
                    if 'frequency domain' in self.mode:
                        freq, psd_acc = sc.signal.periodogram(data['t_acc'], fs=sampling_rate)
                            # sc.signal.welch(data['t_acc'],fs=sampling_rate,nperseg=sampling_rate,window='blackman',noverlap=0)
                        freq = freq[1:-1]
                        psd_acc = psd_acc[1:-1]

                        data['f'] = freq
                        data['psd_acc'] = psd_acc
                        data['psd_pos'] = psd_acc / freq ** 2

                self.channel_data[channel] = data
            self.prog['iteration'] += 1
            print('Channel completed')

        self.yk.close()

    # Assumes that the oscilloscope inputs are known and fixed
    def save(self, label, save_dir):
        df = pd.DataFrame({key: value['t_volt'] for key, value in self.channel_data.items()})

        # Relabel the columns, based on the mode
        if 'X vs Y' in self.mode:
            # Specify the column renaming
            new_column_names = {1: 'channel1 - load cell', 2: 'channel2 - disp. sensor'}

            # Rename the columns
            df = df.rename(columns=new_column_names)


        # Generate a timestamp for the file name
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        file_name = f"{timestamp}_{label}.csv"

        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, file_name), index=False)

    def plot(self):
        figs = []
        if 'time domain' in self.mode:
            for i, (key, data) in enumerate(self.channel_data.items()):
                # Time Domain Plot
                t = data['t']
                t_data = data['t_volt']
                fig = go.Figure(data=go.Scatter(
                    x=t,
                    y=t_data,
                    mode='lines'))
                fig.update_layout(
                    title_text='Time Domain Signal, Channel ' + str(key),
                    xaxis_title='Time (s)',
                    yaxis_title='Voltage (V)')
                figs.append(fig)

        if 'frequency domain' in self.mode:
            for i, (key, data) in enumerate(self.channel_data.items()):
                # Frequency Domain Plot
                f = data['f']
                psd_data = data['psd_pos']

                x_lim = [0, 6E2]
                y_lim = []
                i_xlim = np.argmax(f > 1E3)
                y_lim.append(1E-1 * np.min(psd_data[0:i_xlim]))
                y_lim.append(1E1 * np.max(psd_data[0:i_xlim]))
                log_y_lim = [math.log10(bound) for bound in y_lim]

                fig = go.Figure(data=go.Scatter(
                    x=f,
                    y=psd_data,
                    mode='lines'))
                fig.update_layout(
                    title='Frequency Domain Signal, Channel ' + str(key),
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Position PSD (m/√Hz)',
                    xaxis_range=x_lim,
                    #yaxis_range=log_y_lim
                )
                fig.update_yaxes(type="log")
                figs.append(fig)

        if 'resonance' in self.mode:
            for i, (key, data) in enumerate(self.channel_data.items()):
                # Time Domain Plot
                t = data['t']
                t_data = data['t_acc']
                popt, pcov = sc.optimize.curve_fit(damping_func, t, t_data)
                [A, l, w, p] = popt
                fn = w / (2 * math.pi)
                zeta = l / np.sqrt(l ** 2 + w ** 2)
                delta = 2 * 3.1416 * zeta / np.sqrt(1 - zeta ** 2)
                t_fit = damping_func(t, A, l, w, p)
                data_trace = go.Scatter(
                    x=t,
                    y=t_data,
                    mode='lines')
                fit_trace = go.Scatter(
                    x=t,
                    y=t_fit,
                    mode='lines')
                fig = go.Figure(data=[data_trace, fit_trace])
                titlestring = 'Resonance Measurement of QZS Flexure Component, Channel ' + str(key) + '<br>'
                #titlestring += r'fit to $y = A \exp{-\lambda t} \cos{\omega t - \varphi}$' + '<br>'
                titlestring += r"A={:.5g}, l={:.5g}, w={:.5g}, fn={:.5g} p={:.5g}, d={:.5g}".format(A, l, w, fn, p, delta)
                fig.update_layout(
                    title_text=titlestring,
                    xaxis_title='Time (s)',
                    yaxis_title='Acceleration (m/s^2)')
                figs.append(fig)

        if 'X vs Y' in self.mode:

            # Get the distance data and process it to get the correct values
            # x = -7.766 * np.array(self.channel_data[self.channels[0]]['t_volt'])
            print(str(self.volt[0]))
            x = -38.1/self.volt[0] * np.array(self.channel_data[self.channels[0]]['t_volt']) # Change from negative to positive
            x = x - np.min(x)
            self.channel_data[self.channels[1]]['distance (mm)'] = x
            #self.channel_data[self.channels[1]]['force (N)'] = []

            # Get the force data in Newtons and process it to get the correct values
            #y = 4.108 * (np.array(self.channel_data[self.channels[1]]['t_volt']) - 4.547)
            # y = 9.81 * 8.1 * (np.array(self.channel_data[self.channels[1]]['t_volt']) - 5.0715)
            
            # Assuming a linear relation between output voltage range and measurable force range
            y = 4.44822162 * (0.25 * (np.array(self.channel_data[self.channels[1]]['t_volt']) - 1.25))

            ### Grams to Newtons, calibration with the weights
            y = 9.81 * (1.2506 * np.array(self.channel_data[self.channels[1]]['t_volt']) - 0.6525)



            self.channel_data[self.channels[0]]['force (N)'] = y
            #self.channel_data[self.channels[1]]['distance (mm)'] = []

            # Currently averages the data in chunks after collection
            x_reduced = average_reduce(x, 10)
            y_reduced = average_reduce(y, 10)

            fig = go.Figure(data=go.Scatter(
                x=x_reduced,
                y=y_reduced,
                mode='lines+markers'))
            fig.update_layout(
                title_text='Force vs Distance',
                xaxis_title='Distance [mm]',
                yaxis_title='Force [N]')
            figs.append(fig)

        return figs

def save_figures(figures, output_dir='./figures'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, fig in enumerate(figures):
        # Generate a filename based on the figure's title or use a default name
        title = fig.layout.title.text if fig.layout.title.text else f"figure_{i+1}"
        # Remove any characters that are not suitable for filenames
        filename = ''.join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        filename = filename.replace(' ', '_') + '.png'
        
        # Full path for the output file
        filepath = os.path.join(output_dir, filename)
        
        # Save the figure as a PNG file
        pio.write_image(fig, filepath)
        print(f"Saved figure to {filepath}")

### Beginning of Main ###
if __name__ == "__main__":

    save_folder=datetime.now().strftime('%Y%m%d')+str("_flexure_control")
    save_dir=os.path.join(os.path.expanduser("~\\Desktop\SoyeonChoi\QZS"), save_folder)

    # Parameters (The sample rate is not being reflected in the settings, defaults to 1kHz. Why?)
    desired_sample_rate='10000'
    desired_acquisition_time_ms=80000 # milliseconds
    # record_length is not specified
    noise_period_ms=41.67

    force_cal_params = []
    dis_cal_params = []

    print("Test F vs. D acquisition")
    osc=acq(
        channels=[1,2],
        mode=['X vs Y'],
        volt=4.0
    )

    osc.open_instruments()

    # During the initialization, osc object's sampling_rate and other variables will get updated
    osc.initialize_instruments(sampling_rate=desired_sample_rate,
                               time_div='200s',
                               acquisition_time=desired_acquisition_time_ms)
    print("After initialization, the acquisition time is ")
    print(osc.acquisition_time)

    print("Running the measurement")


    try:
        osc.run()
    except Exception as e:
        print(f"Error opening running Yokogawa oscilloscope: {e}")

    print(osc.channel_data)
    figs = osc.plot()
    save_figures(figs)
    osc.save('test_FvsD', save_dir)
    
    # dataframe = acq.save_data_to_csv(save_dir)


    # acq.save_data_to_csv(save_dir)
    # acq.plot_arrays(save_dir)

    # acq.plot(save_dir)







