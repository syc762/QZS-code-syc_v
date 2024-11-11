import os
import logging
import pyvisa
import re
import numpy as np
import math
import plotly.graph_objects as go
import scipy as sc
from collections import namedtuple
from tqdm import tqdm
from datetime import datetime
from typing import Optional
import plotly.io as pio
import pandas as pd


#####
## Things to change
#####
# record_length = 1k
# record time = 200s
# Timebase - 20s / div

# Channel named tuple
Channel = namedtuple('Channel', ['port', 'data_type', 'data'])


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
                 noise_period_ms=None):
        """
        Initialize the acq Data collector with time control options.

        Args:
            yokogawaAddress - the usb port to which the yokogawa is connected
            yk - the oscilloscope object
            chunkSize - increment size for progress bar update
            channels - named tuple that contains the data type and data for each channel
            mode - the type of data collection to perform
            amp_gain - the amplification factor
            prog - the progress of the data collection
        """

        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.yk = None
        self.chunkSize = chunkSize
        self.channels = [
            Channel(port=1, data_type='force', data={}),
            Channel(port=2, data_type='displacement', data={})
        ]
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
        self.timestamp = None

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
    


    def _initialize_oscilloscope(self, sampling_rate, time_div, record_length):

        # Initialize an instance of the acq object
        
        # Ensure a fresh restart
        self.yk.write(':STOP')
        self.yk.write(':WAVEFORM:FORMAT WORD')
        self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')
        
        ###### General Timebase settings
        # Set the time division
        self.yk.write(':TIMebase:TDIV ' + time_div)
        logging.info(f"Finished setting the desired time division: {time_div}" )
        
        # Set the sampling rate
        self.yk.write(':TIMebase:SRATe ' + sampling_rate)
        logging.info(f"Finished setting the desired sampling rate: {sampling_rate}")
        self.sampling_rate = extract_number(sampling_rate)

        # Set the record length
        self.yk.write(':WAVeform:RECordlength ' + str(record_length))
       

        # Set acquisition mode
        self.yk.write(':ACQuire:MODE NORMAL')


        self.logger.info(f"Noise period in ms: {self.noise_period_ms}")

        self.logger.info(f"Chunk size: {self.chunkSize}")

        
        # Initialize each of the channels
        for channel in self.channels:
            
            self.yk.write(':WAVeform:TRACE ' + str(channel.port))
            result = self.yk.query('WAVEFORM:RECord? MINimum')
            min_record = int(extract_number(result))
            self.yk.write(':WAVeform:RECord ' + str(min_record))
        
    
    def initialize_instruments(self, sampling_rate, time_div, record_length):
        self._initialize_oscilloscope(sampling_rate, time_div, record_length)


    # Verify that the sampling rate matches the expected value.
    def _verify_sampling_rate(self):
        
        actual_rate = extract_number(self.yk.query(':TIMebase:SRATe?'))
        
        if self.sampling_rate and actual_rate != self.sampling_rate:
            self.logger.warning(
                f"Sampling rate mismatch - Expected: {self.sampling_rate} Hz, "
                f"Actual: {actual_rate} Hz"
            )
        return actual_rate

    # Generate a consistent timestamp format for all saving operations
    def _generate_timestamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    # Currently, the run() function contains initialization steps and yk.close()
    def run(self):

        # Check that the sampling rate is what was specified, and use the sampling rate in the oscilloscope if it is different
        sampling_rate = self._verify_sampling_rate()
        
         # Calculate total iterations for all channels
        total_channels = len(self.channels)
        length = int(extract_number(self.yk.query(':WAVEFORM:LENGth?')))
        chunks_per_channel = int(np.floor(length / self.chunkSize)) + 1
        total_iterations = total_channels * chunks_per_channel

        # Create a progress bar for the entire operation
        with tdqm(total=total_iterations, desc="Processing Channels") as pbar:
        for channel in self.channels:

            self.logger.info(f"Processing channel {channel.port}-{channel.data_type}")
            length = int(extract_number(self.yk.query(':WAVEFORM:LENGth?')))
            
            data = {}
            n = int(np.floor(length / self.chunkSize))
            t_data = []


            for i in tqdm(range(n + 1)):
                
                m = min(length, (i + 1) * self.chunkSize) - 1

                print("Set the start and end data point in the waveform")
                
                # Capturing each chunk of data
                self.yk.write(":WAVEFORM:START {};:WAVEFORM:END {}".format(i * self.chunkSize, m))
                waveform_srate = self.yk.query(':WAVeform:SRATe?')  # Get sampling rate
                sampling_rate = extract_number(waveform_srate)

                buff = self.yk.query_binary_values(':WAVEFORM:SEND?', datatype='h', container=list)

                t_data.extend(buff)
                self.prog['prog'] = (i + 1) / (n + 1)
                
                # Query the waveform offset
                waveform_offset = self.yk.query(':WAVEFORM:OFFSET?')
                offset = extract_number(waveform_offset)

                # Query the waveform range
                waveform_range = self.yk.query(':WAVeform:RANGe?')
                w_range = extract_number(waveform_range)

                print("THE waveform capture length is: " + str(self.yk.query(":WAVEFORM:LENGth?")))
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

                # Save the collected data to the oscilloscope channel attribute
                
                # Find the channel whose port matches the current port we are on
                # update_channel = next((f for f in self.channels if f.port == channel.port))
                # Using a list comprehension to replace the element
                self.channels = [
                    f._replace(data=data) if f.port == channel.port else f for f in self.channels
                ]
                # Alternatively: 
                # Find the index of the channel to update
                # channel_index = next(i for i, f in enumerate(self.channels) if f.port == channel.port)
                # Update the channel at that index
                # self.channels[channel_index] = self.channels[channel_index]._replace(data=data)
                self.prog['iteration'] += 1

        self.yk.close()
        print("The updated osc object's channel data looks like")
        print(self.channels)

    def save(self, label, save_dir):

        print("Beginning save")
        df = pd.DataFrame()
        print(self.channels)
        
        for channel in self.channels:
            print(f"In for loop, on channel {channel.port}")
            print(channel.data)
            
            # For each key in the data dictionary (like 't_volt')
            for data_key, values in channel.data.items():

                if len(channel.data) == 1:
                    column_name = f"channel{channel.port}_{channel.data_type}"
                else:
                    column_name = f"channel{channel.port}_{channel.data_type}_{data_key}"
            
                logging.info('Setting column name to: ' + column_name)
                # Convert numpy array to pandas series
                df[column_name] = values
        
        print(df)

        # Generate a timestamp for this set of data
        self.timestamp = self._generate_timestamp()
        file_name = f"{self.timestamp}_{label}.csv"

        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, file_name), index=False)

    def plot(self):
        figs = []
        
        if 'time domain' in self.mode:
            for channel in self.channels:
                # Time Domain Plot
                t = channel.data['t']
                t_data = channel.data['t_volt']
                fig = go.Figure(data=go.Scatter(
                    x=t,
                    y=t_data,
                    mode='lines'))
                fig.update_layout(
                    title_text=f'Time Domain Signal, Channel {channel.port} ({channel.data_type})',
                    xaxis_title='Time (s)',
                    yaxis_title='Voltage (V)')
                figs.append(fig)

        if 'frequency domain' in self.mode:
            for channel in self.channels:
                # Frequency Domain Plot
                f = channel.data['f']
                psd_data = channel.data['psd_pos']

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
                    title=f'Frequency Domain Signal, Channel {channel.port} ({channel.data_type})',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Position PSD (m/√Hz)',
                    xaxis_range=x_lim)
                fig.update_yaxes(type="log")
                figs.append(fig)

        if 'resonance' in self.mode:
            for channel in self.channels:
                # Time Domain Plot
                t = channel.data['t']
                t_data = channel.data['t_acc']
                popt, pcov = sc.optimize.curve_fit(damping_func, t, t_data)
                [A, l, w, p] = popt
                fn = w / (2 * math.pi)
                zeta = l / np.sqrt(l ** 2 + w ** 2)
                delta = 2 * 3.1416 * zeta / np.sqrt(1 - zeta ** 2)
                t_fit = damping_func(t, A, l, w, p)
                
                data_trace = go.Scatter(x=t, y=t_data, mode='lines', name='Data')
                fit_trace = go.Scatter(x=t, y=t_fit, mode='lines', name='Fit')
                
                fig = go.Figure(data=[data_trace, fit_trace])
                titlestring = f'Resonance Measurement of QZS Flexure Component, Channel {channel.port} ({channel.data_type})<br>'
                titlestring += f"A={A:.5g}, l={l:.5g}, w={w:.5g}, fn={fn:.5g} p={p:.5g}, d={delta:.5g}"
                
                fig.update_layout(
                    title_text=titlestring,
                    xaxis_title='Time (s)',
                    yaxis_title='Acceleration (m/s^2)')
                figs.append(fig)

        if 'X vs Y' in self.mode:

            """Get force channel data"""
            force_channel = next(ch for ch in self.channels if ch.data_type == 'force')
            force_voltage = np.array(force_channel.data['t_volt'])

            ###### Convert the force data from volts to Newtons
            #### Assuming a linear relation between output voltage range and measurable force range
            # Conversion to Newtons as given by the sensor spec sheet
            y = 4.44822162 * (0.25 * (force_voltage - 1.25))
            # Grams to Newtons, calibration with the weights
            # y = 9.81 * (1.2506 * force_voltage - 0.6525)
            

            """Get displacement data (x-axis)"""
            disp_channel = next(ch for ch in self.channels if ch.data_type == 'displacement')
            disp_voltage = np.array(disp_channel.data['t_volt'])

            ########## Convert the displacement data from volts to mm
            ##### Assuming a linear relation between the voltage range and mechanical travel range
            #### Using the electrical travel distance of 38.1mm
            x = -38.1/self.volt[0] * disp_voltage
            x = x - np.min(x)  # Zero the minimum displacement
            
            # Average reduction
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

        

    def save_figures(self, figures, output_dir='./figures'):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for i, fig in enumerate(figures):
            # Generate a filename based on the figure's title or use a default name
            title = fig.layout.title.text if fig.layout.title.text else f"figure_{i+1}"
            # Remove any characters that are not suitable for filenames
            filename = ''.join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
            filename = filename.replace(' ', '_')
            filename = f"{self.timestamp}_{filename}.png"
            
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
    desired_sample_rate='5Hz' # Try 5 samples / sec
    # desired_acquisition_time_ms=80000 # milliseconds
    record_length = 10000
    noise_period_ms=41.67
    my_channels = [
        Channel(port=1, data_type='force', data=[]),
        Channel(port=2, data_type='displacement', data=[])
        ]
    
    print("Beginning of main")
    osc=acq(
        channels=my_channels,
        mode=['X vs Y'],
        volt=0.4,
        noise_period_ms=noise_period_ms
    )

    osc.open_instruments()

    # During the initialization, osc object's sampling_rate and other variables will get updated
    osc.initialize_instruments(sampling_rate=desired_sample_rate,
                               time_div='20s',
                               record_length=1000)
    print("After initialization, the acquisition time is ")
    print(osc.acquisition_time)

    print("Running the measurement")


    try:
        osc.run()
    except Exception as e:
        print(f"Error opening running Yokogawa oscilloscope: {e}")

    try:
        print('Saving the data')
        osc.save('test_FvsD', save_dir)
    except Exception as e:
        print(f"Error saving the data: {e}")

    figs = osc.plot()
    osc.save_figures(figs)