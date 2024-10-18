import os
import sys
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
    def __init__(self, chunkSize = int(1E5), channels=[1,2], mode=['X vs Y'], amp_gain=1):
        self.prog = {}
        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.yk = None
        self.chunkSize = chunkSize
        self.channels = channels
        self.channel_data = {}
        self.mode = mode
        self.amp_gain = amp_gain

    def open_instruments(self) -> Optional[pyvisa.resources.MessageBasedResource]:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
            
        if self.yokogawaAddress not in resources:
            print(f"Error: Yokogawa oscilloscope not found")
            print("Available resources:", resources)
            return None

        try:
            self.yk = rm.open_resource(self.yokogawaAddress)
            print(f"Successfully opened Yokogawa oscilloscope at {self.yokogawaAddress}")
            # return self.yokogawaInstrument
            
        except pyvisa.errors.VisaIOError as e:
            print(f"Error opening Yokogawa oscilloscope: {e}")
            return None
    


    def _initialize_oscilloscope(self, sample_rate='10k', time_div='2s'):

        # Initialize the instance of the acq object
        self.channel_data = {}
        self.prog = {
            'iteration': 1,
            'prog': 0
        }

        # Yokogawa initialization
        self.yk.write(':STOP')
        self.yk.write(':WAVEFORM:FORMAT WORD')
        self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')
        
        # General Timebase settings
        self.yk.write(':TIMebase:TDIV ' + '1s')
        self.yk.write(':TIMebase:SRATe' + sample_rate)
        # self.yk.write(':TIMebase:TDIV ' + time_div)
        
        for channel in self.channels:
            self.channel_data[channel] = None
            self.yk.write(':WAVeform:TRACE ' + str(channel))
            result = self.yk.query('WAVEFORM:RECord? MINimum')
            min_record = int(extract_number(result))
            self.yk.write(':WAVeform:RECord ' + str(min_record))
        
        # Initialize each of the channels


    
    def initialize_instruments(self, sample_rate, time_div):
        self._initialize_oscilloscope(sample_rate, time_div)


    # Currently, the run() function contains initialization steps and yk.close()
    def run(self):

        flag = True
        self.yk.write(':TIMebase:SRATe' + sample_rate)
        result = self.yk.query(':WAVeform:SRATe?')  # Get sampling rate
        sampling_rate = extract_number(result)

        # Some initialization of yk needed here

        for channel in self.channels:
            if flag:

                print("set length")
                length = int(extract_number(self.yk.query(':WAVEFORM:LENGth?')))
                print("the length of the waveform is:" + str(length))
                print("THE waveform capture length is: " + str(self.yk.query(":WAVEFORM:CAPTure:LENGth?")))

                print("Waveform:start is " + str(self.yk.query(":WAVeform:STARt?")))
                print("Waveform:end is " + str(self.yk.query(":WAVeform:END?")))
                
    
                # Calculate the time
                #time_length = (end - start) / sample_rate
                print("the queried sample rate: " + str(sampling_rate))

            if flag:

                data = {}

                print("Defining n")
                n = int(np.floor(length / self.chunkSize))
                t_data = []

                print("starting the for loop for tqdm")
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
            x = -7.766 * np.array(self.channel_data[self.channels[0]]['t_volt'])
            x = x - np.min(x)
            self.channel_data[self.channels[1]]['distance (mm)'] = x
            #self.channel_data[self.channels[1]]['force (N)'] = []

            # Get the force data and process it to get the correct values
            #y = 4.108 * (np.array(self.channel_data[self.channels[1]]['t_volt']) - 4.547)
            # y = 9.81 * 8.1 * (np.array(self.channel_data[self.channels[1]]['t_volt']) - 5.0715)
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
    sample_rate='10000Hz'

    force_cal_params = []
    dis_cal_params = []

    print("Test F vs. D acquisition")
    acq=acq()
    acq.open_instruments()
    acq.initialize_instruments(sample_rate=sample_rate, time_div='2s')

    print("Running the measurement")


    try:
        acq.run()
    except Exception as e:
        print(f"Error opening running Yokogawa oscilloscope: {e}")

    print(acq.channel_data)
    figs = acq.plot()
    save_figures(figs)
    acq.save('test_FvsD', save_dir)
    
    # dataframe = acq.save_data_to_csv(save_dir)


    # acq.save_data_to_csv(save_dir)
    # acq.plot_arrays(save_dir)

    # acq.plot(save_dir)







