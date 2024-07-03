import pyvisa
import re
import numpy as np
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy as sc
import time
from tqdm import tqdm
import pint
from datetime import datetime
import os
import pandas as pd

# Globals
ureg = pint.UnitRegistry()

# converts time SI string to float
def convert_to_seconds(time_string):
    try:
        int_time = ureg(time_string)
        return int_time.to('second').magnitude
    except pint.errors.UndefinedUnitError:
        raise ValueError('Invalid unit')
    except pint.errors.DimensionalityError:
        raise ValueError('Invalid time string')
    

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
        array = array.tolist()  # Convert NumPy array to Python list

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
        return rm, resources
    except:
        return ['No devices found!']


class acq:
    def __init__(self):
        self.prog = {}
        self.chunkSize = int(1E5) 
        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.channels = [1,2,3] # By default, channel 1
        self.channel_data = {}
        self.mode = ['X vs Y'] # ['time domain'] # By default, collects data in the time doumain
        self.amp_gain = 1 # By default, amp_gain is set to 1

    def __initialize_yokogawa(self, sample_rate, time_div):

        self.yk.write(':STOP')
        self.yk.write(':CALIBRATE:MODE OFF')

        # General Timebase settings
        self.yk.write(':TIMebase:TDIV ' + '1s')
        self.yk.write(':TIMebase:SRATe' + sample_rate)
        self.yk.write(':TIMebase:TDIV ' + time_div)

        ##### *!* 
        # Waveform setting initialization 
        for c in self.channels:
            self.yk.write(':WAVeform:TRACE ' + str(c))
            result = self.yk.query('WAVEFORM:RECord? MINimum')
            min_record = int(extract_number(result))
            self.yk.write(':WAVeform:RECord ' + str(min_record))
            print("min_record: " + str(min_record))
            self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')
            self.yk.write(':WAVeform:FORMat WORD')

        print("Finished writing the settings to YK")


    def initialize_instruments(self, sample_rate='10kHz', time_div='200ms'):
        self.__sample_rate = sample_rate
        self.__time_div = time_div
        self.__initialize_yokogawa(sample_rate=self.__sample_rate, time_div=self.__time_div)
        
    
    def open_instruments(self):
        rm, resources = get_devices()
        
        if self.yokogawaAddress in resources:
            self.yk = rm.open_resource(self.yokogawaAddress)
            print("Yokogawa opened!")


    def run(self):

        flag = True
        
        self.channel_data = {}
        
        self.prog = {
            'iteration': 1,
            'prog': 0
        }

        time.sleep(0.5)

        ### Refresh to ensure that we're getting the most current data
        self.yk.write(':TIMebase:TDIV ' + '1s')
        self.yk.write(':TIMebase:SRATE ' + self.__sample_rate)
        self.yk.write(':TIMebase:TDIV ' + self.__time_div)

        ##### Yokogawa waveform capture sequence
        self.yk.write(':START')
        
        print("time division: " + self.__time_div)

        ### Yk is on & measuring for the specified time_div
        if convert_to_seconds(self.__time_div) < 0.5:
            time.sleep(12 * convert_to_seconds(self.__time_div))
        else:
            time.sleep(11 * convert_to_seconds(self.__time_div))
        self.yk.write(':STOP')   

                
        for channel in self.channels:
            
            self.channel_data[channel] = None

            if flag:
                self.yk.write(':WAVeform:TRACE ' + str(channel)) # Sets the waveform that WAVeform commands will be applied to

                # Query information needed to collect the waveform data 
                result = self.yk.query(':WAVEFORM:LENGth?') # Queries the total number of data points in the waveform specified by the WAVeform:TRACe command
                length = int(extract_number(result)) # length = total number of data points in the waveform
                print("waveform length: "+str(length))

                result = self.yk.query(':WAVeform:SRATe?')  # Get sample rate
                sampling_rate = extract_number(result)
                print("waveform sample rate: "+str(sampling_rate))

                # Dictionary to save all the data to. One dictionary for each channel
                data = {}

                n = int(np.floor(length / self.chunkSize))
                t_data = [] # bit_data in tf.py and tf_copy.py

                # Timebar
                for i in tqdm(range(n + 1)):
                    m = min(length, (i + 1) * self.chunkSize) - 1

                    # timebar settings
                    self.yk.write(":WAVEFORM:START {};:WAVEFORM:END {}".format(i * self.chunkSize, m))
                    buff = self.yk.query_binary_values(':WAVEFORM:SEND?', datatype='h', container=list) # Queries the waveform data specified by the :WAVeform:TRACe command (the main waveform data)
                    t_data.extend(buff)
                    self.prog['prog'] = (i + 1) / (n + 1)

                result = self.yk.query(':WAVEFORM:OFFSET?')
                offset = extract_number(result)

                result = self.yk.query(':WAVeform:RANGe?')
                w_range = extract_number(result) # The measurement range used to convert the waveform data specified by the :WAVeform:TRACe command to physical values.

                # Convert the voltage data from the oscilloscope
                t_data = w_range * np.array(t_data) * 10 / 24000 + offset  # some random bullshit formula in the communication manual


                if 'time domain' or 'X vs Y' or 'resonance' in self.mode:
                    data['t_volt'] = t_data
                    if 'time domain' or 'resonance' in self.mode:
                        data['t'] = np.arange(len(t_data)) / sampling_rate

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

            self.prog['iteration'] += 1
            print('Channel completed')

        self.yk.close()
        return

    # Saves data to a .csv file
    def save_data_to_csv(self, save_dir, label='FvsD'):

        df = pd.DataFrame()

        if 'X vs Y' in self.mode:
            # Create a DataFrame using the data dictionary
            # df = pd.DataFrame.from_dict(data_dict)
            # channel_data is a dictionary containing key-value pairs

            # Force data 
            df['uncalibrated_force']=self.channel_data[self.channels[1]]['t_volt']

            # Displacement data
            df['uncalibrated_displacement']=self.channel_data[self.channels[2]]['t_volt']

            
        else:
            df = pd.DataFrame.from_dict(self.channel_data)

        # Pirnt the data to check that it is what we except
        
        # Debug
        print(df) # Channel 1 is just 
        
        # Create a temporary directory to save the file
        # temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(save_dir, exist_ok=True)

        # Generate a timestamp for the file name
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        file_name = f"{timestamp}_{label}.csv"

        # Save the DataFrame to a CSV file in the temporary directory
        save_file_path = os.path.join(save_dir, file_name)
        df.to_csv(save_file_path, index=False)
        print(f"File saved to: {save_file_path}")
        return df

    def plot_arrays(self, save_dir):
        
        if 'X vs Y' in self.mode:
            # Force data
            # x = np.array(self.channel_data[1]['t_volt'])

            # Displacement data
            # y = np.array(self.channel_data[2]['t_volt'])
            x_raw = np.array(self.channel_data[1]['t_volt'])
            x1 = -7.766 * x_raw
            x1 = x1 - np.min(x1)

            y_raw = np.array(self.channel_data[2]['t_volt']) 
            y1 = 4.108 * y_raw - 4.547

            plt.plot(x1,y1,'og-')
            plt.title('Force vs. Displacement')
            plt.xlabel('Displacement [mm]')
            plt.ylabel('Force ($m/s^2$)')
            plt.savefig(os.path.join(save_dir, datetime.now().strftime('%Y%m%d%H%M') +"_f_vs.d.png"))
            print("Figure successfully saved!")
            plt.show()
        else:
            print("Nothing to plot")


    def plot_goFigure(self, saveDir):
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

                x_lim = [0, 10E2]
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
                #fig.update_xaxes(type="log")
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

            # Unknown constants: distance 
            # channels[0] = channel 2, load cell
            x = -7.766 * np.array(self.channel_data[self.channels[0]]['t_volt'])
            x = x - np.min(x)
            self.channel_data[self.channels[0]]['distance (mm)'] = x
            self.channel_data[self.channels[0]]['force (N)'] = []

            y = 4.108 * (np.array(self.channel_data[self.channels[1]]['t_volt']) - 4.547)
            self.channel_data[self.channels[1]]['force (N)'] = y
            self.channel_data[self.channels[1]]['distance (mm)'] = []

            x_reduced = average_reduce(x, 100)
            y_reduced = average_reduce(y, 100)
            fig = go.Figure(data=go.Scatter(
                x=x_reduced,
                y=y_reduced,
                mode='lines'))
            fig.update_layout(
                title_text='Force vs Distance',
                xaxis_title='Distance (mm)',
                yaxis_title='Force (N)')
            figs.append(fig)
        
        
        fig.write_image(os.path.join(saveDir, "new_image.png"))
        return fig


### Beginning of Main ###
if __name__ == "__main__":

    save_folder=datetime.now().strftime('%Y%m%d')+str("original_flexure")
    save_dir=os.path.join(os.path.expanduser("~\\Desktop\SoyeonChoi"), save_folder)

    # Parameters
    sample_rate='10kHz'

    force_cal_params = []
    dis_cal_params = []

    print("Test F vs. D acquisition")
    acq=acq()
    acq.open_instruments()
    acq.initialize_instruments(sample_rate='10kHz', time_div='2s')
    acq.run()
    print(acq.channel_data)
    # dataframe = acq.save_data_to_csv(save_dir)


    acq.save_data_to_csv(save_dir)
    acq.plot_arrays(save_dir)

    # acq.plot(save_dir)



