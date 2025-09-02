import os
import pint
import logging
import pyvisa
import re
import numpy as np
import math
import plotly.graph_objects as go
import scipy as sc
from collections import namedtuple
from tqdm import tqdm
import time
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

# Globals
ureg = pint.UnitRegistry()


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


# converts time SI string to float
def convert_to_seconds(time_string):
    try:
        int_time = ureg(time_string)
        return int_time.to('second').magnitude
    except pint.errors.UndefinedUnitError:
        raise ValueError('Invalid unit')
    except pint.errors.DimensionalityError:
        raise ValueError('Invalid time string')


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
    def __init__(self, channels=None, mode=['X vs Y'], amp_gain=1,
                 volt=None, load_volt=None,
                 acquisition_time=None, sampling_rate=None, record_length=None,
                 V_o_force = 0.57271308, V_o_disp = 0.200629953):
        """
        Initialize the acq Data collector with time control options.

        Args:
            yokogawaAddress - the usb port to which the yokogawa is connected
            yk - the oscilloscope object
            channels - named tuple that contains the data type and data for each channel
            mode - the type of data collection to perform
            amp_gain - the amplification factor
            prog - the progress of the data collection
        """

        self.yokogawaAddress = 'USB0::0x0B21::0x003F::39314B373135373833::INSTR'
        self.yk = None
        self.channels = channels
       
        self.mode = mode
        self.amp_gain = amp_gain
        # Parameters that will vary by measurement
        self.load_volt = load_volt
        self.volt = volt # The total voltage applied to the displacement sensor (Needed to compute the distance traveled)
        self.V_o_force = V_o_force
        self.V_o_disp = V_o_disp
        self.acquisition_time = acquisition_time # Desired acquisition time in seconds
        self.sampling_rate = sampling_rate # Desired sampling rate in Hz
        self.record_length = record_length # Desired number of points to collect 
        self.timestamp = None
        self.corrected_csv = None # Store the corrected file path

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
            self.yk = rm.open_resource(self.yokogawaAddress, timeout=5000)
            logging.info(f"Successfully opened Yokogawa oscilloscope at {self.yokogawaAddress}")
            # return self.yokogawaInstrument
            
        except pyvisa.errors.VisaIOError as e:
            logging.info(f"Error opening Yokogawa oscilloscope: {e}")
            return None
    


    def _initialize_oscilloscope(self, sampling_rate, time_div, record_length):
        """
        Initialize oscilloscope with specified record length and verify settings
    
        Args:
            sampling_rate: Desired sampling rate string (e.g., '5Hz')
            time_div: Time per division string (e.g., '20s')
            record_length: Desired number of points to collect
        """
        
        # Remote control
        remote_status = self.yk.write(':COMMunicate:REMOTE?')
        self.logger.info(f"Remote status: {remote_status}")
        time.sleep(1)
        self.yk.write(':STOP')
        self.yk.write(':CALIBRATE:MODE OFF')
        self.yk.write(':COMMunicate:REMote ON')
        self.logger.info(f"Remote control enabled")

        # Ensure a fresh restart, and set acquisition mode to normal and waveform format to WORD
        self.yk.write(':STOP')
        cal_mode = self.yk.write(':CALibrate:MODE?')
        self.logger.info(f"Current calibration mode: {cal_mode}")
        self.yk.write(':ACQuire:MODE NORMal')
        self.yk.write(':WAVEFORM:FORMAT WORD')
        self.yk.write(':WAVEFORM:BYTEORDER LSBFIRST')
        

        # Set the time division
        self.yk.write(':TIMebase:TDIV ' + time_div)
        time.sleep(0.5)
        logging.info(f"Finished setting the desired time division: {time_div}" )

        # Set and verify the record length
        self.record_length = record_length
        self.yk.write(':ACQuire:RLENgth ' + str(record_length))
        time.sleep(0.5)
        self.logger.info(f"Finished setting the desired record length: {record_length}")
        
        # Set the sampling rate
        self.yk.write(':TIMebase:SRATE ' + sampling_rate)
        time.sleep(0.5)
        logging.info(f"Finished setting the desired sampling rate: {sampling_rate}")
        self.sampling_rate = extract_number(sampling_rate)
       
        # Initialize each of the channels
        for channel in self.channels:
            
            result = self.yk.query(':WAVEFORM:RECord? MINimum')
            min_record = int(extract_number(result))
            self.yk.write(':WAVeform:RECord ' + str(min_record))
            self.logger.info(f"Minimum record point for channel {channel.port} is: {min_record}")

            # Set the channel probe
            self.yk.write(f':CHANnel{channel.port}:PROBe 1')
            self.logger.info(f"Set the probe setting for channel {channel.port} to 1")
            after_setting = self.yk.query(f':CHANnel{channel.port}:PROBe?')
            self.logger.info(f"The probe setting for channel {channel.port} is now: {after_setting}")

            # Set the channel coupling
            self.yk.write(f':CHANnel{channel.port}:VOLTage:COUPling DC')
            self.logger.info(f"Set the coupling for channel {channel.port} to DC")
            after_coupling = self.yk.query(f':CHANnel{channel.port}:VOLTage:COUPling?')
            self.logger.info(f"The coupling for channel {channel.port} is now: {after_coupling}")




    def initialize_instruments(self, sampling_rate, time_div, record_length):
        self._initialize_oscilloscope(sampling_rate, time_div, record_length)


    # Verify that the sampling rate matches the expected value.
    def _verify_sampling_rate(self):

        time.sleep(1)  # Allow update before querying
        actual_rate = extract_number(self.yk.query(':TIMebase:SRATe?'))
        self.logger.info(f"Verified Sampling Rate: {actual_rate} Hz")

        if self.sampling_rate and actual_rate != self.sampling_rate:
            self.logger.warning(
                f"Sampling rate mismatch - Expected: {self.sampling_rate} Hz, "
                f"Actual: {actual_rate} Hz"
            )
        return actual_rate
    
    # Verify record length
    def _verify_record_length(self):
         
        time.sleep(1)  # Allow update before querying
        actual_length = extract_number(self.yk.query(':ACQuire:RLENgth?'))
        self.logger.info(f"Verified Record Length: {actual_length} points")
            
        if self.record_length and actual_length != self.record_length:
                self.logger.warning(
                    f"Record length mismatch - Expected: {self.record_length} points, "
                    f"Actual: {actual_length} points")
        return actual_length


    # Generate a consistent timestamp format for all saving operations
    def _generate_timestamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _verify_format_settings(self):
        format_type = self.yk.query(':WAVeform:FORMat?')
        byte_order = self.yk.query(':WAVeform:BYTeorder?')
        sign = self.yk.query(':WAVeform:SIGN?')

        self.logger.info(f"Waveform format: {format_type}")
        self.logger.info(f"Byte order: {byte_order}")
        self.logger.info(f"Sign setting: {sign}")

        # If sign = 1, then datatype for query_binary_values should be 'h'

    
    # Currently, the run() function contains initialization steps and yk.close()
    def run(self):
        # Check sampling rate & settings
        sampling_rate = self._verify_sampling_rate()
        self.record_length = self._verify_record_length()
        self._verify_format_settings()

        acquisition_time = self.record_length / sampling_rate
        self.logger.info(f"Oscilloscope will be acquiring data for: {acquisition_time:.2f} seconds")
        self.yk.write(':START')
        self.yk.write(':BEEP')
        start_time = time.time()
        time.sleep(acquisition_time)
        self.yk.write(':STOP')
        self.yk.write(':BEEP')
        self.logger.info(f"Oscilloscope acquired data for ~{time.time()-start_time:.2f} s")

        # Ensure both channels are displayed
        self.yk.write(':CHANnel1:DISPlay ON')
        self.yk.write(':CHANnel2:DISPlay ON')

        # Dict to collect columns for CSV
        data_raw = {}

        for channel in self.channels:
            self.logger.info(f"Gathering data from channel {channel.port}-{channel.data_type}")
            self.yk.write(f':WAVeform:TRACe {channel.port}')

            # Expected length
            self.logger.info(f"Expected waveform length is: {self.record_length}")
            acquired_waveform_length = int(extract_number(self.yk.query(':ACQuire:RLENgth?')))
            self.logger.info(f"The acquired waveform length is: {acquired_waveform_length}")

            # Full range for this channel
            self.yk.write(f":WAVEFORM:START 0;:WAVEFORM:END {acquired_waveform_length - 1}")

            # Retrieve the binary waveform data, h=signed short
            t_data_raw = self.yk.query_binary_values('WAVEFORM:SEND?', datatype='h', container=list)
            self.logger.info(f"Raw data first 5 values for channel {channel.port}: {t_data_raw[:5]}")
            self.logger.info(f"Raw data last 5 values for channel {channel.port}: {t_data_raw[-5:]}")
            self.logger.info(f"Waveform data sent to the computer")

            # Query the waveform parameters
            waveform_offset = self.yk.query(':WAVEFORM:OFFSET?')
            offset = extract_number(waveform_offset)

            waveform_range = self.yk.query(':WAVeform:RANGe?')
            w_range = extract_number(waveform_range)

            # Convert raw codes to volts
            t_data_volt = (w_range * np.array(t_data_raw, dtype=float) * 10.0) / 24000.0 + offset
            self.logger.info(f"Converted data first 5 values for channel {channel.port}: {t_data_volt[:5]}")

            # Store converted voltages in the CSV data dict
            data_raw[f"{channel.port}_{channel.data_type}"] = t_data_volt


            # Build per-channel dict for later processing
            data = {}
            if any(mode in self.mode for mode in ['time domain', 'X vs Y', 'resonance']):
                data['t_volt'] = t_data_volt
                if any(mode in self.mode for mode in ['time domain', 'resonance']):
                    data['t'] = np.arange(len(t_data_volt)) / sampling_rate

            if any(mode in self.mode for mode in ['frequency domain', 'resonance']):
                data['t_acc'] = (9.81 / 10) * t_data_volt / self.amp_gain
                if 'frequency domain' in self.mode:
                    freq, psd_acc = sc.signal.periodogram(data['t_acc'], fs=sampling_rate)
                    freq = freq[1:-1]; psd_acc = psd_acc[1:-1]
                    data['f'] = freq
                    data['psd_acc'] = psd_acc
                    data['psd_pos'] = psd_acc / freq**2

            # Update the channel’s data container
            self.channels = [
                f._replace(data=data) if f.port == channel.port else f
                for f in self.channels
            ]

        self.yk.write(':COMMunicate:REMote OFF')
        self.yk.close()

        # Harmonize lengths just in case
        min_len = min(len(np.asarray(v)) for v in data_raw.values())
        for k in list(data_raw.keys()):
            arr = np.asarray(data_raw[k])
            if len(arr) != min_len:
                self.logger.warning(f"Column {k} length {len(arr)} != {min_len}; trimming.")
                data_raw[k] = arr[:min_len]
            else:
                data_raw[k] = arr

        self.timestamp = self._generate_timestamp()
        t_data_raw_df = pd.DataFrame(data_raw)

        self.timestamp = self._generate_timestamp()
        raw_df = pd.DataFrame(t_data_raw_df)
            
        raw_df.to_csv(os.path.join(save_dir, f"{self.timestamp}_raw_data.csv"), index=False)
        
        # Print first few rows to verify
        print(raw_df.head())

    def save(self, save_dir):

        print("Beginning save")

                # Save corrected force vs displacement CSV if both channels are present
        try:
            force_channel = next(ch for ch in self.channels if ch.data_type == 'force')
            disp_channel = next(ch for ch in self.channels if ch.data_type == 'displacement')

            force_voltage = np.array(force_channel.data['t_volt'])
            disp_voltage = np.array(disp_channel.data['t_volt'])

            # Convert to N and mm
            force_measured = 9.81 * (1.2066 * (force_voltage - 0.6936)) # 9.81 * (1.2506 * force_voltage - 0.6525) # 
            a = 38.1 / (2e-5 - self.V_o_disp)
            displacement_mm = a * (disp_voltage - self.V_o_disp) # Older version: -38.1 / self.volt * disp_voltage
            sensor_compression_mm = displacement_mm
            
            # Compute spring force
            k_sensor = 0.101  # N/mm, user-defined
            spring_force = k_sensor * sensor_compression_mm

            # Corrected force
            corrected_force = force_measured - spring_force

            df_force_corrected = pd.DataFrame({
                'measured_force_N': force_measured,
                'displacement_mm': displacement_mm,
                'sensor_compression_mm': sensor_compression_mm,
                'sensor_spring_force_N': spring_force,
                'corrected_force_N': corrected_force
            })

            # Save to CSV
            corr_csv_name = f"{self.timestamp}_force_corrected.csv"
            self.corrected_csv = os.path.join(save_dir, corr_csv_name)
            df_force_corrected.to_csv(self.corrected_csv, index=False)
            print(f"Saved corrected force data to {corr_csv_name}")

        except StopIteration:
            print("Either force or displacement channel missing. Skipping correction CSV.")


    def plot(self, avg_factor=50):
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

            df = None
            """Get force and displacement data from the corrected csv file"""
            if self.corrected_csv and os.path.isfile(self.corrected_csv):
                df = pd.read_csv(self.corrected_csv)
            else:
                print("Warning: corrected CSV not found. Skipping plot.")
                return figs
            
            x = df['displacement_mm'].to_numpy()
            y = df['corrected_force_N'].to_numpy()

            # Average reduction
            x_reduced = average_reduce(x, avg_factor)
            # x_reduced = x_reduced - np.min(x_reduced)  # Zero the minimum displacement
            
            y_reduced = average_reduce(y, avg_factor)
            # y_reduced = y_reduced - np.min(y_reduced)  # Zero the minimum force
            
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

    avg_factor = 50
    flexure_type = f'sec8.88_0rot_belleville_noOring_{avg_factor}pts'
    save_folder=datetime.now().strftime('%Y%m%d')+str("_")+flexure_type
    save_dir=os.path.join(os.path.expanduser("~\\Desktop\SoyeonChoi\QZS\FvsD_August13"), save_folder)

    os.makedirs(save_dir, exist_ok=True)

    # Parameters (The sample rate is not being reflected in the settings, defaults to 1kHz. Why?)
    desired_sample_rate='100Hz' # Try 5 samples / sec
    
    # desired_acquisition_time_ms=80000 # milliseconds
    desired_record_length = 10000 # 250s if record length is 20000 and desired_sample_rate is 100Hz
    noise_period_ms=41.67
    my_channels = [
        Channel(port=1, data_type='force', data=[]),
        Channel(port=2, data_type='displacement', data=[])
        ]
    
    print("Beginning of main")
    osc=acq(
        channels=my_channels,
        mode=['X vs Y'],
        volt=0.2,
        load_volt=5.0,
        record_length=desired_record_length,
        sampling_rate=100
    )

    print("Open instruments")
    osc.open_instruments()

    # During the initialization, osc object's sampling_rate and other variables will get updated

    print("Initializing the instruments")
    try:
        osc.initialize_instruments(sampling_rate=desired_sample_rate,
                               time_div='100s',
                               record_length=desired_record_length)
    except Exception as e:
        print(f"Error initializing Yokogawa oscilloscope: {e}")
        
    print("Running the measurement")


    try:
        osc.run()
    except Exception as e:
        print(f"Error running Yokogawa oscilloscope: {e}")

    
    try:
        print('Saving the data')
        osc.save(save_dir)
    except Exception as e:
        print(f"Error saving the data: {e}")

    try:
        print('Plotting and saving the graph')
        figs = osc.plot(avg_factor=avg_factor)
        osc.save_figures(figs, save_dir)
    except Exception as e:
        print(f"Error plotting and saving the data: {e}")