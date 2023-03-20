"""
Send multi-channel EEG streams through LSL.
"""

# Import necessary libralies
from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np
# import time

n_channel = 8
sample_rate = 250
# scale_factor_for_uv = 4.5*1e6/ (24 * ((2 ** 23) - 1))
SCALE_FACTOR_EEG = (4.5*1e6)/24/(2**23-1)      #uV/count

channel_names = ['Fp1', 'unkown1', 'unkown2', 'unkown3', 'unkown4','unkown5', 'unkown6', 'unkown7']   

print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCIEEGtest\n")      
# info_eeg = StreamInfo(name='OpenBCIEEG', type='EEG', channel_count=8, 
#                       nominal_srate=250, channel_format='float32', source_id='OpenBCIEEGtest')
info_eeg = StreamInfo('OpenBCIEEG', 'EEG', n_channel, sample_rate, 'float32', 'OpenBCIEEGtest')

info_eeg.desc().append_child_value("manufacturer", "LSLTestAmp")

eeg_channels = info_eeg.desc().append_child("channels")

for c in channel_names:
    eeg_channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "microvolts") \
                .append_child_value("type", "EEG")
                
# Make an outlet
outlet_eeg = StreamOutlet(info_eeg)

def lsl_streamers(sample):
    
    # Send EEG data and wait for a bit
    outlet_eeg.push_sample(np.array(sample.channels_data) * SCALE_FACTOR_EEG, local_clock())
    
    # print(np.array(sample.channels_data)*SCALE_FACTOR_EEG)
    
    # time.sleep(1/sample_rate)

# Set (daisy = True) when stream 16 channels    
board = OpenBCICyton(port='COM3', daisy=False)

# Begins the LSL stream
board.start_stream(lsl_streamers)     
