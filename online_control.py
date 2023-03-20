from psychopy import visual, event, core
from pylsl import StreamInlet, resolve_byprop   # Module to receive EEG data
import numpy as np  
import time
import serial
import utils   # Our own utility functions

send_conmmand = False
if send_conmmand:
    serialPort = "COM6"
    baudRate = 115200
    ser = serial.Serial(serialPort, baudRate, timeout = None) 

width = 600
hight = 600
# width = 2560
# hight = 1440
fresh_rate = 60  # window fresh rate

win = visual.Window(size = [width, hight], units = "pix", fullscr = False, color = [1, 1, 1])

radius = 150
wedge = visual.Pie(win, radius = radius, units = "pix", start = 0, end = 15, fillColor = [-1,-1,-1], ori = 0.0)

text = visual.TextStim(win = win, text = 'Attention Score:', pos = (0, -2 * hight / 8), color = [-1, -1, -1], height = 32)
    
# Search for active LSL streams
print('Looking for an EEG stream...')
streams = resolve_byprop('type', 'EEG', timeout = 2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')

# Set active EEG stream to inlet and apply time correction
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen = 12)
eeg_time_correction = inlet.time_correction()

# Get the stream info and description
info = inlet.info()
description = info.desc()

# Get the sampling frequency
fs = int(info.nominal_srate())

channel_id = [0]
NCHAN = len(channel_id)
BUFFER_LENGTH = 5
EPOCH_LENGTH = 2
STEP_LENGTH = 0.2
FEEDBACK_LENGTH = 1

# Initialize raw EEG data buffer
eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), NCHAN))
eeg_buffer_filtered = np.zeros((int(fs * BUFFER_LENGTH), NCHAN))
eeg_epoch = np.zeros((int(fs * EPOCH_LENGTH), NCHAN))
max_metric_buffer_length = int(FEEDBACK_LENGTH/STEP_LENGTH)
attention = 0
v = 0

theta_buffer = []
alpha_buffer = []
beta_buffer = []
attention_buffer = []

threshold = 2.5
total_time_above_threshold = 0
max_duration_above_threshold = 0
temp_duration_max = 0
freeze_time = 0

def process_metrics_buffer(buffer, value):
    buffer.append(value)
    if len(buffer) > max_metric_buffer_length:
        buffer.pop(0)
        
    # return buffer
    return [round(i,2) for i in buffer]

def get_metrics():
    global eeg_buffer, eeg_buffer_filtered, eeg_epoch
    global theta_buffer, alpha_buffer, beta_buffer
    global attention_buffer, attention, v
    
    # get eeg data
    eeg_data, _ = inlet.pull_chunk(timeout=0.0) 
    # print(np.shape(eeg_data))
    # print(eeg_data)
    
    if len(eeg_data) > 0:
        ch_data = np.array(eeg_data)[:, channel_id]
        # print(ch_data)
        
        # Update EEG buffer with the new data
        eeg_buffer = utils.update_buffer(eeg_buffer, ch_data)
        
        # filter
        for i, _ in enumerate(channel_id):           
            # eeg_pad =  np.pad(eeg_buffer[:, i], 10*fs, 'edge')
            # eeg_buffer_filtered[:, i] = utils.butter_bandpass_filter(eeg_pad, 0.5, 40, fs, notch=50)[10*fs:-10*fs]
            
            # eeg_pad =  np.pad(eeg_buffer[:, i], 10*fs, 'edge')
            # eeg_buffer_filtered[:, i] = utils.butter_sos_filt(eeg_pad, 0.5, 40, fs)[10*fs:-10*fs]
           
            
            eeg_buffer_filtered[:, i] = utils.butter_bandpass_filter(eeg_buffer[:, i], 0.5, 40, fs, notch=50)
            
            # eeg_buffer_filtered[:, i] = utils.butter_sos_filt(eeg_buffer[:, i], 0.5, 40, fs)
            
        # Get newest samples from the buffer
        eeg_epoch = utils.get_last_data(eeg_buffer_filtered, int(EPOCH_LENGTH * fs))
        
        # Compute band powers
        powers_and_metrics = utils.compute_band_powers_and_metrics(eeg_epoch, fs)
        
        theta_buffer = process_metrics_buffer(theta_buffer, powers_and_metrics['powers']['theta'])
        alpha_buffer = process_metrics_buffer(alpha_buffer, powers_and_metrics['powers']['alpha'])
        beta_buffer = process_metrics_buffer(beta_buffer, powers_and_metrics['powers']['beta'])
        
        attention_buffer = process_metrics_buffer(attention_buffer, powers_and_metrics['metrics']['Attention Index 3'])
                         
count = 0
blinks = 0                                            
trial_start_time = time.time()
start_time = time.time()

frameN = 0
while True:
    text.text = 'Attention Score: ' + str(attention)
    text.draw()
    wedge.setOri(wedge.ori + v)
    wedge.draw()
    win.flip()
    
    if frameN % (STEP_LENGTH*fresh_rate) == 0:
        get_metrics()
        if frameN >= EPOCH_LENGTH*fresh_rate:                           
            count +=1
            if count == max_metric_buffer_length:
                current_time = time.time()  
                print("\n#### Trial " + str(int((frameN - EPOCH_LENGTH*fresh_rate) / (STEP_LENGTH*fresh_rate*max_metric_buffer_length))) + " ####")  
                print('duration: {} (s)'.format(round(current_time - trial_start_time, 3)))
                trial_start_time = current_time
               
                print("beta: ", round(np.mean(beta_buffer),2))
                print("alpha: ", round(np.mean(alpha_buffer),2))
                print("theta: ", round(np.mean(theta_buffer),2))
                
                print("beta / (theta + alpha): ", round(np.mean(attention_buffer),2))
                
                # calculate attention index
                attention = round(np.mean(attention_buffer),2)
                print("Attention:", attention)
                
                # detect eye blinks
                # blinks = utils.eye_blink(eeg_buffer_filtered, fs, plot = True)
                blinks = utils.eye_blink(eeg_epoch[-fs:], fs, plot = False)                           
                print("blinks:", blinks)
                                
                if blinks >=2 and freeze_time <= 0:
                    attention = 101
                    print("Attention augment!")
                    freeze_time = 10                
                freeze_time -= 1            
                print("freeze_time:", freeze_time)
                
                # send conmand
                if send_conmmand:
                    ser.write(b'\x00\x11')
                    if attention >= 0 and attention < 0.5:
                        ser.write(b'\x00')
                    elif attention >= 0.5 and attention < 1.0:
                        ser.write(b'\x01')
                    elif attention >= 1.0 and attention < 2.0:
                        ser.write(b'\x02') 
                    elif attention >= 2.0 and attention < 5.0:
                        ser.write(b'\x03') 
                    elif attention >= 5.0 and attention < 10.0:
                        ser.write(b'\x04') 
                    elif attention >= 10.0 and attention < 100.0:
                        ser.write(b'\x05')
                    elif attention >= 100.0:
                        ser.write(b'\x08')
                    ser.write(b'\x0D\x0A')
                    
                v = 4.0 * attention
                count = 0
                
                if attention > threshold:
                    total_time_above_threshold += 1
                    temp_duration_max += 1
                else:
                    temp_duration_max = 0              
                total_duration = time.time() - start_time    
                print("total time above threshold: ", total_time_above_threshold, " /", round(total_duration,2), "(s)")
                
                if max_duration_above_threshold < temp_duration_max:
                    max_duration_above_threshold = temp_duration_max
                print("max duration above threshold: ", max_duration_above_threshold, "(s)")
                
    frameN += 1       
    #handle key presses each frame
    for keys in event.getKeys():
        if keys in ['escape', 'q']:
            if send_conmmand:
                ser.write(b'\x00\x11')
                ser.write(b'\00')
                ser.write(b'\x0D\x0A')                
                ser.close()           
            win.close()
            core.quit()
        
win.close()
core.quit()
