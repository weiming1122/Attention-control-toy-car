import numpy as np
from scipy.signal import butter, filtfilt, lfilter, sosfilt, welch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, SpatialDropout1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization

def butter_sos_filt(data, low, high, fs, order=2):
    sos_p = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    sos_s = butter(order, [48, 52], btype='bandstop', fs=fs, output='sos') 
    yf = sosfilt(sos_p, data)
    yf = sosfilt(sos_s, yf)   
    
    return yf

def butter_bandstop(lowpass, highpass, fs, order=4):
    '''
    Create a Butterworth band stop filter

    Parameters
    ==========
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.

    Returns
    ======
    b, a: ndarray, ndarray
    Numerator (b) and denominator (a) polynomials of the IIR filter. 
    '''
    nyq = 0.5 * fs
    low =  lowpass / nyq
    high = highpass / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=4):
    '''
    Create a Butterworth band pass filter

    Parameters
    ==========
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.

    Returns
    ======
    b, a: ndarray, ndarray
    Numerator (b) and denominator (a) polynomials of the IIR filter. 
    '''
    nyq = 0.5 * fs
    low =  lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, notch=50):
    '''
    Filter data using a bandpass filter.

    Parameters
    ==========
    data: ndarray, data to be filtered.
    lowcut: float, low bound of the band pass.
    highcut: float, high bound of the band pass.
    fs: int, sampling frequency.
    order: int, filter order.
    notch: int 0 or 50/60, whether or not applying a notch filter.
    Returns
    ======
    y: ndarray, filtered data. 
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    
    if notch > 0:
         b, a = butter_bandstop(notch-1, notch+1, fs)
         y = filtfilt(b, a, y)
        
    return y

def update_buffer(data_buffer, new_data):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer

def compute_single_band_power(powers, ind, dx):
    power = np.mean(np.trapz(powers[ind, :], dx=dx, axis=0))
    
    return power

def compute_band_powers_and_metrics(eegdata, fs):
    """Extract the features (band powers) from the EEG.
    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata
    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # get EEG shape
    _, nbCh = eegdata.shape
    nfft = 256
    power_spectrum_density=np.zeros((nfft//2+1, nbCh))  
    for i in range(nbCh):
        f, power_spectrum_density[:, i] = welch(eegdata[:,i], fs=fs, nperseg=128, nfft=nfft)
        
    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    dx = f[1]-f[0]
    ind_delta, = np.where(f < 4)
    delta = compute_single_band_power(power_spectrum_density, ind_delta, dx)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f < 8))
    theta = compute_single_band_power(power_spectrum_density, ind_theta, dx)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f < 12))
    alpha = compute_single_band_power(power_spectrum_density, ind_alpha, dx)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    beta = compute_single_band_power(power_spectrum_density, ind_beta, dx)
    # Gamma 35-45
    ind_gamma, = np.where((f >= 35) & (f < 45))
    gamma = compute_single_band_power(power_spectrum_density, ind_gamma, dx)
    
    # Low Beta 12-20
    ind_low_beta, = np.where((f >= 12) & (f < 20))
    low_beta = compute_single_band_power(power_spectrum_density, ind_low_beta, dx)
    
    # Attention Index
    attention_index_1 = beta / theta
    attention_index_2 = beta / alpha
    attention_index_3 = beta / (theta + alpha)    
    
    powers_dict = {'delta':delta, 'theta': theta, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'low_beta': low_beta}
    metrics_dict = {'Attention Index 1': attention_index_1,
                    'Attention Index 2': attention_index_2,
                    'Attention Index 3': attention_index_3}

    return {'powers': powers_dict, 'metrics':metrics_dict}

def multi_windowed_sdw(data, window_list):
    '''
    Cacluate multi-resolution sdw.
    '''
    deris = data[1:] - data[0:-1]
    m_sdw = []
    for w in window_list:
        sdw = []
        for d in range(0, len(deris)-w):
            sdw.append(np.sum(deris[d:d+w]))
        m_sdw.append(sdw)
    return m_sdw

def eye_blink(eeg, fs, plot = False):
    eeg = np.reshape(eeg, -1)
    m_sdw = multi_windowed_sdw(eeg, [10])
    peaks, _ = find_peaks(m_sdw[0], height=60, distance=50)
    # peaks, _ = find_peaks(m_sdw[0], height=15, distance=10)
     
    if plot:
        time = np.linspace(0, len(eeg)/fs, len(eeg))
        plt.plot(time, eeg)
        axes = plt.gca()
        axes.set_ylim([-150,150])
        plt.plot(time[peaks], [eeg[p] for p in peaks], "x")
        plt.title("Last one second EEG")
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude(V)")
        plt.show()
    
    return len(peaks)

def read_txt(eeg):
    data = []
    for i, e in enumerate(eeg):
        e_split = e.split(',')
        for es in e_split:
            data.append(float(es))
    return data

# def cnn_model_1(nsamples = 1024):
#     model = Sequential()
#     model.add(Conv1D(filters=128, kernel_size=2, activation="relu", padding='same', input_shape=(nsamples, 1)))
#     model.add(BatchNormalization())
#     model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
#     model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Flatten())
#     model.add(Dense(units=100, activation="relu"))
#     model.add(Dense(units=1))
    
#     return model

def cnn_model_1(nsamples = 500):
    model = Sequential()    
    model.add(Conv1D(filters=64, kernel_size=125, activation='relu', input_shape=(nsamples, 1)))
    model.add(Dropout(0.5))    
    model.add(Conv1D(filters=32, kernel_size=25, activation='relu'))    
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1))
    
    model.add(Conv1D(filters=32, kernel_size=5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1))
    
    return model  

def cnn_model_2(nsamples = 500):
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=5, input_shape=(nsamples, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(SpatialDropout1D(0.25))
    
    model.add(Conv1D(filters=64, kernel_size=5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(SpatialDropout1D(0.25))
    
    model.add(Conv1D(filters=64, kernel_size=5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=1))
    
    return model



  