from heapq import nsmallest
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.ndimage.filters as ndif
import tensorflow as tf
from tensorflow import keras
from scipy import signal

# to import the files per my formatting
def import_files(emg_filename, q_filename, old=False):
    # Import from txt files, convert timestamps (recorded in two different formats, hence old)
    if not old:
        emg = np.loadtxt(emg_filename, dtype='O,'+'f8,'*12, delimiter=',', unpack=True, \
            converters={0: lambda d: dt.datetime.strptime(d.decode("utf-8"), '%H:%M:%S.%f').replace(year=2022)})
        q = np.loadtxt(q_filename, dtype='O,'+'f8,'*72, delimiter=',', unpack=True, \
            converters={0: lambda d: dt.datetime.strptime(d.decode("utf-8"), '%H:%M:%S.%f').replace(year=2022)})
    else:
        emg = np.loadtxt(emg_filename, dtype='O,'+'f8,'*12, delimiter=',', unpack=True, \
            converters={0: lambda d: dt.datetime.strptime(d.decode("utf-8"), '%H.%M.%S.%f').replace(year=2022)})
        q = np.loadtxt(q_filename, dtype='O,'+'f8,'*72, delimiter=',', unpack=True, \
            converters={0: lambda d: dt.datetime.strptime(d.decode("utf-8"), '%H:%M:%S.%f').replace(year=2022)})    

    
    # Pull timestamps
    emgTimes = emg[0]
    # Stack EMG data
    emg = np.vstack(emg[1:]).T
    # ...
    qTimes = q[0]
    q = np.vstack(q[1:]).T

    return emg, emgTimes, q, qTimes


# Interpolate the EMG and q data, inherit the timestamps of the series with slower aquisition rate
def unify_timeseries_low(emg, emgTimes, q, qTimes):
    # convert to seconds
    emgSecs = np.array([emgTime.timestamp() for emgTime in emgTimes])
    qSecs = np.array([qTime.timestamp() for qTime in qTimes])\
    
    # Determine the data acquisition rate of each source
    emgDt = np.average(np.diff(emgSecs))
    qDt = np.average(np.diff(qSecs))

    # Later start time dictates start of useful data
    if qTimes[0] > emgTimes[0]:
        clipIndex = next(x for x, time in enumerate(emgTimes) if time > qTimes[0])
        emgTimes = emgTimes[clipIndex:]
        emg = emg[clipIndex:, :]
        emgSecs = emgSecs[clipIndex:]

    elif qTimes[0] < emgTimes[0]:
        clipIndex = next(x for x, time in enumerate(qTimes) if time > emgTimes[0])
        qTimes = qTimes[clipIndex:]
        q = q[clipIndex:, :]
        qSecs = qSecs[clipIndex:]

    # Linear interpolation to match times of hf source to lf
    if emgDt >= qDt:
        qF = [np.interp(emgSecs, qSecs, q[:,i]) for i in range(q.shape[1])]
        qF = np.array(qF).T
        emgF = emg
        tF = emgSecs
        timestampsF = emgTimes
    else:
        qF = q
        emgF = [np.interp(qSecs, emgSecs, emg[:,i]) for i in range(emg.shape[1])]
        emgF = np.array(emgF).T
        tF = qSecs
        timestampsF = qTimes
    
    return qF, emgF, tF, timestampsF


# interpolate the emg and q data to inherit the timestamps of the series with higher aquisition rate
def unify_timeseries_high(emg, emgTimes, q, qTimes, verbose = False):
    # convert to seconds
    emgSecs = np.array([emgTime.timestamp() for emgTime in emgTimes])
    qSecs = np.array([qTime.timestamp() for qTime in qTimes])\
    
    # Determine the data acquisition rate of each source
    emgDt = np.diff(emgSecs)
    emgDt = np.average(emgDt)
    qDt = np.diff(qSecs)
    qDt = np.average(qDt)

    if verbose:
        emgDtVar = np.var(emgDt)
        qDtVar = np.var(qDt)

        print("EMG dt -> frequency: ", 1/emgDt)
        print("EMG dt variance: ", emgDtVar)
        print("Q dt -> freqency: ", 1/qDt)
        print("Q dt variance: ", qDtVar)

    # Later start time dictates start of useful data
    if qTimes[0] > emgTimes[0]:
        clipIndex = next(x for x, time in enumerate(emgTimes) if time > qTimes[0])
        emgTimes = emgTimes[clipIndex:]
        emg = emg[clipIndex:, :]
        emgSecs = emgSecs[clipIndex:]

    elif qTimes[0] < emgTimes[0]:
        clipIndex = next(x for x, time in enumerate(qTimes) if time > emgTimes[0])
        qTimes = qTimes[clipIndex:]
        q = q[clipIndex:, :]
        qSecs = qSecs[clipIndex:]

    # Linear interpolation to match times of hf source to lf
    if emgDt <= qDt:
        qF = [np.interp(emgSecs, qSecs, q[:,i]) for i in range(q.shape[1])]
        qF = np.array(qF).T
        emgF = emg
        tF = emgSecs
        timestampsF = emgTimes
    else:
        qF = q
        emgF = [np.interp(qSecs, emgSecs, emg[:,i]) for i in range(emg.shape[1])]
        emgF = np.array(emgF).T
        tF = qSecs
        timestampsF = qTimes    
    return qF, emgF, tF, timestampsF


# Highpass filter, fs is the sampling frequency
def notch_filter(data, tF, notch_freq = 50, quality_factor = 20.0, order=5):

    # Compute average sample frequency
    sample_freq = 1/np.average(np.diff(tF))
    # and filter
    b, a = signal.iirnotch(notch_freq, sample_freq, quality_factor = quality_factor)
    y = signal.filtfilt(b, a, data)
    return y


# Highpass filter, fs is the sampling frequency
def highpass_filter(data, tF, cutoff_freq = 5, order=5):

    # Compute average sample frequency
    sample_freq = 1/np.average(np.diff(tF))
    # and filter
    def butter_highpass(cutoff_freq, order):
        nyq = 0.5 * sample_freq
        normal_cutoff = cutoff_freq / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    b, a = butter_highpass(cutoff_freq, sample_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def spectrogram(emgF):
    return signal.spectrogram()

# Normalize by removing a running average, dividing by columnwise standard deviation, and removing outliers
def normalize_emg_rolling(emgF, runningAverageWindow = 200, abs = False, std = False, z_min = None, z_max = None):
    # Remove running average
    emgFF = emgF - ndif.uniform_filter1d(emgF, runningAverageWindow, mode='nearest', axis = 0, origin=-(runningAverageWindow//2))
    
    # Normalize by standard deviation
    if std:
        emgFF = (emgFF)/np.std(emgF)
    
    # Take absolute value
    if abs:
         emgFF = np.abs(emgFF)

    # Set datapoints with z-score above zMax to zMax and below zMin to 0
    if z_max != None: 
        emgFF[emgFF > z_max] = z_max
        emgFF[emgFF < -z_max] = -z_max
    if z_min != None: 
        emgFF[np.abs(emgFF) < z_min] = 0
    return emgFF


def normalize_emg_kfolds(emgF, k = 2, i = 0, z_min = None, z_max = None):
    # Normalize by moments of k-fold train data
    if (i==None or k==None) or k<2 or i>=k:
        print("Incorrect parameters")
    else:
        trn_start_ind = int(i/float(k)*emgF.shape[0])
        trn_end_ind = int((i+1)/float(k)*emgF.shape[0])
        emgFF = (emgF - np.mean(emgF[trn_start_ind:trn_end_ind], axis=0))/np.std(emgF[trn_start_ind:trn_end_ind])

    # Take absolute value
    if abs:
         emgFF = np.abs(emgFF)

    # Set datapoints with z-score above zMax to zMax and below zMin to 0
    if z_max != None: 
        emgFF[emgFF > z_max] = z_max
        emgFF[emgFF < -z_max] = -z_max
    if z_min != None: 
        emgFF[np.abs(emgFF) < z_min] = 0
    return emgFF


def normalize_emg(emgF, trn_portion = None, z_min = None, z_max = None):
    # Normalize by moments of train data as selected by trn_portion
    if trn_portion == None:
        emgFF = (emgF - np.mean(emgF, axis=0))/np.std(emgF)
    else:
        trn_ind = int(trn_portion*emgF.shape[0])
        emgFF = (emgF - np.mean(emgF[:trn_ind], axis=0))/np.std(emgF[:trn_ind])

    # Take absolute value
    if abs:
         emgFF = np.abs(emgFF)

    # Set datapoints with z-score above zMax to zMax and below zMin to 0
    if z_max != None: 
        emgFF[emgFF > z_max] = z_max
        emgFF[emgFF < -z_max] = -z_max
    if z_min != None: 
        emgFF[np.abs(emgFF) < z_min] = 0
    return emgFF


def finger_nodes(qF, finger = 'indextip'):
    # Dimension of space of collected data
    DIM = 3

    # Extract Individual Fingertip Features
    if finger == 'fingertips': return qF[:,19*DIM:]
    elif finger == 'thumbtip': return qF[:,19*DIM+DIM*0:19*DIM+DIM*1]
    elif finger == 'indextip': return qF[:,19*DIM+DIM*1:19*DIM+DIM*2]
    elif finger == 'middletip': return qF[:,19*DIM+DIM*2:19*DIM+DIM*3]
    elif finger == 'ringtip': return qF[:,19*DIM+DIM*3:19*DIM+DIM*4]
    elif finger == 'pinkeytip': return qF[:,19*DIM+DIM*4:19*DIM+DIM*5]

    # or finger features
    elif finger == 'thumb': return qF[:,3*DIM:6*DIM]
    elif finger == 'index': return qF[:,6*DIM:9*DIM]
    elif finger == 'middle': return qF[:,9*DIM:12*DIM]
    elif finger == 'ring': return qF[:,12*DIM:15*DIM]
    elif finger == 'pinkey': return qF[:,15*DIM:19*DIM]

    # Otherwise return the original array
    else:
        print("Finger selection not recognized.")
        return qF


def features(emgF, axiss = 0):
    # Used repeatedly
    std = np.std(emgF, axis = axiss)
    diff = np.diff(emgF, axis = axiss)
    size = emgF.shape[0]
    # Features
    def zero_crossingsf(emgF):
     return (-1*np.count_nonzero(np.diff(np.sign(emgF), axis = axiss), axis = axiss) + size)

    zero_crossings = zero_crossingsf(emgF)
    slope_sign_changes = zero_crossingsf(diff)
    waveform_length = np.sum(np.abs(diff), axis = axiss)
    wilson_amplitude = ((diff - std) > 0).sum(axis = axiss)
    mean_absolute_value = np.average(np.abs(emgF), axis = axiss)
    mean_square = np.average(np.square(emgF), axis = axiss)
    difference_absolute_standard_deviation = np.std(np.abs(diff), axis = axiss)
    myopulse_percentage_rate = ((np.abs(emgF) - std) > 0).sum(axis = axiss)

    return np.vstack((zero_crossings, slope_sign_changes, waveform_length,
    wilson_amplitude, mean_absolute_value, mean_square, difference_absolute_standard_deviation,
    myopulse_percentage_rate))


def features_full(emgF, axiss = 0):
    # Used repeatedly
    std = np.std(emgF, axis = axiss)
    diff = np.diff(emgF, axis = axiss)
    size = emgF.shape[0]
    # Features
    def zero_crossingsf(emgF):
     return (-1*np.count_nonzero(np.diff(np.sign(emgF), axis = axiss), axis = axiss) + size)

    zero_crossings = zero_crossingsf(emgF)
    slope_sign_changes = zero_crossingsf(diff)
    waveform_length = np.sum(np.abs(diff), axis = axiss)
    wilson_amplitude = ((diff - std) > 0).sum(axis = axiss)
    mean_absolute_value = np.average(np.abs(emgF), axis = axiss)
    mean_square = np.average(np.square(emgF), axis = axiss)
    root_mean_square = np.sqrt(mean_square)
    v_order_3 = np.power(np.average(np.power(np.abs(emgF), 3), axis=axiss), 1/3.0)
    log_detector = np.exp(np.average(np.log(np.abs(emgF)),axis=axiss))
    difference_absolute_standard_deviation = np.std(np.abs(diff), axis = axiss)
    maximum_fractal_length = difference_absolute_standard_deviation - 1/2*np.log(size-1)
    myopulse_percentage_rate = ((np.abs(emgF) - std) > 0).sum(axis = axiss)
    mean_absolute_value_slope = (np.sum(np.abs(emgF[:int(size/2),:]), axis=axiss) - \
        np.sum(np.abs(emgF[int(size/2):,:]), axis=axiss))/(size/2)
    weighted_mean_absolute_value = np.sum(0.5*emgF[:int(0.25*size)], axis=axiss) + \
        np.sum(1*emgF[int(0.25*size):int(0.75*size)], axis=axiss) + \
        np.sum(0.5*emgF[:int(0.75*size)], axis=axiss)

    return np.vstack((zero_crossings, slope_sign_changes, waveform_length,
    wilson_amplitude, mean_absolute_value, mean_square, root_mean_square, v_order_3, 
    log_detector, difference_absolute_standard_deviation, maximum_fractal_length,
    myopulse_percentage_rate, mean_absolute_value_slope, weighted_mean_absolute_value))


def create_feature_dataset(tF, emgModel, qModel, trn_portion = 0.5, sequence_length_features = 50, stride_features = 15, full = False):
    # Compute input and output shapes (useful for keras models)
    input_shape = emgModel.shape[-1]
    output_shape = qModel.shape[-1]
    ds_length = (emgModel.shape[0]-sequence_length_features)//stride_features
    nSamples_trn = int(trn_portion * ds_length)
    nSamples_val = int((1 - trn_portion)/2 * ds_length)

    if not full:
        NFEATURES = int(8)
    else:
        NFEATURES = int(14)

    dt = np.mean(np.diff(tF))

    print("Computing features from", sequence_length_features*dt, "seconds prior at intervals of", stride_features*dt,"seconds")

    # Output array
    data = np.zeros((ds_length, NFEATURES, input_shape))
    for i in range(ds_length):
        if not full:
            data[i,:,:] = features(emgModel[i*stride_features : sequence_length_features+i*stride_features])
        else:
            data[i,:,:] = features_full(emgModel[i*stride_features : sequence_length_features+i*stride_features])
    # Output targets
    targets = qModel[stride_features::stride_features,:]

    # print(np.argwhere(data!=data))
    print("Data shape:", data.shape)
    # print(np.argwhere(targets!=targets))
    print("Target shape:", targets.shape)

    # Create the train, validation, and test datasets from the data and targets
    trn = tf.data.Dataset.from_tensor_slices((data[:nSamples_trn,:], targets[:nSamples_trn,:]))
    val = tf.data.Dataset.from_tensor_slices((data[nSamples_trn:nSamples_trn+nSamples_val,:], targets[nSamples_trn:nSamples_trn+nSamples_val,:]))
    tst = tf.data.Dataset.from_tensor_slices((data[nSamples_trn+nSamples_val:,:], targets[nSamples_trn+nSamples_val:,:]))   

    trn = trn.batch(1)
    val = val.batch(1)
    tst = tst.batch(1)

    return trn, val, tst, input_shape, output_shape, NFEATURES

def create_feature_dataset_stacked(tF, emgModel, qModel, trn_portion = 0.5, sequence_length = 50, stride_features = 15, sequence_length_features=5, flatter = True, batch_size = None, full = False):
    # Compute input and output shapes (useful for keras models)
    input_shape = emgModel.shape[-1]
    output_shape = qModel.shape[-1]
    ds_length = (emgModel.shape[0]-sequence_length)//stride_features
    nSamples_trn = int(trn_portion * (ds_length-sequence_length_features))
    nSamples_val = int((1 - trn_portion)/2 * (ds_length-sequence_length_features))

    if not full:
        NFEATURES = int(8)
    else:
        NFEATURES = int(14)

    dt = np.mean(np.diff(tF))

    print("Computing features from", sequence_length*dt, "seconds prior at intervals of", stride_features*dt,"seconds")
    if (batch_size != None): print("Using total data from", batch_size*stride_features*dt)

    # Intermediate array; for faster processing, eliminate, but easier to understand
    data = np.zeros((ds_length, NFEATURES, input_shape))
    # print('D',data.shape)
    
    for i in range(ds_length):
        if not full:
            data[i,:,:] = features(emgModel[i*stride_features : sequence_length+i*stride_features])
        else:
            data[i,:,:] = features_full(emgModel[i*stride_features : sequence_length+i*stride_features])
    # Intermediate targets
    targets = qModel[stride_features::stride_features,:]

    # Do the timeseries_from_array bit but internally because otherwise serious errors ??
    data_out =  np.zeros((ds_length-sequence_length_features, sequence_length_features, NFEATURES, input_shape))
    for i in range(ds_length-sequence_length_features-1):
      for j in range(sequence_length_features):
        data_out[i,j,...] = data[i+j,...]

    # Most networks do not accept higher dimensional data
    if flatter:
      data_out = np.reshape(data_out,(ds_length-sequence_length_features, sequence_length_features, NFEATURES*input_shape))
    # print('DO:',data_out.shape)


    targets = targets[sequence_length_features:,...]
    # print('T',targets.shape)

    # Create dataset objects
    trn = tf.data.Dataset.from_tensor_slices((data_out[:nSamples_trn,...], targets[:nSamples_trn,:]))
    val = tf.data.Dataset.from_tensor_slices((data_out[nSamples_trn:nSamples_trn+nSamples_val,...], targets[nSamples_trn:nSamples_trn+nSamples_val,:]))
    tst = tf.data.Dataset.from_tensor_slices((data_out[nSamples_trn+nSamples_val:,...], targets[nSamples_trn+nSamples_val:,:]))   
  
    trn = trn.batch(1)
    val = val.batch(1)
    tst = tst.batch(1)

    return trn, val, tst, input_shape, output_shape, NFEATURES


def create_dataset(tF, emgModel, qModel, trn_portion = 0.5, sequence_length = 100, batch_size = 256):
    # Compute input and output shapes (useful for keras models)
    input_shape = emgModel.shape[-1]
    output_shape = qModel.shape[-1]

    # Compute length of train, validation, and test sets
    nSamples_trn = int(trn_portion * len(emgModel))
    nSamples_val = int((1 - trn_portion)/2 * len(emgModel))

    # Normalize to mean/std of train set
    emgModelMean = emgModel[:nSamples_trn,:].mean(axis = 0)
    emgModelStd = emgModel[:nSamples_trn,:].std(axis = 0)
    emgModel = (emgModel - emgModelMean)/emgModelStd

    # Timeseries generation
    sampling_rate = 1
    stride = 1
    delay = sequence_length # At present, only try to predict hand position at current timestep
    print("Using data from ", sequence_length*np.mean(np.diff(tF)), " seconds prior")

    # Use the nice Keras function
    trn = keras.utils.timeseries_dataset_from_array(
    emgModel[:-delay, :],
    targets = qModel[delay:, :],
    sampling_rate = sampling_rate,
    sequence_stride = stride,
    sequence_length = sequence_length,
    shuffle = False,
    batch_size = batch_size,
    start_index = 0,
    end_index = nSamples_trn)

    val = keras.utils.timeseries_dataset_from_array(
    emgModel[:-delay, :],
    targets = qModel[delay:, :],
    sampling_rate = sampling_rate,
    sequence_stride = stride,
    sequence_length = sequence_length,
    shuffle = False,
    batch_size = batch_size,
    start_index = nSamples_trn,
    end_index = nSamples_trn + nSamples_val)

    tst = keras.utils.timeseries_dataset_from_array(
    emgModel[:-delay, :],
    targets = qModel[delay:, :],
    sampling_rate = sampling_rate,
    sequence_stride = stride,
    sequence_length = sequence_length,
    shuffle = False,
    batch_size = batch_size,
    start_index = nSamples_trn + nSamples_val)

    return trn, val, tst, input_shape, output_shape


def create_feature_dataset_cv(tF, emgModel, qModel, sequence_length = 100, stride = 5, NFEATURES = 8):
    # Compute input and output shapes (useful for keras models)
    input_shape = emgModel.shape[-1]
    output_shape = qModel.shape[-1]
    ds_length = (emgModel.shape[0]-sequence_length)//stride

    dt = np.mean(np.diff(tF))

    print("Computing features from", sequence_length*dt, "seconds prior at intervals of", stride*dt,"seconds")

    # Output array
    data = np.zeros((ds_length, NFEATURES, input_shape))
    for i in range(ds_length):
        data[i,:,:] = features(emgModel[i*stride : sequence_length+i*stride])
    # Output targets
    targets = qModel[::stride,:]

    return data, targets, input_shape, output_shape


# Single dataset if cross-validation is to be used
def create_dataset_cv(tF, emgModel, qModel, sequence_length = 100, batch_size = None):
    # Compute input and output shapes (useful for keras models)
    input_shape = emgModel.shape[-1]
    output_shape = qModel.shape[-1]

    # Timeseries generation
    sampling_rate = 1
    stride = 1
    delay = sequence_length # At present, only try to predict hand position at current timestep
    print("Using data from ", sequence_length*np.mean(np.diff(tF)), " seconds prior")

    # Use the nice Keras function
    dataset = keras.utils.timeseries_dataset_from_array(
    emgModel[:-delay, :],
    targets = qModel[delay:, :],
    sampling_rate = sampling_rate,
    sequence_stride = stride,
    sequence_length = sequence_length,
    shuffle = False,
    batch_size = emgModel.shape[0])

    # Convert to numpy
    data = list(dataset)[0][0].numpy()
    targets = list(dataset)[0][1].numpy()
    return data, targets, input_shape, output_shape


def get_model_name(k):
    return 'model_'+str(k)+'.h5'


def naive_method(qModel, trn_portion = 0.5, guess_coord = 0):
    # Compute length of train, validation, and test sets
    nSamples_trn = int(trn_portion * len(qModel))
    nSamples_val = int((1 - trn_portion)/2 * len(qModel))
    nSamples_tst = len(qModel) - nSamples_trn - nSamples_val

    # compute the RMSE of guessing that the guess_coord recorded value is the position at all timesteps
    val_squares = np.square(qModel[nSamples_trn:nSamples_trn+nSamples_val,:]-qModel[nSamples_trn + guess_coord,:])
    naive_rmse_val = np.sqrt(np.sum(val_squares)/nSamples_val)
    tst_squares = np.square(qModel[nSamples_trn+nSamples_val:,:]-qModel[nSamples_trn+nSamples_val + guess_coord,:])
    naive_rmse_tst = np.sqrt(np.sum(tst_squares)/nSamples_tst)
    std_val = np.sqrt(np.sum(np.square(qModel[nSamples_trn:nSamples_trn+nSamples_val,:] - \
        np.average(qModel[nSamples_trn:nSamples_trn+nSamples_val,:],axis=0)))/qModel[nSamples_trn:nSamples_trn+nSamples_val,:].shape[0])
    std_tst = np.sqrt(np.sum(np.square(qModel[nSamples_trn+nSamples_val:,:] - np.average(qModel[nSamples_trn+nSamples_val:,:],axis=0)))/ \
        qModel[nSamples_trn+nSamples_val:,:].shape[0])

    return naive_rmse_val, naive_rmse_tst, std_val, std_tst


# Functions for visualization
def plot_emg(tF, emgF):
    plt.rc('font', size=10)
    fig0, ax0 = plt.subplots(12, figsize=(30, 50), sharex=True)

    for i in range(emgF.shape[-1]):
        ax0[i].plot(tF[:], abs(emgF[:,i]), color='tab:orange', label='Sensor Data')
        ax0[i].grid(True)
        ax0[i].set_ylabel('EMG Data')

    ax0[emgF.shape[-1] - 1].set_xlabel('Time (s)')
    ax0[0].legend(loc='upper left')
    ax0[0].set_title('EMG Data')


def model_summary(model, history, tst):  
    print(f"Test RMSE: {model.evaluate(tst)[1]:.4f}")
    print(model.summary())

    loss = history.history["root_mean_squared_error"]
    val_loss = history.history["val_root_mean_squared_error"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training RMSE")
    plt.plot(epochs, val_loss, "b", label="Validation RMSE")
    plt.title("Training and Validation RMSE")
    plt.legend()