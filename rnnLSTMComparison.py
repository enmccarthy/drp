#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1"; 
import tensorflow as tf
import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score
from datapipe import bin_ndarray, calcDff
from sklearn.externals import joblib
#TODO: read tf documentation on what the static_rnn does
#do I want to one hot encode the data? or already too many dimensions


# In[19]:


import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, GRU, Embedding, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K


# In[3]:


np.random.seed(0)


# In[4]:


tf.__version__
from tensorflow.python.ops import control_flow_ops

orig_while_loop = control_flow_ops.while_loop

def patched_while_loop(*args, **kwargs):
    kwargs.pop("maximum_iterations", None)  # Ignore.
    return orig_while_loop(*args, **kwargs)


control_flow_ops.while_loop = patched_while_loop


# In[ ]:


warmup_steps =2


# In[ ]:


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# In[5]:


#loading full file on disk, can access small slices without loading into memory
dy = np.load('./data/dffAV_2.npy', mmap_mode='r')
# dy = process_dy(dy)
# dy = calcDff(dy, .2, 30)
f = open('./data/masdfAV_2.pkl', 'rb')
df = pickle.load(f)
(length, y, x) = dy.shape
print(length, x, y)


# In[14]:


#30hz data 
dfcropped = df[["stimType", "pupilCurr", "snoutCurr","jawCurr","wheelCurr","responseT", "H", "M", "FA", "CR",
                 "whiskCurr", "lastStimT","vidCurr"]]
dfcropped = dfcropped[-10000:]
dy = dy[-10000:]
(length, leny, lenx) = dy.shape
dycropped = dy.reshape((length, leny*lenx))


# In[15]:


#make a test set for all models
x_scaler30 = joblib.load('x_scaler30hz.pkl')
x_scaler15 = joblib.load('x_scaler15hz.pkl')
x_scaler10 = joblib.load('x_scaler10hz.pkl')
y_scaler30 = joblib.load('y_scaler30hz.pkl')
y_scaler15 = joblib.load('y_scaler15hz.pkl')
y_scaler10 = joblib.load('y_scaler10hz.pkl')


# In[16]:


def r2_keras(y_true, y_pred):
    SS_res =  np.sum(np.square(np.subtract(y_true,y_pred))) 
    print(SS_res)
    SS_tot = np.sum(np.square(np.subtract(y_true, np.mean(y_true)))) 
    print(SS_tot)
    return ( 1 - SS_res/(SS_tot))


# In[24]:


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    K.clear_session()
   
    model30hz = load_model("bestDRPLSTM30hzDS2.h5", custom_objects={'loss_mse_warmup':loss_mse_warmup})
    # Use test-data.
    x = dfcropped
    y_true = dycropped
    y_true = y_true.astype(np.float32)
    x_scaled = x_scaler30.transform(x.values)
    y_scaled = y_scaler30.transform(y_true)
    # Input-signals for the model.
    x_scaled = np.expand_dims(x_scaled, axis=0)
    
    # Use the model to predict the output-signals.
    y_pred = model30hz.predict(x_scaled)
    print("y_pred")
    print(y_pred.dtype)
    print(y_true.dtype)
    sum_pred = []
    sum_true = []
 
    # For each output-signal.
    for i in range(len(x_scaled[0])):
        sum_pred.append(np.sum(y_pred[0][i])/(lenx*leny))
        sum_true.append(np.sum(y_scaled[i])/(lenx*leny))
    # Plot and compare the two signals.
    plt.plot(sum_true, label='true30hz')
    plt.plot(sum_pred, label='pred30hz')
    print("r230hz")
    print(r2_keras(y_scaled, y_pred[0]))
    print(r2_score(y_scaled, y_pred[0], multioutput='uniform_average'))
    print(explained_variance_score(y_scaled, y_pred[0],multioutput='uniform_average'))
    print(model30hz.evaluate(x=x_scaled, y=np.expand_dims(y_scaled, axis=0)))    
# Plot grey box for warmup-period.
    K.clear_session()
    x_scaled = x_scaler15.transform(x.values)
    y_scaled = y_scaler15.transform(y_true)
    # Input-signals for the model.
    x_scaled = np.expand_dims(x_scaled, axis=0)

    model15hz = load_model("bestDRPLSTM15hzDS2.h5", custom_objects={'loss_mse_warmup':loss_mse_warmup})
    # Use the model to predict the output-signals.
    y_pred = model15hz.predict(x_scaled)
    print("y_pred")
    print(y_pred.dtype)

    sum_pred = []
    sum_true = []
 
    # For each output-signal.
    for i in range(len(x_scaled[0])):
        sum_pred.append(np.sum(y_pred[0][i])/(lenx*leny))
        sum_true.append(np.sum(y_scaled[i])/(lenx*leny))
    # Plot and compare the two signals.
    plt.plot(sum_true, label='true15hz')
    plt.plot(sum_pred, label='pred15hz')
    print("r215hz")
    print(r2_keras(y_scaled, y_pred[0]))
    print(r2_score(y_scaled, y_pred[0], multioutput='uniform_average'))
    print(explained_variance_score(y_scaled, y_pred[0],multioutput='uniform_average'))
    print(model15hz.evaluate(x=x_scaled, y=np.expand_dims(y_scaled, axis=0)))    
# Plot grey box for warmup-period.
    K.clear_session()
    model10hz = load_model("bestDRPLSTM10hzDS2.h5", custom_objects={'loss_mse_warmup':loss_mse_warmup})
    x_scaled = x_scaler10.transform(x.values)
    y_scaled = y_scaler10.transform(y_true)
    # Input-signals for the model.
    x_scaled = np.expand_dims(x_scaled, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model10hz.predict(x_scaled)
    print("y_pred")
    print(y_pred.dtype)
    print(y_true.dtype)
    sum_pred = []
    sum_true = []
 
    # For each output-signal.
    for i in range(len(x_scaled[0])):
        sum_pred.append(np.sum(y_pred[0][i])/(lenx*leny))
        sum_true.append(np.sum(y_scaled[i])/(lenx*leny))
    # Plot and compare the two signals.
    plt.plot(sum_true, label='true10hz')
    plt.plot(sum_pred, label='pred10hz')
    print("r210hz")
    print(r2_keras(y_scaled, y_pred[0]))
    print(r2_score(y_scaled, y_pred[0], multioutput='uniform_average'))
    print(explained_variance_score(y_scaled, y_pred[0],multioutput='uniform_average'))
    print(model10hz.evaluate(x=x_scaled, y=np.expand_dims(y_scaled, axis=0)))
    K.clear_session()
    # Plot grey box for warmup-period.
    # Plot labels etc.
    plt.ylabel("sum")
    plt.legend()
    plt.savefig('compare.png')


# In[25]:


plot_comparison(start_idx=2, length=1000, train=False)


# In[ ]:





