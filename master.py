from pandas import DataFrame, Series, concat, datetime, read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt, ceil
import operator
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback,ModelCheckpoint
from keras.layers import Dense, TimeDistributed, LSTM, Dropout, GRU
import os



# frame a sequence as a supervised learning problem
def timeseries_to_supervised(df, lag=1):
	shift = df.shift(-lag)
	shift =  shift[shift.columns.values[0]]
	df['target'] = Series(shift)
	return df


def bin_data(raw_data,action,time_start,time_end,future):
	time_start,time_end = min(raw_data['time']),max(raw_data['time'])
	n_bins = int(ceil((time_end - time_start)/(future*60)))
	data = np.zeros((n_bins,len(list(raw_data.columns.values))-1))


	colnames = list(raw_data.columns.values)[1:]

	for i in range(n_bins):
		temp = raw_data.loc[ raw_data['time'].isin(np.arange(time_start+(60*future)*(i),time_start+(60*future)*(i+1))) ]

		for j in range(len(colnames)):
			if action[j] == "mean":
				if len(temp[colnames[j]]) == 0:
					data[i][j] = data[i-1][j]
				else:
					data[i][j] = temp[colnames[j]].mean()
			else:
				data[i][j] = temp[colnames[j]].sum()

	df = DataFrame(data=data, columns=colnames)
	return df

def difference(data, interval=1):
	columns = list(data.columns.values)
	for i in range(data.shape[1]):
		tempseries = []
		for k in range(interval):
			tempseries.append(0)
		for j in range(interval,data.shape[0]):
			tempseries.append( data[columns[i]][j] - data[columns[i]][j - interval])
		data[columns[i]] = Series(tempseries[interval-1:])
	return data

def inverse_difference(history, prediction, interval=1):
	return history[interval] + prediction

def scale(data,train_frac=0.7):
	# fit scaler
	scalers = list()
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(data['price'].values.reshape(-1,1)[:int(train_frac*len(data))])
	columns = list(data.columns.values)
	data['price'] = scaler.transform(data['price'].values.reshape(-1,1))
	scalers.append(scaler)

	scaler = StandardScaler()
	scaler = scaler.fit(data['price'].values.reshape(-1,1)[:int(train_frac*len(data))])
	data['price'] = scaler.transform(data['price'].values.reshape(-1,1))
	scalers.append(scaler)

	for i in range(data.shape[1]-1):
		tempscaler = MinMaxScaler(feature_range=(-1, 1))
		tempscaler = tempscaler.fit(data[columns[i+1]].values.reshape(-1,1)[:int(train_frac*len(data))])
		data[columns[i+1]] = tempscaler.transform(data[columns[i+1]].values.reshape(-1,1))

		tempscaler = StandardScaler()
		tempscaler = tempscaler.fit(data[columns[i+1]].values.reshape(-1,1)[:int(train_frac*len(data))])
		data[columns[i+1]] = tempscaler.transform(data[columns[i+1]].values.reshape(-1,1))

	return scalers, data

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	for i in range(len(scaler)):
		array = scaler[i].inverse_transform(array)
	return array[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons,loss,val_loss):

	optimizer = "adam"


	columns = list(train.columns.values)[:-1]
	X, y = train[columns], train['target']
	X = X.values.reshape(X.shape[0], 1, X.shape[1])


	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))


	early = EarlyStopping(monitor ='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
	checkpointer = ModelCheckpoint(filepath="temp_weights.hdf5", verbose=0,save_best_only=True,save_weights_only=True)
	rms = RMSprop()
	adam = Adam()
	if optimizer == "rmsprop":
		model.compile(loss="mean_squared_error", optimizer=rms)
	elif optimizer =="adam":
		model.compile(loss="mean_squared_error", optimizer=adam)
	callbacks = [early,checkpointer]
	model.compile(loss='mean_squared_error', optimizer='adam')


	history = model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False,callbacks=callbacks,validation_split=0.2)
	loss.append(history.history['loss'])
	val_loss.append(history.history['val_loss'])
	model.reset_states()
	model.load_weights("temp_weights.hdf5")
	encrypt_file(open(os.path.join(os.path.expanduser('~'),'BitnetsAESKey.txt'), "r").read(),"temp_weights.hdf5")
	return model, loss, val_loss

def forecast_lstm(model, batch_size, X):
	"Make a forward Prediction"
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


from Crypto.Cipher import AES
from Crypto import Random
import os, random, struct

def encrypt_file(key, in_filename, out_filename=None, chunksize=64*1024):

    if not out_filename:
        out_filename = in_filename + '.enc'

    iv = ''.join(chr(random.randint(0, 0xFF)) for i in range(16))
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)

    with open(in_filename, 'rb') as infile:
        with open(out_filename, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)

            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += ' ' * (16 - len(chunk) % 16)

                outfile.write(encryptor.encrypt(chunk))
    os.remove(in_filename)

def decrypt_file(key, in_filename, out_filename=None, chunksize=24*1024):
    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)
    os.remove(in_filename)