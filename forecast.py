from master import *






training_fraction = 0.8
repeats = 1
epochs = 20
nb_neurons = 10
batch_size = 1
future =  5
n_samples = 10



# load and prepare price dataset
bitcoindata = read_csv(os.path.join(os.path.expanduser('~'),'krakenEUR.csv'), header=None,names=['time','price','volume'], squeeze=True,nrows=n_samples)
time_start,time_end = min(bitcoindata['time']),max(bitcoindata['time'])
action = ['mean','none']


df = bin_data(bitcoindata,action,time_start,time_end,future)

# I will need to combine all datasets here in the future

df = DataFrame()
df['price'] = np.sin(np.arange(0,10,0.1))


# Make a differenced series
df = difference(df, future)
df = df.dropna()


# Scale series between -1 and 1
scaler, df = scale(df,training_fraction)

df = timeseries_to_supervised(df, future)
df = df.dropna()



train, test = df[:-int( (1-training_fraction)*len(df)) ], df[-int( (1-training_fraction)*len(df)):]
loss, val_loss = [],[]
# EVALUATE THE BEST PREDICTIONS
decrypt_file(open(os.path.join(os.path.expanduser('~'),'BitnetsAESKey.txt'), "r").read(),"best_model.hdf5.enc")
keras.models.load_model("best_model.hdf5")
lstm_model.load_weights("best_model.hdf5")
encrypt_file(open(os.path.join(os.path.expanduser('~'),'BitnetsAESKey.txt'), "r").read(),"best_model.hdf5")
lstm_model.reset_states()
columns = list(train.columns.values)
X = train[columns[:-1]]
X = X.values.reshape(X.shape[0], 1, X.shape[1])
lstm_model.predict(X, batch_size=batch_size)
# walk-forward validation on the test data
predictions = list()
for i in range(len(test)-future):
	X, y = test[columns[:-1]].values[i], test['target'].values[i]
	prediction = forecast_lstm(lstm_model, batch_size, X)
	prediction = invert_scale(scaler, X, prediction)
	prediction = inverse_difference(df['price'].values[-len(test):], prediction, i)
	predictions.append(prediction)
predictions = list(reversed(predictions))
for i in range(future):
	predictions.append(predictions[-1])
predictions = list(reversed(predictions))
rmse = sqrt(mean_squared_error(df['price'].values[-len(test):], predictions))
print "-----------------"
print('%d) BEST RMSE: %.3f' % (r+1, rmse))

plt.plot(predictions,label="Predictions")
plt.plot(df['price'].values[-len(test):],label="True Values")
plt.ylabel("Price")
plt.xlabel("Time - minutes")
plt.legend()
plt.show()


loss = reduce(operator.add,loss)
val_loss= reduce(operator.add,val_loss)
plt.plot(loss,label="Training")
plt.plot(val_loss,label="Validation")
plt.legend()
plt.xlabel("Training Epoch")
plt.ylabel("Error")
plt.semilogy()
plt.show()

if repeats != 1:
	# summarize results
	results = DataFrame()
	results['rmse'] = error_scores
	print(results.describe())
	results.boxplot()
	plt.show()

#"""