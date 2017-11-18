from master import *
from encrypter import *



training_fraction = 0.8
repeats = 1
epochs = 2
nb_neurons = 2
batch_size = 25
future =  5
n_samples = 1
thetype = "lstm"
stateful = False

df = DataFrame()
df['price'] = np.sin(np.linspace(0,10,50))



# Make a differenced series
df = difference(df, future)
df = df.dropna()


# Scale series between -1 and 1
scaler, df = scale(df,training_fraction)

df = timeseries_to_supervised(df, future)
df = df.dropna()
batch_trim = len(df)%batch_size
df = df.iloc[batch_trim:]

train, test = df[:-int( (1-training_fraction)*len(df)) ], df[-int( (1-training_fraction)*len(df)):]


loss,val_loss = [], []
error_scores = list()


net1 = neuralNet(nb_neurons,thetype,batch_size,stateful)

net1.fit(train,epochs)

columns = list(train.columns.values)
X = train[columns[:-1]]
X = X.values.reshape(X.shape[0], 1, X.shape[1])

net1.forecast(X, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test)-future):
	# make one-step forecast
	X, y = test[columns[:-1]].values[i], test['target'].values[i]
	prediction = net1.forecast(X,1)
	prediction = invert_scale(scaler, X, prediction)

	prediction = inverse_difference(df['price'].values[-len(test):], prediction, i)
	predictions.append(prediction)

predictions = list(reversed(predictions))
for i in range(future):
	predictions.append(predictions[-1])
predictions = list(reversed(predictions))
rmse = sqrt(mean_squared_error(df['price'].values[-len(test):], predictions))
print('%d) Test RMSE: %.3f' % (r+1, rmse))
if r == 0:
	net1.save("best_model.h5")

else:
	if rmse < min(error_scores):
		print "Best model so far has been found: Saving model"
		net1.save("best_model.h5")

error_scores.append(rmse)



# EVALUATE THE BEST PREDICTIONS
crypt.decrypt("best_model.h5")
lstm_model.load_weights("best_model.h5")
crypt.encrypt("best_model.h5")
net1.reset_states()
columns = list(train.columns.values)
X = train[columns[:-1]]
X = X.values.reshape(X.shape[0], 1, X.shape[1])
net1.forecast(X, batch_size=1)
# walk-forward validation on the test data
predictions = list()
for i in range(len(test)-future):
	X, y = test[columns[:-1]].values[i], test['target'].values[i]
	prediction = net1.forecast(lX,1)
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