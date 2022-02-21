import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from make_model_function import make_model
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree
batch_size = 256

physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def plot_data(X, y,prediction, filename):

	fig, ax = plt.subplots(figsize=(8,8))
	image = np.asarray(X).squeeze()
	plt.imshow(image)
	plt.title('Actual: {}\nPrediction: {}'.format(y, prediction))
	plt.savefig(filename, dpi=200)
	

if __name__ == '__main__':

	tf.random.set_seed(100)

	model_file = './models/model.yaml'

	model = make_model(model_file)


	X = np.loadtxt('./data/train_images.txt') / 255
	X = X.reshape(60000, 28, 28, 1)
	
	y = np.loadtxt('./data/train_labels.txt',dtype=np.int32)
	y = y.reshape(60000,1)

	X_valid = np.loadtxt('./data/test_images.txt') / 255
	X_valid = X_valid.reshape(10000, 28, 28, 1)

	y_valid = np.loadtxt('./data/test_labels.txt',dtype=np.int32)
	y_valid = y_valid.reshape(10000,1)

	n_epochs = 100
	
	#plot_data(X,y,0)

	save_model = tf.keras.callbacks.ModelCheckpoint(os.path.join('./outputs','saved_models'), monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False)


	tensorboard_dir = os.path.join('./outputs','tensorboard_logs')

	try:
	
		rmtree(tensorboard_dir)
	except:
		pass


	lr_dec = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20)
	model.fit(X, y,epochs=n_epochs, validation_data = (X_valid, y_valid),verbose=1,batch_size=batch_size,callbacks=[lr_dec, save_model, tensorboard, early_stop])


	predictions = model.predict(X, batch_size=batch_size)
	predictions = np.argmax(predictions,axis=1)

	for i in range(100):
		plot_data(X[i], y[i], predictions[i], os.path.join('./outputs','plots','output_{:03d}'.format(i)))
	breakpoint()


