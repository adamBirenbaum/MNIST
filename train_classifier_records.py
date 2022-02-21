import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from make_model_function import make_model
import numpy as np
import matplotlib.pyplot as plt
import glob


batch_size = 256

physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_tfr_example(example):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'label':tf.io.FixedLenFeature([], tf.int64),
      'features' : tf.io.FixedLenFeature([], tf.string)
    }

    
  content = tf.io.parse_single_example(example, data)
  
  label = content['label']
  raw_image = content['features']
  
  
  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(raw_image, out_type=tf.float64)
  #feature = tf.reshape(feature, shape=[height,width,depth])
  return feature, label

def get_dataset(tfr_dir, pattern):
    files = glob.glob(tfr_dir+'/'+pattern, recursive=False)

    #create the dataset
    
    dataset = tf.data.TFRecordDataset(files)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_example
    )

    return dataset


if __name__ == '__main__':

	tf.random.set_seed(100)

	model_file = './models/model.yaml'

	model = make_model(model_file)


	train_dataset = get_dataset('./data/tf_records','train*').batch(batch_size).repeat().prefetch(1)
	valid_dataset = get_dataset('./data/tf_records','test*').batch(batch_size).repeat().prefetch(1)



	n_epochs = 100
	
	#plot_data(X,y,0)

	save_model = tf.keras.callbacks.ModelCheckpoint(os.path.join('./outputs','saved_models2'), monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False)

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('./outputs','tensorboard_logs2'),write_images=True)

	model.fit(train_dataset,epochs=n_epochs, steps_per_epoch=60000//batch_size, validation_data = valid_dataset,verbose=1,validation_steps=10000//batch_size,batch_size=batch_size,callbacks=[save_model, tensorboard])


	predictions = model.predict(train_dataset, batch_size=batch_size)
	predictions = np.argmax(predictions,axis=1)

	for i in range(100):
		plot_data(X[i], y[i], predictions[i], os.path.join('./outputs','plots','output_{:03d}'.format(i)))
	breakpoint()


