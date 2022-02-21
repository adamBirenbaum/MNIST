import gzip
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import os

def write_TFRecords(features, labels, filename):
	out_dir = './data/tf_records/'
	#determine the number of shards (single TFRecord files) we need:


	for i in tqdm.tqdm(range(features.shape[0])):
		current_shard_name = os.path.join(out_dir, "{}.tfrecords".format(filename))
		writer = tf.io.TFRecordWriter(current_shard_name)

		current_shard_count = 0
		current_feature = features[i]
		current_label = labels[i]

		#create the required Example representation
		out = create_example(current_feature, current_label)

		writer.write(out.SerializeToString())


	writer.close()


def create_example(features, label):

	#define the dictionary -- the structure -- of our single example
	data = {
				'features' : _convert_array_to_bytes_feature(features),
				'label' : _int64_feature(label)
		}
	#create an Example, wrapping the single features
	out = tf.train.Example(features=tf.train.Features(feature=data))

	return out

def _bytes_feature(value):
		"""Returns a bytes_list from a string / byte."""
		if isinstance(value, type(tf.constant(0))): # if value ist tensor
				value = value.numpy() # get value of tensor
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a floast_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
	array = tf.io.serialize_tensor(array)
	return array

def _convert_array_to_bytes_feature(array):
	if isinstance(array, np.ndarray):
		array = tf.convert_to_tensor(array)

	serialize_array = tf.io.serialize_tensor(array)
	feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialize_array.numpy()]))

	return feature



#def process_image(file_path, num_images)
f = gzip.open('./data/raw_files/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000


f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
np.savetxt('./data/train_images.txt',data)


f = gzip.open('./data/raw_files/train-labels-idx1-ubyte.gz','r')
f.read(8)
labels=np.zeros(num_images)
for i in range(num_images):   
    buf = f.read(1)
    labels[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

np.savetxt('./data/train_labels.txt',labels,fmt='%d')
labels = labels.astype(int)

data = data.reshape(num_images, image_size, image_size, 1)
write_TFRecords(data, labels, 'train_data')







f = gzip.open('./data/raw_files/t10k-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 10000


f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
np.savetxt('./data/test_images.txt',data)


f = gzip.open('./data/raw_files/t10k-labels-idx1-ubyte.gz','r')
f.read(8)
labels=np.zeros(num_images)
for i in range(num_images):   
    buf = f.read(1)
    labels[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

np.savetxt('./data/test_labels.txt',labels,fmt='%d')
labels = labels.astype(int)


data = data.reshape(num_images, image_size, image_size, 1)
write_TFRecords(data, labels, 'test_data')



breakpoint()






    
