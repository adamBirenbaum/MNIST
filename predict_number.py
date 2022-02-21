import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from make_model_function import make_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys


batch_size = 256

physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def crop_image(im):

	def get_border(repeats):
		max_val = np.max(repeats)
		max_ind = np.argmax(repeats)

		#if max_ind - max_val <= 0:
		#	max_ind = np.argsort(np.diff(repeats))[1]
		min_ind = max_ind - repeats[max_ind]
		#breakpoint()
		return [int(min_ind), int(max_ind)]

	im[im >= thresh_val] = 255
	im[im < thresh_val] = 0

	# plt.close()
	# plt.imshow(im)
	# plt.show()
	min_rows = np.min(im,axis=0)

	min_cols = np.min(im, axis=1)



	repeats1 = 0
	repeats_vec1 = np.zeros(len(min_cols))
	repeats2 = 0
	repeats_vec2 = np.zeros(len(min_rows))

	for i, val1 in enumerate(min_cols):
		if val1 == 0:
			repeats1 += 1
		else:
			repeats1 = 0
		repeats_vec1[i] = repeats1
	
	for i, val2 in enumerate(min_rows):
		if val2 == 0:
				repeats2 += 1
		else:
			repeats2 = 0
			
		repeats_vec2[i] = repeats2



	row_border = get_border(repeats_vec1)
	col_border = get_border(repeats_vec2)

	width = col_border[1]-  col_border[0]
	height = row_border[1] - row_border[0]
	max_dim = max([width, height])
	
	final_dim = int(np.power(np.floor(np.sqrt(max_dim)) + 1,2))

	c1 = int((final_dim - width) / 2)
	if (final_dim - width) %2 == 0:
		c2 = c1
	else:
		c2 = final_dim - width - c1

	r1 = int((final_dim - height) / 2)
	if (final_dim - height) %2 == 0:
		r2 = r1
	else:
		r2 = final_dim - height - r1

	row_border[0], row_border[1] = row_border[0] - r1, row_border[1] + int(r2)
	col_border[0], col_border[1] = col_border[0] - c1, col_border[1] + int(c2)




	im = im[row_border[0]:row_border[1], col_border[0]:col_border[1]]

	# pad with zeros

	extra_dim = int(np.ceil((final_dim*1.8 - final_dim) / 2))
	extra_row = np.ones((extra_dim, final_dim))*255
	im = np.vstack([extra_row, im, extra_row])
	new_row = im.shape[0]
	extra_col = np.ones((new_row, extra_dim))*255
	im = np.hstack((extra_col, im, extra_col))
	return im

if __name__ == '__main__':

	thresh_val = 200

	inputs = sys.argv

	image = inputs[1]

	im = np.array(Image.open(image).convert('L'))

	

	fig, ax = plt.subplots(1,3)

	pos = ax[0].imshow(np.abs(255.0-im))
	ax[0].set_title('original')

	im = crop_image(im)
	im = np.abs(255.0 - im)

	Image.fromarray(im).convert('L').save('./test.jpg')


	pos = ax[1].imshow(im)
	ax[1].set_title('auto-cropped')
	im = np.array(Image.open('./test.jpg').resize((28,28),Image.LANCZOS).convert('L')) / 255.0
	im[im> 0.1] = 1
	pos = ax[2].imshow(im)
	ax[2].set_title('Input for Classifier')
	model = tf.keras.models.load_model(os.path.join('./outputs/saved_models/'))

	prediction = model.predict(im.reshape(1,28,28,1))
	print(prediction)
	fig.suptitle('Prediction: {}'.format(np.argmax(prediction[0])))
	plt.show()

	breakpoint()
	
