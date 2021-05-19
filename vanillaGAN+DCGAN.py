import os
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape
from keras import losses, optimizers
from keras.models import Sequential, Model
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import glob
from IPython.display import Image, display
import imageio

os.makedirs("fake_images", exist_ok=True) # create a folder for generated images
matplotlib.use('Agg')

# PARAMETERS SETTINGS
lr = 0.0002
noise_dim = 100
img_samples = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 64
EPOCHS = 40
gan_type = 'DCGAN'
# gan_type = 'vGAN' #uncomment this line to change gan type to vanilla gan



# DATASET PREPRARE
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # normalize the images to (-1, 1)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



# GAN MODEL
def Discriminator(type):
	if type == 'vGAN':
		return Discriminator_vn()
	else:
		return Discriminator_dc()

def Generator(type):
	if type == 'vGAN':
		return Generator_vn()
	else:
		return Generator_dc()

# Vanilla GAN
def Discriminator_vn():
	model = Sequential()
	model.add(Flatten(input_shape=[28, 28, 1]))
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(128))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(64))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(1, activation='sigmoid'))

	return model

def Generator_vn():
	model = Sequential()

	model.add(Dense(64, input_dim=noise_dim))
	model.add(LeakyReLU(0.3))
	model.add(Dropout(0.3))

	model.add(Dense(128, input_dim=noise_dim))
	model.add(LeakyReLU(0.3))
	model.add(Dropout(0.3))

	model.add(Dense(256))
	model.add(LeakyReLU(0.3))
	model.add(Dropout(0.3))

	model.add(Dense(np.prod([28,28,1]), activation='tanh'))
	model.add(Reshape([28, 28, 1]))

	return model


# DCGAN
def Discriminator_dc():
	model = Sequential()

	# (1×28×28) --> (64×14×14)
	model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same',input_shape=[28, 28, 1]))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.3))
	model.add(Dropout(0.3))

	# (64×14×14) --> (128×7×7) 
	model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.3))
	model.add(Dropout(0.3))

	# (128×7×7) -> (6272×1) -> (1)
	model.add(Flatten())
	model.add(Dense(1))

	return model

def Generator_dc():
	model = Sequential()
	
	# 100 --> (256×7×7)
	model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.3))
	model.add(Reshape((7, 7, 256)))
	
	# (256×7×7) --> (128×7×7), s=1, p=2
	model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.3))

	# (128×7×7) --> (64×14×14)
	model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.3))
	

	# (64×14×14) --> (1×28×28)
	model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))


	return model


D = Discriminator(gan_type)
D.summary()
G = Generator(gan_type)
G.summary()


# LOSS func, OPTIMIZER
g_optimizer = optimizers.Adam(lr)
d_optimizer = optimizers.Adam(lr)

criterion = losses.BinaryCrossentropy()
if gan_type == 'DCGAN':
	criterion = losses.BinaryCrossentropy(from_logits=True)

def get_g_loss(fake_output):
	return criterion(tf.ones_like(fake_output), fake_output)

def get_d_loss(real_output, fake_output):
	real_loss = criterion(tf.ones_like(real_output), real_output)
	fake_loss = criterion(tf.zeros_like(fake_output), fake_output)
	total_loss = (real_loss + fake_loss) / 2
	return total_loss


# TRAINING

tf.random.set_seed(0) # a fixed noise to feed in trained G model

z_seed = tf.random.normal([img_samples, noise_dim]) 
if gan_type == 'vGAN':
	z_seed = np.random.normal(0, 1, (img_samples, noise_dim))
  
# record loss after each epoch
losses_hist_g = []
losses_hist_d = []

def train(dataset, epochs):
	for epoch in range(epochs):
		dataset_len = len(dataset)
		epoch_loss_g, epoch_loss_d = 0.0, 0.0
		
		for image_batch in dataset:
			# Train D and G
			gen_loss, disc_loss = train_step(image_batch)
			epoch_loss_g += gen_loss
			epoch_loss_d += disc_loss

		# Compute loss for each epoch
		epoch_loss_g /= dataset_len
		epoch_loss_d /= dataset_len
		losses_hist_g.append(epoch_loss_g)
		losses_hist_d.append(epoch_loss_d)
		print (f"Epoch {epoch+1} of {EPOCHS}:	Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

		# plot and save image using fixed noise and trained model
		generate_and_save_images(epoch+1)


# helper function for training batch-image for both D and G
@tf.function
def train_step(images, ):
	# create noise
	z = tf.random.normal([BATCH_SIZE, noise_dim])
	if gan_type == 'vGAN':
		z = np.random.normal(0, 1, (BATCH_SIZE, noise_dim))
	   
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_images = G(z, training=True)

		real_out = D(images, training=True)
		fake_out = D(gen_images, training=True)

		gen_loss = get_g_loss(fake_out)
		disc_loss = get_d_loss(real_out, fake_out)

	grad_of_g = gen_tape.gradient(gen_loss, G.trainable_variables)
	grad_of_d = disc_tape.gradient(disc_loss, D.trainable_variables)

	g_optimizer.apply_gradients(zip(grad_of_g, G.trainable_variables))
	d_optimizer.apply_gradients(zip(grad_of_d, D.trainable_variables))
  
	return gen_loss, disc_loss

# helper function to plot and save fake images
def generate_and_save_images(epoch):
	out = G(z_seed, training=False)

	fig = plt.figure(figsize=(6, 6))
	for i in range(out.shape[0]):
		plt.subplot(10, 10, i+1)
		plt.imshow(out[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
		plt.axis('off')

	plt.savefig('fake_images/img_{:02d}.png'.format(epoch))


# MAIN
def main():
	print("============================= START TRAINING ==============================")
	start = time.time()
	train(train_dataset, EPOCHS)
	time_cost = time.time() - start
	print("============================== END TRAINING ===============================")
	print(f"Total time cost: {time_cost:.2f} secs.")

if __name__ == "__main__":
	main()