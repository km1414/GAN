"""
Simple Generative adversarial network, learned to generate
images from selected video file (located in 'data' folder).
Adapted from Advanced Machine Learning Specialization, Coursera.
2018, July.
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers as L
import tensorflow as tf
import os, cv2


# selected size for real data and generated images
IMG_SHAPE = (50, 50, 3)
# selected size for image encoding
CODE_SIZE = 512


class GAN():

    def __init__(self):

        """ Initialize GAN object. """

        self.global_epoch = 0
        self.IMG_SHAPE = IMG_SHAPE
        self.CODE_SIZE = CODE_SIZE

    def get_data(self, video_file='data/video.avi', frame_skip=5):

        """ Generate pictures from video, reading every
        frame_skip'th frame and resizing to fit model. """

        print('Generating data from video.')
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        frames_generated = 0
        data = []
        while success:
            y_dim,  x_dim = image.shape[:2]
            dim_diff = x_dim - y_dim
            # get random portion of x dim.
            x_start = np.random.randint(dim_diff)
            image = image[:, x_start:x_start + y_dim, :]
            # resize to fit model
            image = cv2.resize(image, self.IMG_SHAPE[:2])
            # BGR to RGB
            image = image[:, :, ::-1]

            if count % frame_skip == 0:
                data.append(image)
                frames_generated += 1
            count += 1
            success, image = vidcap.read()
        # scale data
        self.data = np.asarray(data)/255.
        print('Data prepared. %d frames generated.' % frames_generated)

    def create_generator(self):

        """ Model to generate images from random noise."""

        generator = Sequential()
        generator.add(L.InputLayer([self.CODE_SIZE], name='noise'))
        generator.add(L.Dense(16 * 16 * 10, activation='elu'))
        generator.add(L.Reshape((16, 16, 10)))

        generator.add(L.Deconv2D(64, kernel_size=(5, 5), activation='elu'))
        generator.add(L.Deconv2D(64, kernel_size=(5, 5), activation='elu'))
        generator.add(L.UpSampling2D(size=(2, 2)))

        generator.add(L.Deconv2D(32, kernel_size=3, activation='elu'))
        generator.add(L.Deconv2D(32, kernel_size=3, activation='elu'))
        generator.add(L.Conv2D(3, kernel_size=3, activation=None))

        self.generator = generator
        print('Generator created successfully.')
        
    def create_discriminator(self):

        """ Model to distinguish real images from generated ones."""

        discriminator = Sequential()
        discriminator.add(L.InputLayer(self.IMG_SHAPE))

        discriminator.add(L.Conv2D(16, kernel_size=(7, 7), padding='same', activation='elu'))
        discriminator.add(L.Conv2D(16, kernel_size=(7, 7), padding='same', activation='elu'))
        discriminator.add(L.AveragePooling2D(strides=2))

        discriminator.add(L.Conv2D(32, kernel_size=(5, 5), padding='same', activation='elu'))
        discriminator.add(L.Conv2D(32, kernel_size=(5, 5), padding='same', activation='elu'))
        discriminator.add(L.AveragePooling2D(strides=2))

        discriminator.add(L.Conv2D(64, kernel_size=(3, 3), padding='same', activation='elu'))
        discriminator.add(L.Conv2D(64, kernel_size=(3, 3), padding='same', activation='elu'))
        discriminator.add(L.AveragePooling2D(strides=2))

        discriminator.add(L.Flatten())
        discriminator.add(L.Dense(256, activation='tanh'))
        discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))

        self.discriminator = discriminator
        print('Discriminator created successfully.')

    def create_tf_objects(self):

        """ Initialize TF objects and start new session. """

        self.s = tf.InteractiveSession()
        self.noise = tf.placeholder('float32', [None, self.CODE_SIZE])
        self.real_data = tf.placeholder('float32', [None, ] + list(self.IMG_SHAPE))
        self.logp_real = self.discriminator(self.real_data)
        self.generated_data = self.generator(self.noise)
        self.logp_gen = self.discriminator(self.generated_data)

        # discriminator training
        self.d_loss = -tf.reduce_mean(self.logp_real[:, 1] + self.logp_gen[:, 0])
        self.d_loss += tf.reduce_mean(self.discriminator.layers[-1].kernel ** 2)  # regularize
        self.disc_optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(self.d_loss, var_list=self.discriminator.trainable_weights)

        # generator training
        self.g_loss = -tf.reduce_mean(self.logp_gen[:, 1])
        self.gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.g_loss, var_list=self.generator.trainable_weights)

        self.s.run(tf.global_variables_initializer());
        print('TF objects created successfully.')

    def sample_noise_batch(self, bsize):

        """ Generate random noise for image generation. """

        return np.random.normal(size=(bsize, self.CODE_SIZE)).astype('float32')

    def sample_data_batch(self, bsize):

        """ Sample images from real data. """

        idxs = np.random.choice(np.arange(self.data.shape[0]), size=bsize)
        return self.data[idxs]

    def sample_images(self, nrow, ncol, epoch, sharp=False):

        """ Generate images from random noise and save to file. """

        images = self.generator.predict(self.sample_noise_batch(bsize=nrow * ncol))
        if np.var(images) != 0:
            images = images.clip(np.min(self.data), np.max(self.data))
        for i in range(nrow * ncol):
            plt.subplot(nrow, ncol, i + 1)
            if sharp:
                plt.imshow(images[i].reshape(IMG_SHAPE), cmap="gray", interpolation="none")
                plt.xticks([], [])
                plt.yticks([], [])
            else:
                plt.imshow(images[i].reshape(IMG_SHAPE), cmap="gray", interpolation="kaiser")
                plt.xticks([], [])
                plt.yticks([], [])
        plt.suptitle('Epochs: ' + str(epoch))
        # save generated images
        sample_dir = 'output'
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        plt.savefig(sample_dir+'/render_epochs_' + str(epoch) + '.png', dpi=200)
        plt.close()

    def train(self, epochs, batch_size, discriminator_steps=5, generator_steps=1):

        """ Perform training steps and get results. """

        for epoch in range(epochs):

            feed_dict = {self.real_data: self.sample_data_batch(batch_size),
                         self.noise: self.sample_noise_batch(batch_size)}

            for d in range(discriminator_steps):
                self.s.run(self.disc_optimizer, feed_dict)
            for g in range(generator_steps):
                self.s.run(self.gen_optimizer, feed_dict)
            g_loss = self.s.run(self.g_loss, feed_dict)
            d_loss = self.s.run(self.d_loss, feed_dict)

            if epoch % 100 == 0:
                self.sample_images(4, 5, self.global_epoch, True)
                print('Images successfully generated.')

            if epoch % 1 == 0:
                print('Epoch: %d' % self.global_epoch,
                      'Generator loss: %.4f' % g_loss,
                      'Discriminator loss: %.4f' % d_loss)
            self.global_epoch += 1


# start the process
gan = GAN()
gan.get_data(video_file='data/simpsons.avi', frame_skip=5)
gan.create_generator()
gan.create_discriminator()
gan.create_tf_objects()
gan.train(epochs=100000, batch_size=100, discriminator_steps=5, generator_steps=1)
