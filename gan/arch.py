import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

BATCH_SIZE = 100
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5

class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon


class VAEGAN(Model):
    def __init__(self, encoder, generator, discriminator, r_loss_factor, **kwargs):
        super(VAEGAN, self).__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.r_loss_factor = r_loss_factor

        self.inner_loss_coef = 1
        self.normal_coef = 0.1
        self.kl_coef = 0.01


    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        
        latent_r = tf.random.normal((BATCH_SIZE, Z_DIM))
        with tf.GradientTape(persistent=True) as tape:
            latent, kl_loss = self.encoder(data)
            fake = self.generator(latent)
            dis_fake, dis_inner_fake = self.discriminator(fake)
            dis_fake_r, _ = self.discriminator(self.generator(latent_r))
            dis_true, dis_inner_true = self.discriminator(data)

            vae_inner = dis_inner_fake - dis_inner_true
            vae_inner = vae_inner * vae_inner

            mean, var = tf.nn.moments(self.encoder(data)[0], axes=0)
            var_to_one = var - 1

            reduced_mean = tf.reduce_mean(mean*mean)
            reduced_var_to_one = tf.reduce_mean(var_to_one*var_to_one)

            normal_loss = reduced_mean + reduced_var_to_one

            kl_loss = tf.reduce_mean(kl_loss)
            vae_diff_loss = tf.reduce_mean(vae_inner)

            f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
            r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
            t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
            gan_loss = (0.5*t_dis_loss + 0.25*f_dis_loss + 0.25*r_dis_loss)
            vae_loss = tf.reduce_mean(tf.abs(data-fake)) 
            
            E_loss = vae_diff_loss + self.kl_coef*kl_loss + self.normal_coef*normal_loss
            G_loss = self.inner_loss_coef*vae_diff_loss - gan_loss
            D_loss = gan_loss

        E_grad = tape.gradient(E_loss, self.encoder.trainable_variables)
        G_grad = tape.gradient(G_loss, self.generator.trainable_variables)
        D_grad = tape.gradient(D_loss, self.discriminator.trainable_variables)
        
        del tape

        self.optimizer.apply_gradients(zip(E_grad, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(G_grad, self.generator.trainable_variables))
        self.optimizer.apply_gradients(zip(D_grad, self.discriminator.trainable_variables))

        # with tf.GradientTape() as tape:
        #     z_mean, z_log_var, z = self.encoder(data)
        #     reconstruction = self.decoder(z)
        #     reconstruction_loss = tf.reduce_mean(
        #         tf.square(data - reconstruction), axis = [1,2,3]
        #     )
        #     reconstruction_loss *= self.r_loss_factor
        #     kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        #     kl_loss = tf.reduce_sum(kl_loss, axis = 1)
        #     kl_loss *= -0.5
        #     total_loss = reconstruction_loss + kl_loss
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "e_loss": E_loss,
            "g_loss": G_loss,
            "d_loss": D_loss
        }
        # return {
        #     "loss": total_loss,
        #     "reconstruction_loss": reconstruction_loss,
        #     "kl_loss": kl_loss,
        # }
    
    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)



class GAN():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE
    
    def _build(self):
        # ENCODER
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM, name='mu', activation='tanh')(vae_z_in)
        vae_z_log_var = Dense(Z_DIM, name='log_var', activation='tanh')(vae_z_in)

        vae_z = Sampling(name='z')([vae_z_mean, vae_z_log_var])
        
        kl_loss = 1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var)
        kl_loss = K.mean(kl_loss, axis=-1)

        # GENERATOR: 
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        vae_dense = Dense(1024, name='dense_layer')(vae_z_input)
        vae_unflatten = Reshape((1,1,DENSE_SIZE), name='unflatten')(vae_dense)
        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0], name='deconv_layer_1')(vae_unflatten)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1], name='deconv_layer_2')(vae_d1)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2], name='deconv_layer_3')(vae_d2)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='deconv_layer_4')(vae_d3)
        
        ### DISCRIMINATOR
        disc_input = Input(shape=INPUT_DIM)
        disc_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(disc_input)
        disc_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(disc_c1)
        disc_c3 = Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(disc_c2)
        disc_c4 = Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(disc_c3)
        disc_inner_output = Flatten()(disc_c4)

        disc_flat = Flatten()(disc_c4)
        disc_d1 = Dense(DENSE_SIZE)(disc_flat)
        disc_output = Dense(1)(disc_d1)

        #### MODELS
        encoder = Model(vae_x, [vae_z, kl_loss], name = 'encoder')
        generator = Model(vae_z_input, vae_d4, name = 'decoder')
        discriminator = Model(disc_input, [disc_output, disc_inner_output], name='discriminator')

        gan_full = VAEGAN(encoder, generator, discriminator, 10000)

        opti = Adam(lr=LEARNING_RATE)
        gan_full.compile(optimizer=opti)
        
        return (gan_full, encoder, generator, discriminator)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):
        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE)
        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
