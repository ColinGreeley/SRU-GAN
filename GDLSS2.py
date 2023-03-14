import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG19, EfficientNetB2, InceptionV3
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras import layers, Model, optimizers, losses, models, activations
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
#from imgaug import augmenters
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tqdm import tqdm
import cv2
import sys
import os
import gc

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
instance_num = 75


def get_data():
    dir = "C:/Users/Greeley/python/Data/ImageNet/download/train"
    n = instance_num
    HR_images = []
    LR_images = []
    labels = np.zeros((n*1000,), dtype="int32")
    max_size = 450
    min_size = 200
    for i, im_dir in enumerate(os.listdir(dir)):
        class_dir = os.path.join(dir, im_dir)
        j = 0
        for im in os.listdir(class_dir):
            if j >= instance_num:
                break
            new_im = Image.open(os.path.join(class_dir, im))
            if new_im.size[1] > min_size and new_im.size[0] > min_size:
                x_diff = int((min(max_size, new_im.size[0]) // 16) * 16)
                y_diff = int((min(max_size, new_im.size[1]) // 16) * 16)
                #x_diff = 384
                #y_diff = 384
                HR_images.append(np.asarray(new_im.resize((x_diff, y_diff), resample=Image.Resampling.LANCZOS).convert("RGB")).astype("uint8"))
                LR_images.append(np.asarray(new_im.resize((x_diff//2, y_diff//2), resample=Image.Resampling.LANCZOS).convert("RGB")).astype("uint8"))
                labels[i*n + j] = i
                j += 1
    print("Images:", len(HR_images))
    return (HR_images, LR_images), labels


class SRU_GAN:
    
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        if len(physical_devices) > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:{}".format(i) for i in range(len(physical_devices))])
            with mirrored_strategy.scope():
                self.g_optimizer = optimizers.Adam(PiecewiseConstantDecay(boundaries=[500_000], values=[1e-4, 1e-5]))#, clipvalue=1.0)
                self.d_optimizer = optimizers.Adam(PiecewiseConstantDecay(boundaries=[500_000], values=[1e-4, 1e-5]))#, clipvalue=1.0)
                vgg = VGG19(input_shape=(None, None, 3), include_top=False)
                self.vgg = Model(vgg.input, vgg.layers[20].output)
                self.generator = self.make_generator()
                self.discriminator = self.make_discriminator()
        
                self.avg_generator = self.make_generator()
                self.update_target(self.avg_generator.weights, self.generator.weights, 1.0)
                self.avg_discriminator = self.make_discriminator()
                self.update_target(self.avg_discriminator.weights, self.discriminator.weights, 1.0)
        else:
            self.g_optimizer = optimizers.Adam(PiecewiseConstantDecay(boundaries=[500_000], values=[1e-4, 1e-5]))#, clipvalue=1.0)
            self.d_optimizer = optimizers.Adam(PiecewiseConstantDecay(boundaries=[500_000], values=[1e-4, 1e-5]))#, clipvalue=1.0)
            vgg = VGG19(input_shape=(None, None, 3), include_top=False)
            self.vgg = Model(vgg.input, vgg.layers[20].output)
            self.generator = self.make_generator()
            self.discriminator = self.make_discriminator()
            
            self.avg_generator = self.make_generator()
            self.update_target(self.avg_generator.weights, self.generator.weights, 1.0)
            self.avg_discriminator = self.make_discriminator()
            self.update_target(self.avg_discriminator.weights, self.discriminator.weights, 1.0)
        
            
    def normalize(self, x, rgb_mean=DIV2K_RGB_MEAN):
        return (x - rgb_mean) / 127.5

    def denormalize(self, x, rgb_mean=DIV2K_RGB_MEAN):
        return x * 127.5 + rgb_mean

    def normalize_01(self, x):
        """Normalizes RGB images to [0, 1]."""
        return x / 255.0

    def normalize_m11(self, x):
        """Normalizes RGB images to [-1, 1]."""
        return x / 127.5 - 1

    def denormalize_m11(self, x):
        """Inverse of normalize_m11."""
        return (x + 1) * 127.5

    def load(self):
        im = np.zeros((1, 256, 256, 3))
        self.generator(im)
        self.discriminator(im)
        self.avg_generator(im)
        self.avg_discriminator(im)
        self.generator.load_weights("Generator_weights1.h5")
        self.avg_generator.load_weights("Generator_avg_weights1.h5")
        self.discriminator.load_weights("Discriminator_weights1.h5")
        #self.update_target(self.avg_generator.weights, self.generator.weights, 1.0)
        #self.update_target(self.avg_discriminator.weights, self.discriminator.weights, 1.0)
        #self.d_optimizer = optimizers.Adam(1e-5)
        #self.g_optimizer = optimizers.Adam(1e-5)
        
    def AttentionModule(self, name="AttentionModule"):
        """Attention module"""
        def channel_attention(x):
            squeeze = layers.Dense(x.shape[-1]//8)
            excitation = layers.Dense(x.shape[-1])
            av_pool = layers.GlobalAveragePooling2D()(x)
            mx_pool = layers.GlobalMaxPooling2D()(x)
            av_sq = squeeze(av_pool)
            av_sq = layers.Activation("swish")(av_sq)
            mx_sq = squeeze(mx_pool)
            mx_sq = layers.Activation("swish")(mx_sq)
            av_ex = excitation(av_sq)
            mx_ex = excitation(mx_sq)
            pool = layers.Add()([av_ex, mx_ex])
            attention_scores = layers.Activation("sigmoid")(pool)
            attention_scores = layers.Reshape((1,1,x.shape[-1]))(attention_scores)
            return layers.multiply([x, attention_scores])
        def spatial_attention(x):
            avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
            max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
            feature_map = layers.Concatenate(axis=3)([avg_pool, max_pool])
            attention_scores = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(feature_map)
            return layers.multiply([x, attention_scores])
        def attention_module(x_in):
            x = channel_attention(x_in)
            x = spatial_attention(x)
            return x
        return attention_module
    
    def DenseBlock(self, width):
        def apply(x):
            if x.shape[-1] != width:
                residual = layers.Conv2D(width, kernel_size=1)(x)
            else:
                residual = x
            x1 = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
            x1 = layers.Concatenate()([residual, x1])
            x2 = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x1)
            x2 = layers.Concatenate()([x1, x2])
            x3 = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x2)
            x = layers.Add()([residual, x3])
            return x
        return apply

    def ResBlock(self, width):
        def apply(x):
            if x.shape[-1] != width:
                residual = layers.Conv2D(width, kernel_size=1)(x)
            else:
                residual = x
            x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
            x = layers.Add()([residual, x])
            return x
        return apply

    def make_generator(self):
        res = []
        res_depth = 2
        x_in = layers.Input((None, None, 3))            # x
        x = layers.Lambda(self.normalize_01)(x_in)
        x = layers.Conv2D(64, 7, padding="same", activation="swish")(x)
        for _ in range(res_depth):
            x = self.ResBlock(64)(x)
        x = self.AttentionModule()(x)
        res.append(x)
        x = layers.MaxPooling2D()(x)                    # x/2
        
        for _ in range(res_depth):
            x = self.ResBlock(96)(x)
        x = self.AttentionModule()(x)
        res.append(x)
        x = layers.MaxPooling2D()(x)                    # x/4
        
        for _ in range(res_depth):
            x = self.ResBlock(128)(x)
        x = self.AttentionModule()(x)
        res.append(x)
        x = layers.MaxPooling2D()(x)                    # x/8
        
        for _ in range(res_depth):
            x = self.ResBlock(256)(x)
        x = layers.Conv2D(512, kernel_size=3, padding="same", activation="swish")(x)
        x = self.AttentionModule()(x)
        x = tf.nn.depth_to_space(x, block_size=2)       # x/4
        x = layers.Concatenate()([x, res.pop()]) 
        
        for _ in range(res_depth):
            x = self.ResBlock(128)(x)
        x = layers.Conv2D(384, kernel_size=3, padding="same", activation="swish")(x)
        x = self.AttentionModule()(x)
        x = tf.nn.depth_to_space(x, block_size=2)       # x/2
        x = layers.Concatenate()([x, res.pop()]) 
        
        for _ in range(res_depth):
            x = self.ResBlock(96)(x)
        x = layers.Conv2D(256, kernel_size=3, padding="same", activation="swish")(x)
        x = self.AttentionModule()(x)
        x = tf.nn.depth_to_space(x, block_size=2)       # x
        x = layers.Concatenate()([x, res.pop()])  
        
        for _ in range(16):
            x = self.DenseBlock(64)(x)
            
        x = layers.Conv2D(256, kernel_size=3, padding='same', activation="swish")(x)
        x = self.AttentionModule()(x)
        x = tf.nn.depth_to_space(x, block_size=2)       # 2x
        #x = layers.Conv2D(256, kernel_size=3, padding='same', activation="swish")(x)
        #x = self.AttentionModule()(x)
        #x = tf.nn.depth_to_space(x, block_size=2)       # 4x
        
        x = layers.Conv2D(3, 7, activation="tanh", padding="same")(x)
        x = layers.Lambda(self.denormalize_m11)(x)
        generator = Model(x_in, x)
        generator.summary()
        return generator
        
    def make_discriminator(self):
        
        att_blocks = 0
        widths = [64, 64, 128, 256, 512, 512]
        block_depth = 1
        skips = []
        
        x_in = layers.Input((None, None, 3))
        x = layers.Lambda(self.normalize_m11)(x_in)
        x = layers.Conv2D(widths[0], kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        for width in widths:
            residual = layers.Conv2D(width, kernel_size=1)(x)
            x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.Add()([x, residual])
            x = layers.AveragePooling2D()(x)
        
        x = layers.Conv2D(widths[-1]*2, kernel_size=3, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        #x = layers.Dropout(0.3)(x)
        validity = layers.Dense(1, activation="sigmoid")(x)
        label = layers.Dense(1000, activation="softmax", name="label")(x)
        discriminator = Model(x_in, [validity, label], name='Discriminator')
        
        #discriminator = Model(x_in, x)
        discriminator.summary()
        return discriminator
        
    def train_generator(self, augment=False):
        while True:
            if self.batch_size == 1:
                idxs = np.random.choice(len(self.HR_images), self.batch_size)
                Y = np.array([self.HR_images[idx] for idx in idxs])
                X = np.array([self.LR_images[idx] for idx in idxs])
                labels = np.array(self.labels [idxs])
                yield tf.constant(X), [tf.constant(Y), tf.constant(labels)]
            else:
                idxs = np.random.choice(len(self.HR_images), self.batch_size)
                Y = np.array([self.HR_images[idx] for idx in idxs], dtype="object")
                X = np.array([self.LR_images[idx] for idx in idxs], dtype="object")
                labels = np.array(self.labels [idxs])
                yield tf.ragged.constant(X).to_tensor(), [tf.ragged.constant(Y).to_tensor(), tf.constant(labels)]
    
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
            
    @tf.function
    def add_weights(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(a + b)
             
    @tf.function
    def divide_weights(self, target_weights, tau):
        for a in target_weights:
            a.assign(a / tau)
    
    #@tf.function
    def update(self, X, Y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_tape.watch(self.discriminator.trainable_variables)
            x = tf.cast(X, tf.float32)
            y = tf.cast(Y[0], tf.float32)
            labels = tf.cast(Y[1], tf.float32)
            #s2 = self.generator(x, training=True)
            gen_image = self.generator(x, training=True)
            
            hr_output, pred_hr_label = self.discriminator(y, training=True)
            sr_output, pred_sr_label = self.discriminator(gen_image, training=True)
            hr = vgg19_preprocess(y)
            sr = vgg19_preprocess(gen_image)
            sr_features = self.vgg(sr) / 12.75
            hr_features = self.vgg(hr) / 12.75
            con_loss = losses.MeanSquaredError()(hr_features, sr_features) + losses.MeanSquaredError()(y/127.5-1, gen_image/127.5-1)
            class_loss_gen = 1e-3 * losses.SparseCategoricalCrossentropy()(labels, pred_sr_label)
            gen_loss = 1e-3 * losses.BinaryCrossentropy()(tf.ones_like(sr_output), sr_output)
            perc_loss = con_loss + gen_loss + class_loss_gen
            zero_labels = tf.zeros_like(sr_output)
            one_labels = tf.ones_like(hr_output)
            validity_loss = losses.BinaryCrossentropy()(one_labels, hr_output) + losses.BinaryCrossentropy()(zero_labels, sr_output)
            class_loss = losses.SparseCategoricalCrossentropy()(labels, pred_hr_label) + losses.SparseCategoricalCrossentropy()(labels, pred_sr_label)
            disc_loss = [validity_loss, class_loss]
        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return (validity_loss, class_loss, gen_loss, con_loss, class_loss_gen)
    
    def train(self, images, labels):
        self.HR_images = images[0]
        self.LR_images = images[1]
        self.labels = labels
        self.step = 0
        self.update_step = 1
        train_gen = self.train_generator(False)
        while self.step < 1_000_000:
            x, y = next(train_gen)
            if self.step % 10000 == 0:
                self.update_step = 1
                plt.figure(figsize=(45,15))
                #plt.subplot(1,2,1)
                #plt.imshow((x[0] + 1) / 2)
                plt.subplot(1,3,1)
                plt.imshow(np.asarray(x[0]) / 255.)
                plt.subplot(1,3,2)
                #x1 = self.generator(x)
                plt.imshow(np.clip(self.avg_generator(x)[0] / 255., 0, 1))
                plt.subplot(1,3,3)
                plt.imshow(np.asarray(y[0][0]) / 255.)
                #plt.title(y[0][1])
                plt.savefig("figures/" + str(self.step) + ".png")
                plt.close()
                #plt.show()
                self.generator.save_weights("Generator_weights1.h5")
                self.avg_generator.save_weights("Generator_avg_weights1.h5")
                self.discriminator.save_weights("Discriminator_weights1.h5")
                #del x1
            validity_loss, class_loss, gen_loss, con_loss, class_loss_gen = self.update(x, y)
            print ("%d [D loss: %f; L loss: %f] [G loss: %f; C loss: %f; L loss: %f]" % (self.step, validity_loss, class_loss, gen_loss, con_loss, class_loss_gen))
            #self.add_weights(self.avg_generator.weights, self.generator.weights)
            #self.add_weights(self.avg_discriminator.weights, self.discriminator.weights)
            #self.update_step += 1
            self.update_target(self.avg_generator.weights, self.generator.weights, 0.01)
            #tf.keras.backend.clear_session()
            #gc.collect()
            self.step += 1
                
    
    
if __name__ == "__main__":
    
    batch_size = 1
    
    images, labels = get_data()
    classifier = SRU_GAN(batch_size=batch_size)
    classifier.load()
    classifier.train(images, labels)
    