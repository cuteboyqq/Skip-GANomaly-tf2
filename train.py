import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
import os
import time
import numpy as np
import cv2
from model import Skip_GANomaly

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "buffer size for pseudo shuffle")
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("isize", 64, "input size")
flags.DEFINE_string("ckpt_dir", 'ckpt', "checkpoint folder")
flags.DEFINE_integer("nz", 100, "latent dims")
flags.DEFINE_integer("nc", 3, "input channels")
flags.DEFINE_integer("ndf", 64, "number of discriminator's filters")
flags.DEFINE_integer("ngf", 64, "number of generator's filters")
flags.DEFINE_integer("extralayers", 0, "extralayers for both G and D")
flags.DEFINE_list("encdims", None, "Layer dimensions of the encoder and in reverse of the decoder."
                                   "If given, dense encoder and decoders are used.")
flags.DEFINE_integer("niter",50,"number of training epochs")
flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("w_adv", 1., "Adversarial loss weight")
flags.DEFINE_float("w_con", 50., "Reconstruction loss weight")
flags.DEFINE_float("w_enc", 1., "Encoder loss weight")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_string("dataset", r'C:\factory_data\2022-08-26\f_384_2min\crops_1cls', "name of dataset")
#flags.DEFINE_string("dataset", 'cifar10', "name of dataset")
flags.DEFINE_string("dataset_test", r'C:\factory_data\2022-08-26\f_384_2min\crops_2cls', "name of dataset")
flags.DEFINE_string("dataset_infer", r'C:\factory_data\2022-08-26\f_384_2min\crops_1cls', "name of dataset")
flags.DEFINE_string("dataset_infer_abnormal", r'C:\factory_data\2022-08-26\f_384_2min\defeat', "name of dataset")
DATASETS = ['mnist', 'cifar10']
'''
flags.register_validator('dataset',
                         lambda name: name in DATASETS,
                         message='--dataset must be {}'.format(DATASETS))
'''
flags.DEFINE_integer("anomaly", 5, "the anomaly idx")
flags.mark_flag_as_required('anomaly')
flags.mark_flag_as_required('isize')
flags.mark_flag_as_required('nc')

def batch_resize(imgs, size: tuple):
    img_out = np.empty((imgs.shape[0], ) + size)
    for i in range(imgs.shape[0]):
        img_out[i] = cv2.resize(imgs[i], size, interpolation=cv2.INTER_CUBIC)
    return img_out

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def main(_):
    show_img = False
    TRAIN = True
    opt = FLAGS
    # logging
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
    
    # dataset
    if opt.dataset=='mnist':
        data_train, data_test = tf.keras.datasets.mnist.load_data()
        x_train, y_train = data_train
        x_test, y_test = data_test
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.reshape([-1,])
        y_test = y_test.reshape([-1,])
        # resize to (32, 32)
        if opt.dataset=='mnist':
            x_train = batch_resize(x_train, (32, 32))[..., None]
            x_test = batch_resize(x_test, (32, 32))[..., None]
        # normalization
        mean = x_train.mean()
        stddev = x_train.std()
        x_train = (x_train - mean) / stddev
        x_test = (x_test - mean) / stddev
        logging.info('{}, {}'.format(x_train.shape, x_test.shape))
        # define abnoraml data and normal
        # training data only contains normal
        x_train = x_train[y_train != opt.anomaly, ...]
        y_train = y_train[y_train != opt.anomaly, ...]
        y_test = (y_test == opt.anomaly).astype(np.float32)
        # tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
            opt.batch_size, drop_remainder=True)
        test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)
        test_dataset = test_dataset.shuffle(buffer_size=len(y_test))
    elif opt.dataset=='cifar10':
        data_train, data_test = tf.keras.datasets.cifar10.load_data()
        x_train, y_train = data_train
        x_test, y_test = data_test
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.reshape([-1,])
        y_test = y_test.reshape([-1,])
        # resize to (32, 32)
        if opt.dataset=='mnist':
            x_train = batch_resize(x_train, (32, 32))[..., None]
            x_test = batch_resize(x_test, (32, 32))[..., None]
        # normalization
        mean = x_train.mean()
        stddev = x_train.std()
        x_train = (x_train - mean) / stddev
        x_test = (x_test - mean) / stddev
        logging.info('{}, {}'.format(x_train.shape, x_test.shape))
        # define abnoraml data and normal
        # training data only contains normal
        x_train = x_train[y_train != opt.anomaly, ...]
        y_train = y_train[y_train != opt.anomaly, ...]
        y_test = (y_test == opt.anomaly).astype(np.float32)
        # tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
            opt.batch_size, drop_remainder=True)
        test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)
        test_dataset = test_dataset.shuffle(buffer_size=len(y_test))
    else:
        '''
        Use Custom dataaset
        tf.keras.utils.image_dataset_from_directory(
                                        directory,
                                        labels='inferred',
                                        label_mode='int',
                                        class_names=None,
                                        color_mode='rgb',
                                        batch_size=32,
                                        image_size=(256, 256),
                                        shuffle=True,
                                        seed=None,
                                        validation_split=None,
                                        subset=None,
                                        interpolation='bilinear',
                                        follow_links=False,
                                        crop_to_aspect_ratio=False,
                                        **kwargs
                                    )

        '''
        #raise NotImplementError
        train_data_dir = opt.dataset
        img_height = opt.isize
        img_width = opt.isize
        batch_size = opt.batch_size
        train_dataset = tf.keras.utils.image_dataset_from_directory(
          train_data_dir,
          validation_split=0.1,
          subset="training",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size)
        
        train_dataset = train_dataset.map(process)
        
        if TRAIN == False:
            if show_img==True:
                batch_size_=opt.batch_size
                shuffle=True
            else:
                batch_size_=1
                shuffle=False
        else:
            batch_size_=opt.batch_size
            shuffle=True
            
        val_data_dir = opt.dataset_test
        print(val_data_dir)
        print(batch_size_)
        test_dataset = tf.keras.utils.image_dataset_from_directory(
          val_data_dir,
          validation_split=0.1,
          subset="validation",
          shuffle=shuffle,
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
        
        test_dataset = test_dataset.map(process)
        
        infer_data_dir = opt.dataset_infer
        print(infer_data_dir)
        print(batch_size_)
        infer_dataset = tf.keras.utils.image_dataset_from_directory(
          infer_data_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
        
        infer_dataset = infer_dataset.map(process)
        
        infer_data_abnormal_dir = opt.dataset_infer_abnormal
        print(infer_data_abnormal_dir)
        print(batch_size_)
        infer_dataset_abnormal = tf.keras.utils.image_dataset_from_directory(
          infer_data_abnormal_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
    
        infer_dataset_abnormal = infer_dataset_abnormal.map(process)
    
    skip_ganomaly = Skip_GANomaly(opt,
                        train_dataset,
                        valid_dataset=None,
                        test_dataset=test_dataset)
    if TRAIN:
        # training
        skip_ganomaly.fit(opt.niter)
    
        # evaluating
        skip_ganomaly.evaluate_best(test_dataset)
    else:
        if show_img:
            SHOW_MAX_NUM = 5
        else:
            SHOW_MAX_NUM = 1900
        positive_loss = skip_ganomaly.infer(infer_dataset,SHOW_MAX_NUM,show_img,'normal')
        defeat_loss = skip_ganomaly.infer(infer_dataset_abnormal,SHOW_MAX_NUM,show_img,'abnormal')
        if not show_img:
            skip_ganomaly.plot_loss_distribution( SHOW_MAX_NUM,positive_loss,defeat_loss)
    #print(loss_list)
    #print(loss_abnormal_list)
if __name__ == '__main__':
    app.run(main)
