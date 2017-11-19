
# coding: utf-8

# In[1]:
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import tensorflow as tf
# from tensorflow.contrib import layers
from image_models import BIGAN as Model
import ops


# In[2]:

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = False
run_config.allow_soft_placement = True


# In[3]:

sess = tf.InteractiveSession(config=run_config)


# In[4]:

# Configs

# # [1] Cifar10
# dataset_name = 'cifar10'
# data_dir = 'path to cifar10/'
# config = {'model_name': 'BIGAN_cifar',
#           'ext': 'jpg',
#           's': 32,
#           'c': 3,
#           'lvls': 3,
#           'kernel_size': [4, 4],
#           'z_len': 128,
#           'gf_dim': 64,
#           'df_dim': 64,
#           'interpolations': 4,
#           'sampler_batch_size': 100,
#           'activation_fn': ops.lrelu,
#           'out_activation_fn': tf.nn.tanh,
#           'g_bn': True,
#           'd_bn': False,
#           'lr': 1e-4,
#           'optimizer': 'adam',
#           'loss': 'lol1',
#           'grad_pen': True,
#           'ae_pen': False,
#           'gp_lambda': 1.,
#           'ae_lambda': 10,
#           'add_image_summary': False}

# [2] CelebA
dataset_name = 'celeb_a'
data_dir = 'path to cifar10/'
config = {'model_name': 'BIGAN',
          'ext': 'jpg',
          's': 128,
          'c': 3,
          'lvls': 5,
          'kernel_size': [4, 4],
          'z_len': 128,
          'gf_dim': 64,
          'df_dim': 64,
          'interpolations': 4,
          'sampler_batch_size': 100,
          'activation_fn': ops.lrelu,
          'out_activation_fn': tf.nn.tanh,
          'g_bn': True,
          'd_bn': False,
          'lr': 1e-4,
          'optimizer': 'adam',
          'loss': 'lol2',
          'grad_pen': True,
          'ae_pen': False,
          'gp_lambda': 1,
          'ae_lambda': 10,
          'add_image_summary': False}

# # [3] Traffic
# dataset_name = 'traffic'
# data_dir = 'path to dataset/'
# config = {'ext': 'png',
#           's': 128,
#           'c': 3,
#           'lvls': 5,
#           'kernel_size': [4, 4],
#           'z_len': 128,
#           'gf_dim': 64,
#           'df_dim': 64,
#           'interpolations': 4,
#           'sampler_batch_size': 100,
#           'activation_fn': ops.lrelu,
#           'out_activation_fn': tf.nn.tanh,
#           'g_bn': True,
#           'd_bn': False,
#           'lr': 1e-4,
#           'optimizer': 'adam',
#           'loss': 'v',
#           'grad_pen': False,
#           'ae_pen': False,
#           'gp_lambda': 1.,
#           'ae_lambda': 10,
#           'add_image_summary': False}

# In[5]:

model = Model(sess, dataset_name, 64, 4, config, data_dir, '/new_data/gpu/shashank/outputs', var_summ=False, grad_summ=False, send_email=False)

# In[6]:

model.build()


# In[7]:

# load or init all vars
# model.load()
sess.run(tf.global_variables_initializer())
model.train(160, st_epoch=1, samples_per_epoch=1, saves_per_epoch=1)


# In[8]:

# # Interpolations:
# import utils
# filepaths = ['./results/celeb_a/recons/{}.png'.format(i) for i in range(460)]
# utils.get_interpolations(sess, model, filepaths, 8, './results/celeb_a/')

# # Calculate inception score:
# import numpy as np
# print('Calculating 50k inception score...')
# model.get_inception_score(np.random.uniform(-1., 1., [50000, model.z_len]))
# model.get_inception_score_diff(model._get_cifar_batch('test_batch'))

# In[9]:

sess.close()
