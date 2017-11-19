
# coding: utf-8

# In[1]:

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
# from tensorflow.contrib import layers
from visual_models import vizBIGAN as Model
import ops


# In[2]:

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = False
run_config.allow_soft_placement = True


# In[3]:

sess = tf.InteractiveSession(config=run_config)


# In[4]:

# Configs

# [1] BIGAN
dataset = 'rect'
config = {'ext': 'jpg',
          'lvls': 5,
          'x_len': 2,
          'z_len': 1,
          'gf_dim': 512,
          'df_dim': 512,
          'sampler_batch_size': 360,
          'activation_fn': ops.lrelu,
          'out_activation_fn': tf.nn.tanh,
          'd_bn': False,
          'g_bn': False,
          'lr': 1e-4,
          'optimizer': 'adam',
          'loss': 'lol1',
          'grad_pen': True,
          'ae_pen': False,
          'gp_lambda': .1,
          'ae_lambda': .1}

# # [1] DCGAN
# dataset = 'swiss'
# config = {'ext': 'jpg',
#           'lvls': 5,
#           'x_len': 2,
#           'z_len': 2,
#           'gf_dim': 512,
#           'df_dim': 512,
#           'sampler_batch_size': 360,
#           'activation_fn': ops.lrelu,
#           'out_activation_fn': tf.nn.tanh,
#           'd_bn': False,
#           'g_bn': False,
#           'lr': 1e-4,
#           'optimizer': 'adam',
#           'loss': 'lol2',
#           'grad_pen': True,
#           'ae_pen': False,
#           'gp_lambda': .1,
#           'ae_lambda': .1}


# In[5]:

model = Model(sess, dataset, 256, 1, config, None, var_summ=False, grad_summ=False, send_email=False)


# In[6]:

model.build()


# In[7]:

# load or init all vars
sess.run(tf.global_variables_initializer())
# model.load()
model.train(200, st_epoch=1, samples_per_epoch=1, saves_per_epoch=.1)


# In[8]:

sess.close()
