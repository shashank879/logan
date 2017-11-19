""" Implementations of various 'Generative adversarial networks'
"""

import tensorflow as tf
import ops, utils, input_pipe
from tensorflow.contrib import layers
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception
import os, time, random
import numpy as np
import cv2
from base_models import _multiGPUmodel
import inception_score, pickle, csv
from collections import OrderedDict
from skimage.measure import compare_ssim as ssim


SUMMARIZE_AFTER = 5


class DCGAN(_multiGPUmodel):
    """ 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks'
        https://arxiv.org/abs/1511.06434
    """

    def _extra_init(self):
        super()._extra_init()
        # self.w_init = tf.random_uniform_initializer(-np.sqrt(3) * .02, np.sqrt(3) * .02)
        # self.w_init = tf.uniform_unit_scaling_initializer(1.43)

    def _setup_config(self, config):
        self.ext = config['ext']
        self.s = config['s']
        self.c = config['c']
        self.lvls = config['lvls']
        self.kernel_size = config['kernel_size']
        self.z_len = config['z_len']
        self.gf_dim = config['gf_dim']
        self.df_dim = config['df_dim']
        self.sampler_interpolations = config['interpolations']
        self.sampler_batch_size = config['sampler_batch_size']
        self.activation_fn = config['activation_fn']
        self.out_activation_fn = config['out_activation_fn']
        d_bn = config['d_bn']
        g_bn = config['g_bn']
        self.lr = config['lr']
        self.optimizer = config['optimizer']
        self.loss = config['loss']
        self.grad_pen = config['grad_pen']
        self.ae_pen = config['ae_pen']
        self.gp_lambda = config['gp_lambda']
        self.ae_lambda = config['ae_lambda']
        self.add_image_summary = config['add_image_summary']

        if self.optimizer is 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9)
        elif self.optimizer is 'rms':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)

        self.d_bn_fn = layers.batch_norm if d_bn else None
        self.g_bn_fn = layers.batch_norm if g_bn else None
        self.comments = None

        try:
            # Add mean image:
            img = cv2.imread(os.path.join(self.data_dir, 'mean.{}'.format(self.ext)))
            mean_img = np.array(img.resize([self.s, self.s]))
            self.mean_img = tf.constant(ops.transform(mean_img), tf.float32, [1, self.s, self.s, 3])
        except:
            pass

    def _setup_placeholders(self):
        # Create placeholders
        with tf.name_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.s, self.s, self.c], name="X")
            self.Z = tf.placeholder(tf.float32, shape=[None, self.z_len], name="Z")
            self.EPSILON = tf.placeholder(tf.float32, shape=[None, 1], name="EPSILON")
        return [self.X, self.Z, self.EPSILON]

    def _get_serializable_config(self):
        config = OrderedDict([('dataset_name', self.dataset_name),
                              ('model_name', self.model_name),
                              ('batch_size', self.batch_size),
                              ('activation_fn', self.activation_fn.__name__),
                              ('out_activation_fn', self.out_activation_fn.__name__),
                              ('g_bn', str(self.g_bn_fn)),
                              ('d_bn', str(self.d_bn_fn)),
                              ('optimizer', self.optimizer),
                              ('lr', str(self.lr)),
                              ('lvls', str(self.lvls)),
                              ('kernel_size', str(self.kernel_size)),
                              ('s', str(self.s)),
                              ('c', str(self.c)),
                              ('z_len', str(self.z_len)),
                              ('gf_dim', self.gf_dim),
                              ('df_dim', self.df_dim),
                              ('loss', self.loss),
                              ('grad_pen', self.grad_pen),
                              ('ae_pen', self.ae_pen),
                              ('gp_lambda', self.gp_lambda),
                              ('ae_lambda', self.ae_lambda),
                              ('comments', str(self.comments))])
        return config

    def _generator(self, z, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope('Generator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = int(self.s / 2**self.lvls)
            c = int(self.gf_dim * 2**(min(self.lvls - 1, 3)))
            normalizer_params = {'is_training': is_training, 'decay': 0.9, 'scale': True} if self.g_bn_fn else None
            h0 = layers.fully_connected(z, s * s * c,
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.g_bn_fn,
                                        normalizer_params=normalizer_params,
                                        # weights_initializer=self.w_init,
                                        scope="h0")
            print(h0.name, h0.get_shape())
            x_hat = tf.reshape(h0, shape=[-1, s, s, c], name='h1')
            print(x_hat.name, x_hat.get_shape())
            for l in range(self.lvls - 1, 0, -1):
                c = int(self.gf_dim * 2**(min(l - 1, 3)))
                x_hat = layers.conv2d_transpose(x_hat, c,
                                                kernel_size=self.kernel_size,
                                                stride=2,
                                                activation_fn=self.activation_fn,
                                                normalizer_fn=self.g_bn_fn,
                                                normalizer_params=normalizer_params,
                                                # weights_initializer=self.w_init,
                                                scope="g{}".format(l))
                print(x_hat.name, x_hat.get_shape())
            x_hat = layers.conv2d_transpose(x_hat, self.c,
                                            kernel_size=self.kernel_size,
                                            stride=2,
                                            activation_fn=None,
                                            # weights_initializer=self.w_init,
                                            scope="g0")
            print(x_hat.name, x_hat.get_shape())

            # x_hat = self.out_activation_fn(x_hat + self.mean_img)
            x_hat = self.out_activation_fn(x_hat)
        return x_hat

    def _discriminator(self, x, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            d = x
            normalizer_params = {'is_training': is_training, 'decay': 0.9, 'scale': True} if self.d_bn_fn else None
            for l in range(self.lvls):
                if d.get_shape()[1] <= self.kernel_size[0]:
                    break
                d = layers.conv2d(d, (2**l) * self.df_dim,
                                  kernel_size=self.kernel_size,
                                  stride=2,
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.d_bn_fn if l > 0 else None,
                                  normalizer_params=normalizer_params if l > 0 else None,
                                #   weights_initializer=self.w_init,
                                  scope="d{}".format(l))
                print(d.name, d.get_shape())
            shape = d.get_shape().as_list()
            s = shape[1]
            n = (2**l) * self.df_dim
            d = layers.conv2d(d, n,
                              kernel_size=[s, s],
                              padding='VALID',
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.d_bn_fn,
                              normalizer_params=normalizer_params,
                            #   weights_initializer=self.w_init,
                              scope="d{}".format(l + 1))
            print(d.name, d.get_shape())
            d = tf.reshape(d, shape=[-1, n], name='d_squeeze')
            print(d.name, d.get_shape())
            d = layers.fully_connected(d, n,
                                       activation_fn=self.activation_fn,
                                       normalizer_fn=self.d_bn_fn,
                                       normalizer_params=normalizer_params,
                                    #    weights_initializer=self.w_init,
                                       scope="d{}".format(l + 2))
            print(d.name, d.get_shape())
            d = layers.fully_connected(d, 1,
                                       activation_fn=None,
                                       normalizer_fn=self.d_bn_fn,
                                       normalizer_params=normalizer_params,
                                    #    weights_initializer=self.w_init,
                                       scope="d{}".format(l + 3))
            print(d.name, d.get_shape())
        return d

    def _full_conv(self, x, dim, bn_fn=None, get_layer=None, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope("full_conv", reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = x.get_shape()[1]
            h = x
            i = 0
            normalizer_params = {'is_training': is_training, 'decay': 0.9, 'scale': True} if bn_fn else None
            while s > self.kernel_size[0]:
                c = (2**(i if i < 3 else 3)) * dim
                h = layers.conv2d(h, c,
                                  kernel_size=self.kernel_size,
                                  stride=2,
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=bn_fn,
                                  normalizer_params=normalizer_params,
                                #   weights_initializer=self.w_init,
                                  scope="h{}".format(i))
                print(h.name, h.get_shape())
                if get_layer is not None:
                    A = h
                s = h.get_shape()[1]
                i += 1
            n = (2**(i if i < 3 else 3)) * dim
            h0 = layers.conv2d(h, n,
                               kernel_size=[s, s],
                               padding='VALID',
                               activation_fn=self.activation_fn,
                               normalizer_fn=bn_fn,
                               normalizer_params=normalizer_params,
                            #    weights_initializer=self.w_init,
                               scope="h{}".format(i))
            print(h0.name, h0.get_shape())
            h1 = tf.reshape(h0, shape=[-1, n], name="h{}".format(i + 1))
            print(h1.name, h1.get_shape())
        if get_layer is not None:
            return h1, A
        else:
            return h1

    def additional_summaries(self):
        return [utils.add_conv_summary()]

    def _image_summary(self, name, x, max_outputs=3, collections=None):
        if self.add_image_summary:
            tf.summary.image(name, x, max_outputs, collections)

    def _build_train_tower(self, inputs, batch_size=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X, Z, E = inputs
        X = ops.transform(X)

        print(X.name, X.get_shape())
        print(Z.name, Z.get_shape())

        self._image_summary('X', X, 2)

        G  = self._generator(Z, None, True, reuse)
        D_ = self._discriminator(G, None, True, reuse)
        D  = self._discriminator(X, None, True, True)

        self._image_summary("Generations", G, 4)
        tf.summary.histogram('D_', D_)
        tf.summary.histogram('D', D)
        tf.summary.scalar('D', tf.reduce_mean(D))
        tf.summary.scalar('D_', tf.reduce_mean(D_))
        tf.summary.scalar('sum_D', tf.reduce_mean(D + D_))
        tf.summary.scalar('sig_D', tf.reduce_mean(tf.nn.sigmoid(D)))
        tf.summary.scalar('sig_D_', tf.reduce_mean(tf.nn.sigmoid(D_)))

        with tf.name_scope('sampler'):
            g  = self._generator(Z, None, False, True)
            d  = self._discriminator(X, None, False, True)

        outputs = [g, d]

        d_loss, g_loss = self._calculate_losses(X, G, D_, D, E)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Generator' in var.name]

        with tf.name_scope('Grad_compute'):
            d_grads = self.opt.compute_gradients(d_loss, var_list=d_vars)
            g_grads = self.opt.compute_gradients(g_loss, var_list=g_vars)

        return outputs, [d_loss, g_loss], [d_grads, g_grads]

    def _tower_outputs(self, outputs):
        self.G, self.D = outputs

    def _build_train_ops(self, losses, grads):
        d_loss, g_loss = losses
        d_grads, g_grads = grads
        # Apply the gradients to adjust the shared variables.
        with tf.name_scope('Train_ops'):
            d_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Discriminator' in v.name]
            with tf.control_dependencies(d_update_ops):
                self.d_train_op = self.opt.apply_gradients(d_grads, name='D')
            g_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Generator' in v.name]
            with tf.control_dependencies(g_update_ops):
                self.g_train_op = self.opt.apply_gradients(g_grads, name='G')

        with tf.name_scope('Total_loss'):
            self.d_total_loss = tf.add_n(d_loss, name='D') / self.num_gpus
            self.g_total_loss = tf.add_n(g_loss, name='G') / self.num_gpus

    def _adversarial_loss(self, D, D_):
        with tf.name_scope('adversarial_loss'):
            if self.loss is 'v':
                d_l = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_, labels=tf.zeros_like(D_)) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=tf.ones_like(D))
                g_l = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_, labels=tf.ones_like(D_)) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=tf.zeros_like(D))
            elif self.loss is 'w':
                d_l = D_ - D
                g_l = - d_l
            elif self.loss is 'ls':
                d_l = tf.square(D_) + tf.square(1 - D)
                g_l = tf.square(D) + tf.square(1 - D_)
            elif self.loss is 'lol1':
                d_l = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_, labels=tf.zeros_like(D_)) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=tf.ones_like(D))
                g_l = D - D_
            elif self.loss is 'lol2':
                d_l = tf.square(tf.nn.sigmoid(D_)) + tf.square(1 - tf.nn.sigmoid(D))
                g_l = D - D_
            elif self.loss is 'lol3':
                d_l = tf.square(tf.nn.sigmoid(D_)) + tf.square(1 - tf.nn.sigmoid(D))
                g_l = tf.square(tf.maximum(D, 0)) - tf.square(tf.minimum(D, 0)) + tf.square(tf.minimum(D_, 0)) - tf.square(tf.maximum(D_, 0))

            d_loss = tf.reduce_mean(d_l, name='d_loss')
            g_loss = tf.reduce_mean(g_l, name='g_loss')

            tf.add_to_collection('losses', d_loss)
            tf.add_to_collection('losses', g_loss)
            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('g_loss', g_loss)

        return d_loss, g_loss

    def _calculate_losses(self, X, G, D_, D, EPSILON):
        with tf.name_scope('Losses'):
            d_loss, g_loss = self._adversarial_loss(D, D_)
            d_total_loss = [d_loss]
            g_total_loss = [g_loss]

            if self.grad_pen:
                gp_loss = self._gradient_penalty(X, G, EPSILON, self._discriminator) * self.gp_lambda
                d_total_loss.append(gp_loss)

            d_total_loss = tf.add_n(d_total_loss, name='d_total_loss')
            g_total_loss = tf.add_n(g_total_loss, name='g_total_loss')

        tf.add_to_collection('losses', d_total_loss)
        tf.add_to_collection('losses', g_total_loss)
        tf.summary.scalar('d_total_loss', d_total_loss)
        tf.summary.scalar('g_total_loss', g_total_loss)

        return d_total_loss, g_total_loss

    def _gradient_penalty(self, X, G, EPSILON, D_fn):
        with tf.name_scope('grad_penalty'):
            x_inter = ops.interpolate(X, G, EPSILON, name='x_inter')
            d_inter = D_fn(x_inter, is_training=True, reuse=True)
            grad = tf.gradients(d_inter, x_inter, name="grads")[0]
            n_dim = len(grad.get_shape())
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[i for i in range(1, n_dim)]))
            tf.summary.histogram('slopes', slopes)
            gp_loss = tf.reduce_mean((slopes - 1.)**2, name='gp_loss')
            tf.add_to_collection('losses', gp_loss)
            tf.summary.scalar('gp_loss', gp_loss)
        return gp_loss

    def _setup_input_pipe(self):
        print("Checking data...")
        filepaths = input_pipe.create_paths_file_if_required(self.sess, self.data_dir, self.s, 1, self.ext)
        print("Creating input sources...")
        with tf.name_scope('Train_batch'):
            self.x_batch_op = input_pipe.generate_image_batch(filepaths, self.s, self.batch_size, self.ext, name='X')
            self.z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.batch_size, name='Z')
            self.e_batch_op = input_pipe.generate_epsilon_batch(self.batch_size, name='EPSILON')
            self.sample_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.sampler_batch_size, self.sampler_interpolations, name='Sample_Z')
        return len(filepaths)

    def _get_data_dict(self, counter):
        x_batch, z_batch, e_batch = self.sess.run([self.x_batch_op, self.z_batch_op, self.e_batch_op])
        data_dict = {self.X: x_batch,
                     self.Z: z_batch,
                     self.EPSILON: e_batch}
        return data_dict

    def _run_step(self, counter, data_dict):
        self.sess.run(self.d_train_op, feed_dict=data_dict)
        step_op = [self.d_total_loss, self.g_total_loss]
        summarize = (counter % SUMMARIZE_AFTER == 0)
        if summarize:
            step_op.append(self.summary_op)

        i = 0
        D_ITERS = 5 if self.loss is 'w' else 1
        if counter % D_ITERS == 0:
            step_op = [self.g_train_op] + step_op
            i = 1

        result = self.sess.run(step_op, feed_dict=data_dict)[i:]

        if summarize:
            errD, errG, summ_str = result
            self.writer.add_summary(summ_str, counter)
        else:
            errD, errG = result

        update_str = "d_loss: %.4f, g_loss: %.4f".format(errD, errG)
        return update_str

    def generate_sample(self, counter):
        z = self.sess.run(self.sample_z_batch_op)
        samples = self.sess.run(ops.inverse_transform(self.G), feed_dict={self.Z: z})
        utils.make_sure_path_exits(self.samples_path)
        samplefile_path = os.path.join(self.samples_path, '{}.png'.format(counter))
        utils.save_images_h(path=samplefile_path,
                            images=samples,
                            imagesPerRow=self.sampler_interpolations)
        return [samplefile_path]


class DCGAN_cifar(DCGAN):

    def _extra_init(self):
        super()._extra_init()
        self.inc_scores = []
        self.comments = '5k'
        self.inc_batch_size = 5000

    def _get_cifar_batch(self, filename):
        batch_file = os.path.join(self.data_dir, filename)
        with open(batch_file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        x_batch = np.array(data_dict[b'data'], dtype=np.float32)
        return x_batch.reshape([-1, self.c, self.s, self.s]).transpose([0, 2, 3, 1])

    def _get_data_dict(self, counter):
        i = (counter - 1) % self.total_batches * self.batch_size
        x_batch = self.x_data[i:i + self.batch_size, :, :, :]
        z_batch, e_batch = self.sess.run([self.z_batch_op, self.e_batch_op])
        data_dict = {self.X: x_batch,
                     self.Z: z_batch,
                     self.EPSILON: e_batch}
        return data_dict

    def _setup_input_pipe(self):
        print('Getting cifar10 data...')
        self.x_data = np.concatenate([self._get_cifar_batch('data_batch_' + str(i)) for i in range(1, 6)], 0)
        self.test_data = self._get_cifar_batch('test_batch')
        print("Creating input sources...")
        with tf.name_scope('Train_batch'):
            self.z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.batch_size, name='Z')
            self.e_batch_op = input_pipe.generate_epsilon_batch(self.batch_size, name='EPSILON')
            # self.sample_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.sampler_batch_size, name='Sample_Z')
            self.sample_z_batch_op = tf.constant(np.random.uniform(-1, 1, [self.sampler_batch_size, self.z_len]))
            self.inception_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.inc_batch_size, name='Inception_Z')
        return len(self.x_data)

    def generate_sample(self, counter):
        z_batch = self.sess.run(self.sample_z_batch_op)
        g = self.sess.run(ops.inverse_transform(self.G), feed_dict={self.Z: z_batch})
        # img_array = tf.transpose(samples, [1, 0, 2, 3, 4])
        utils.make_sure_path_exits(self.samples_path)
        samplefile_path = os.path.join(self.samples_path, '{}.png'.format(counter))
        utils.save_images_h(samplefile_path,
                            g,
                            int(np.sqrt(self.sampler_batch_size)))

        self.inc_scores.append(self.get_inception_score())
        self._save_score()
        return [samplefile_path]

    # For calculating inception score
    def get_inception_score(self):
        z_batch = self.sess.run(self.inception_z_batch_op)
        bs = 5000
        n = int(self.inc_batch_size / bs)
        all_samples = []
        for i in range(n):
            g = self.sess.run(ops.inverse_transform(self.G), feed_dict={self.Z: z_batch[i * bs:(i + 1) * bs]})
            all_samples.append(g)
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = all_samples.astype('int32')
        print('Calculating inception score...')
        score = inception_score.get_inception_score(self.sess, list(all_samples))
        print(score)
        return score

    def _save_score(self):
        csvfile = os.path.join(self.log_path, 'inc_scores.csv')
        with open(csvfile, "w") as output:
            writer = csv.writer(output)
            writer.writerows([('mean', 'stddev')] + self.inc_scores)


class BIGAN(DCGAN):

    def _extra_init(self):
        super()._extra_init()
        self.g_updates = 0
        self.d_iters = 100
        self.comments = 'no init;'

    def _encoder(self, X, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope("Encoder"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            e = X
            normalizer_params = {'is_training': is_training, 'decay': 0.9, 'scale': True} if self.g_bn_fn else None
            for l in range(self.lvls):
                c = int(self.gf_dim * 2**(min(l, 3)))
                e = layers.conv2d(e, c,
                                  kernel_size=self.kernel_size,
                                  stride=2,
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.g_bn_fn,
                                  normalizer_params=normalizer_params,
                                #   weights_initializer=self.w_init,
                                  scope="e{}".format(l))
                print(e.name, e.get_shape())
            shape = e.get_shape().as_list()
            h0 = tf.reshape(e, [-1, np.prod(shape[1:])], 'h0')
            print(h0.name, h0.get_shape())
            h1 = layers.fully_connected(h0, self.z_len,
                                        activation_fn=self.out_activation_fn,
                                        # weights_initializer=self.w_init,
                                        scope="h1")
            print(h1.name, h1.get_shape())
        return h1

    def _discriminator(self, X, Z, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            normalizer_params = {'is_training': is_training, 'decay': 0.9, 'scale': True}
            c = self._full_conv(X, self.df_dim, self.d_bn_fn, None, batch_size, is_training, reuse)
            x_z = tf.concat([c, Z], axis=1, name="x_z")
            print(x_z.name, x_z.get_shape())
            d0 = layers.fully_connected(x_z, x_z.get_shape().as_list()[-1],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.d_bn_fn,
                                        normalizer_params=normalizer_params,
                                        # weights_initializer=self.w_init,
                                        scope="d0")
            print(d0.name, d0.get_shape())
            d1 = layers.fully_connected(d0, 1,
                                        activation_fn=None,
                                        # weights_initializer=self.w_init,
                                        scope="d1")
            print(d1.name, d1.get_shape())
        return d1

    def _build_train_tower(self, inputs, batch_size=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X, Z, E = inputs
        X = ops.transform(X)

        print(X.name, X.get_shape())
        print(Z.name, Z.get_shape())

        self._image_summary("Inputs", X)

        L = self._encoder(X, None, True, reuse)
        G  = self._generator(Z, None, True, reuse)
        X_hat = self._generator(L, None, True, True)
        Z_hat = self._encoder(G, None, True, True)
        D_ = self._discriminator(G, Z, None, True, reuse)
        D  = self._discriminator(X, L, None, True, True)

        self._image_summary("G", G)
        self._image_summary("X_hat", X_hat)
        tf.summary.histogram('D_', D_)
        tf.summary.histogram('D', D)
        tf.summary.scalar('D', tf.reduce_mean(D))
        tf.summary.scalar('D_', tf.reduce_mean(D_))
        tf.summary.scalar('sum_D', tf.reduce_mean(D + D_))
        tf.summary.scalar('sig_D', tf.reduce_mean(tf.nn.sigmoid(D)))
        tf.summary.scalar('sig_D_', tf.reduce_mean(tf.nn.sigmoid(D_)))

        with tf.name_scope('sampler'):
            g = self._generator(Z, None, False, True)
            l = self._encoder(X, None, False, True)
            x_recon = self._generator(l, None, False, True)
            d = self._discriminator(X, l, None, False, True)

        outputs = [g, x_recon, l, d]

        d_loss, g_loss, e_loss = self._calculate_losses(X, Z, L, G, X_hat, Z_hat, D_, D, E)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Generator' in var.name]
        e_vars = [var for var in t_vars if 'Encoder' in var.name]

        with tf.name_scope('Grad_compute'):
            d_grads = self.opt.compute_gradients(d_loss, var_list=d_vars)
            g_grads = self.opt.compute_gradients(g_loss, var_list=g_vars)
            e_grads = self.opt.compute_gradients(e_loss, var_list=e_vars)

        return outputs, [d_loss, g_loss, e_loss], [d_grads, g_grads, e_grads]

    def _tower_outputs(self, outputs):
        self.G, self.X_recons, self.L, self.D = outputs

    def _calculate_losses(self, X, Z, L, G, X_hat, Z_hat, D_, D, EPSILON):
        with tf.name_scope('Losses'):
            d_loss, g_loss = self._adversarial_loss(D, D_)
            d_total_loss = [d_loss]
            g_total_loss = [g_loss]
            e_total_loss = [g_loss]

            if self.ae_pen:
                x_recons_loss, z_recons_loss = self._ae_pen(X, X_hat, Z, Z_hat)
                g_total_loss.append(self.ae_lambda * x_recons_loss)
                e_total_loss.append(self.ae_lambda * z_recons_loss)

            if self.grad_pen:
                gp_loss = self._gradient_penalty(X, X_hat, L, G, Z, Z_hat, EPSILON) * self.gp_lambda
                d_total_loss.append(gp_loss)

            d_total_loss = tf.add_n(d_total_loss, name='d_total_loss')
            g_total_loss = tf.add_n(g_total_loss, name='g_total_loss')
            e_total_loss = tf.add_n(e_total_loss, name='e_total_loss')

        tf.add_to_collection('losses', d_total_loss)
        tf.add_to_collection('losses', g_total_loss)
        tf.add_to_collection('losses', e_total_loss)
        tf.summary.scalar('d_total_loss', d_total_loss)
        tf.summary.scalar('g_total_loss', g_total_loss)
        tf.summary.scalar('e_total_loss', e_total_loss)

        return d_total_loss, g_total_loss, e_total_loss

    def _ae_pen(self, X, X_hat, Z, Z_hat):
        ax1 = [i for i in range(1, len(X.get_shape()))]
        ax2 = [i for i in range(1, len(Z.get_shape()))]
        with tf.name_scope('X_recons_loss'):
            x_recons_loss = tf.reduce_mean(tf.abs(X - X_hat), axis=ax1)
            x_recons_loss = tf.reduce_mean(x_recons_loss, name='x_recons_loss')
            tf.summary.scalar('x_recons_loss', x_recons_loss)

        with tf.name_scope('Z_recons_loss'):
            z_recons_loss = tf.reduce_mean(tf.abs(Z - Z_hat), axis=ax2)
            z_recons_loss = tf.reduce_mean(z_recons_loss, name='z_recons_loss')
            tf.summary.scalar('z_recons_loss', z_recons_loss)
        return x_recons_loss, z_recons_loss

    def _gradient_penalty(self, X, X_hat, L, G, Z, Z_hat, EPSILON):
        ax1 = [i for i in range(1, len(X.get_shape()))]
        ax2 = [i for i in range(1, len(Z.get_shape()))]
        with tf.name_scope('grad_penalty'):
            with tf.name_scope('x'):
                x_inter = ops.interpolate(X, X_hat, EPSILON, name='x_inter')
                d_inter_x = self._discriminator(x_inter, L, is_training=True, reuse=True)
                x_grads, z_grads = tf.gradients([d_inter_x], [x_inter, L], name="grads_x")
                x_slopes = tf.sqrt(tf.reduce_sum(x_grads**2, axis=ax1))
                tf.summary.scalar('x_slopes', tf.reduce_mean(x_slopes))
                delta_x = X - X_hat
                x_unit = delta_x / ops.norm(delta_x, ax1, True)
                x_alpha = tf.reduce_sum(x_grads * x_unit, ax1) / x_slopes
                tf.summary.scalar('x_alpha', tf.reduce_mean(x_alpha))
                tf.summary.scalar('x_unit_grads', tf.reduce_mean(ops.norm(x_unit, ax1)))
                x_pen = tf.sqrt(tf.reduce_sum((x_grads - x_unit)**2, axis=ax1))
                gp_loss_x = tf.reduce_mean(x_pen, name='gp_loss_x')
                tf.add_to_collection('losses', gp_loss_x)
                tf.summary.scalar('gp_loss_x', gp_loss_x)
            with tf.name_scope('z'):
                z_inter = ops.interpolate(Z, Z_hat, EPSILON, name='z_inter')
                d_inter_z = - self._discriminator(G, z_inter, is_training=True, reuse=True)
                _, z_grads = tf.gradients([d_inter_z], [G, z_inter], name="grads_z")
                delta_z = Z - Z_hat
                z_unit = delta_z / ops.norm(delta_z, ax2, True)
                z_pen = tf.sqrt(tf.reduce_sum((z_grads - z_unit)**2, axis=ax2))
                gp_loss_z = tf.reduce_mean(z_pen, name='gp_loss_z')
                tf.add_to_collection('losses', gp_loss_z)
                tf.summary.scalar('gp_loss_z', gp_loss_z)
            gp_loss = gp_loss_x
            # gp_loss = gp_loss_x + gp_loss_z
            tf.add_to_collection('losses', gp_loss)
            tf.summary.scalar('gp_loss', gp_loss)
        return gp_loss

    def _build_train_ops(self, losses, grads):
        d_loss, g_loss, e_loss = losses
        d_grads, g_grads, e_grads = grads
        # Apply the gradients to adjust the shared variables.
        with tf.name_scope('Train_ops'):
            d_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Discriminator' in v.name]
            with tf.control_dependencies(d_update_ops):
                d_train_op = self.opt.apply_gradients(d_grads, name='D')
            g_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Generator' in v.name]
            with tf.control_dependencies(g_update_ops):
                g_train_op = self.opt.apply_gradients(g_grads, name='G')
            e_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Encoder' in v.name]
            with tf.control_dependencies(e_update_ops):
                e_train_op = self.opt.apply_gradients(e_grads, name='E')

            self.d_train_op = tf.group(d_train_op)
            self.g_train_op = tf.group(g_train_op, e_train_op)

        with tf.name_scope('Total_loss'):
            self.d_total_loss = tf.add_n(d_loss, name='D') / self.num_gpus
            self.g_total_loss = tf.add_n(g_loss, name='G') / self.num_gpus
            self.e_total_loss = tf.add_n(e_loss, name='E') / self.num_gpus

    def _setup_input_pipe(self):
        print("Checking data...")
        filepaths = input_pipe.create_paths_file_if_required(self.sess, self.data_dir, self.s, 1, self.ext)
        random.shuffle(filepaths)
        train_files = filepaths[:-self.sampler_batch_size]
        test_files = filepaths[-self.sampler_batch_size:]
        print("Creating input sources...")
        with tf.name_scope('Train_batch'):
            self.x_batch_op = input_pipe.generate_image_batch(train_files, self.s, self.batch_size, self.ext, name='X')
            self.z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.batch_size, name='Z')
            self.e_batch_op = input_pipe.generate_epsilon_batch(self.batch_size, name='EPSILON')
            self.sample_x_batch_op = input_pipe.generate_image_batch(train_files, self.s, self.sampler_batch_size, self.ext, name='Sample_X')
            self.sample_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.sampler_batch_size, name='Sample_Z')
            self.test_x_batch_op = input_pipe.generate_image_batch(test_files, self.s, self.sampler_batch_size, self.ext, name='TEST_X')
            self.test_z_batch = np.random.uniform(-1., 1., [self.sampler_batch_size, self.z_len])
        return len(train_files)

    def generate_sample(self, counter):
        x_batch, z_batch = self.sess.run([self.sample_x_batch_op, self.sample_z_batch_op])
        sampler = [ops.inverse_transform(self.G), ops.inverse_transform(self.X_recons)]
        g, x_recons = self.sess.run(sampler, feed_dict={self.X: x_batch,
                                                        self.Z: z_batch})
        filename = '{}.png'.format(counter)
        recons_path = os.path.join(self.samples_path, 'train/recons')
        utils.make_sure_path_exits(recons_path)
        utils.save_images_h(os.path.join(recons_path, filename),
                            np.concatenate([x_batch, x_recons], 0),
                            self.sampler_batch_size)
        gens_path = os.path.join(self.samples_path, 'train/gens')
        utils.make_sure_path_exits(gens_path)
        utils.save_images_h(os.path.join(gens_path, filename),
                            g,
                            int(np.sqrt(self.sampler_batch_size)))

        # Test set
        x_batch = self.sess.run(self.test_x_batch_op)
        g, x_recons = self.sess.run(sampler, feed_dict={self.X: x_batch,
                                                        self.Z: self.test_z_batch})
        filename = '{}.png'.format(counter)
        recons_path = os.path.join(self.samples_path, 'test/recons')
        utils.make_sure_path_exits(recons_path)
        utils.save_images_h(os.path.join(recons_path, filename),
                            np.concatenate([x_batch, x_recons], 0),
                            self.sampler_batch_size)
        gens_path = os.path.join(self.samples_path, 'test/gens')
        utils.make_sure_path_exits(gens_path)
        utils.save_images_h(os.path.join(gens_path, filename),
                            g,
                            int(np.sqrt(self.sampler_batch_size)))
        return [gens_path, recons_path]


class BIGAN_cifar(BIGAN):

    def _extra_init(self):
        super()._extra_init()
        self.inc_scores = []
        self.diff_scores = []
        self.comments = '5k'
        self.inc_batch_size = 5000

    def _get_cifar_batch(self, filename):
        batch_file = os.path.join(self.data_dir, filename)
        with open(batch_file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        x_batch = np.array(data_dict[b'data'], dtype=np.float32)
        return x_batch.reshape([-1, self.c, self.s, self.s]).transpose([0, 2, 3, 1])

    def _get_data_dict(self, counter):
        i = (counter - 1) % self.total_batches * self.batch_size
        x_batch = self.x_data[i:i + self.batch_size, :, :, :]
        z_batch, e_batch = self.sess.run([self.z_batch_op, self.e_batch_op])
        data_dict = {self.X: x_batch,
                     self.Z: z_batch,
                     self.EPSILON: e_batch}
        return data_dict

    def _setup_input_pipe(self):
        print('Getting cifar10 data...')
        self.x_data = np.concatenate([self._get_cifar_batch('data_batch_' + str(i)) for i in range(1, 6)], 0)
        self.test_data = self._get_cifar_batch('test_batch')
        print("Creating input sources...")
        with tf.name_scope('Train_batch'):
            self.z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.batch_size, name='Z')
            self.e_batch_op = input_pipe.generate_epsilon_batch(self.batch_size, name='EPSILON')
            # self.sample_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.sampler_batch_size, name='Sample_Z')
            self.sample_z_batch_op = tf.constant(np.random.uniform(-1., 1., [self.sampler_batch_size, self.z_len]))
            self.inception_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.inc_batch_size, name='Inception_Z')
        return len(self.x_data)

    def generate_sample(self, counter):
        n = int(np.random.random() * (len(self.x_data) - self.sampler_batch_size))
        x_batch = self.x_data[n:n + self.sampler_batch_size, :, :, :]
        z_batch = self.sess.run(self.sample_z_batch_op)
        sampler = [ops.inverse_transform(self.G), ops.inverse_transform(self.X_recons)]
        g, x_recons = self.sess.run(sampler, feed_dict={self.X: x_batch,
                                                        self.Z: z_batch})
        filename = '{}.png'.format(counter)
        recons_path = os.path.join(self.samples_path, 'recons')
        utils.make_sure_path_exits(recons_path)
        utils.save_images_h(os.path.join(recons_path, filename),
                            np.concatenate([x_batch, x_recons], 0),
                            self.sampler_batch_size)
        gens_path = os.path.join(self.samples_path, 'gens')
        utils.make_sure_path_exits(gens_path)
        utils.save_images_h(os.path.join(gens_path, filename),
                            g,
                            int(np.sqrt(self.sampler_batch_size)))
        self.inc_scores.append(self.get_inception_score())
        self.diff_scores.append(self.get_inception_score_diff())
        self._save_score()
        return [gens_path, recons_path]

    # For calculating inception score
    def get_inception_score(self, z_batch=None):
        if z_batch is None:
            z_batch = self.sess.run(self.inception_z_batch_op)
        all_samples = []
        bs = 5000
        n = int(len(z_batch) / bs)
        for i in range(n):
            g = self.sess.run(ops.inverse_transform(self.G), feed_dict={self.Z: z_batch[i * bs: (i + 1) * bs]})
            all_samples.append(g)
        all_samples = np.concatenate(all_samples, axis=0).astype('int32')
        print('Calculating inception score...')
        st = time.time()
        score = inception_score.get_inception_score(self.sess, list(all_samples))
        print(score, '  ...', str(time.time() - st), 's')
        return score

    def get_inception_score_diff(self, x_batch=None):
        if x_batch is None:
            n = int(np.random.random() * (len(self.x_data) - self.inc_batch_size))
            x_batch = self.x_data[n:n + self.inc_batch_size, :, :, :]
        all_recons = []
        bs = 5000
        n = int(len(x_batch) / bs)
        for i in range(n):
            recons = self.sess.run(ops.inverse_transform(self.X_recons), feed_dict={self.X: x_batch[i * bs:(i + 1) * bs]})
            all_recons.append(recons)
        all_recons = np.concatenate(all_recons, axis=0)
        st = time.time()
        # print('Calculating inception diff score...')
        # x_pred = inception_score.get_inception_preds(self.sess, list(x_batch), raw=True)
        # r_pred = inception_score.get_inception_preds(self.sess, list(all_recons), raw=True)
        # scores = np.sum((x_pred - r_pred)**2, 1)
        # scores = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(x_pred), logits=r_pred)
        # scores = inception_score.get_inception_diff(self.sess, x_batch, all_recons)
        # SSIM
        print('Calculating ssim...')
        scores = []
        for i in range(len(x_batch)):
            scores.append(ssim(x_batch[i], all_recons[i], multichannel=True))
        score = (np.mean(scores), np.std(scores))
        print(score, '  ...', str(time.time() - st), 's')
        return score

    def _save_score(self):
        csvfile = os.path.join(self.log_path, 'inc_score.csv')
        with open(csvfile, "w") as output:
            writer = csv.writer(output)
            writer.writerows([('mean', 'stddev')] + self.inc_scores)
        csvfile = os.path.join(self.log_path, 'diff_score.csv')
        with open(csvfile, "w") as output:
            writer = csv.writer(output)
            writer.writerows([('mean', 'stddev')] + self.diff_scores)


# End
