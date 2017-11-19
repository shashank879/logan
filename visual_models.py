import tensorflow as tf
import ops, input_pipe
from tensorflow.contrib import layers
import os
from image_models import BIGAN
import cv2
from collections import OrderedDict
import utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


SUMMARIZE_AFTER = 5


class vizBIGAN(BIGAN):

    def _extra_init(self):
        super()._extra_init()
        # self.w_init = tf.random_uniform_initializer(-np.sqrt(3) * .04, np.sqrt(3) * .04)
        self.w_init = tf.uniform_unit_scaling_initializer(1.43)

    def _setup_config(self, config):
        self.ext = config['ext']
        self.lvls = config['lvls']
        self.x_len = config['x_len']
        self.z_len = config['z_len']
        self.gf_dim = config['gf_dim']
        self.df_dim = config['df_dim']
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

        if self.optimizer is 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9)
        elif self.optimizer is 'rms':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)

        self.d_bn_fn = layers.batch_norm if d_bn else None
        self.g_bn_fn = layers.batch_norm if g_bn else None

        # Seaborn style
        # sns.set(style="white")
        sns.set(style="darkgrid")
        sns.set_context('paper')
        self.fig = plt.figure(figsize=plt.figaspect(1 / self.z_len))
        # ax.set_aspect('equal')
        self.sample_imgs = []

    def _setup_placeholders(self):
        # Create placeholders
        with tf.name_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.x_len], name="X")
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
                              ('loss', self.loss),
                              ('grad_pen', self.grad_pen),
                              ('ae_pen', self.ae_pen),
                              ('lvls', str(self.lvls)),
                              ('x_len', str(self.x_len)),
                              ('z_len', str(self.z_len)),
                              ('gf_dim', self.gf_dim),
                              ('df_dim', self.df_dim),
                              ('gp_lambda', self.gp_lambda),
                              ('ae_lambda', self.ae_lambda)])
        return config

    def _generator(self, Z, batch_size=None, is_training=True, reuse=False, scope='Generator'):
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            normalizer_params = {'is_training': is_training}
            g0 = layers.fully_connected(Z, self.gf_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="g0")
            print(g0.name, g0.get_shape())
            g1 = layers.fully_connected(g0, self.gf_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="g1")
            print(g1.name, g1.get_shape())
            g2 = layers.fully_connected(g1, self.gf_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="g2")
            print(g2.name, g2.get_shape())
            len = self.x_len if scope is 'Generator' else self.z_len
            g = layers.fully_connected(g2, len, activation_fn=None, scope="g")
            print(g.name, g.get_shape())
        return g

    def _discriminator(self, X, Z, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            normalizer_params = {'is_training': is_training}
            x_z = tf.concat([X, Z], 1)
            print(x_z.name, x_z.get_shape())
            d0 = layers.fully_connected(x_z, self.df_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="d0")
            print(d0.name, d0.get_shape())
            d1 = layers.fully_connected(d0, self.df_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="d1")
            print(d1.name, d1.get_shape())
            d2 = layers.fully_connected(d1, self.df_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="d2")
            print(d2.name, d2.get_shape())
            d3 = layers.fully_connected(d2, 1, activation_fn=None, scope="d3")
            print(d3.name, d3.get_shape())
            return d3

    def additional_summaries(self):
        return []

    def _build_train_tower(self, inputs, batch_size=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X, Z, E = inputs

        print(X.name, X.get_shape())
        print(Z.name, Z.get_shape())

        G  = self._generator(Z, None, True, reuse, 'Generator')
        L  = self._generator(X, None, True, reuse, 'Encoder')
        D_ = self._discriminator(G, Z, None, True, reuse)
        D  = self._discriminator(X, L, None, True, True)
        X_hat = self._generator(L, None, True, True, 'Generator')
        Z_hat = self._generator(G, None, True, True, 'Encoder')

        tf.summary.histogram('X', X)
        tf.summary.histogram('Z', Z)
        tf.summary.histogram('L', L)
        tf.summary.histogram('G', G)
        tf.summary.histogram('X_hat', X_hat)
        tf.summary.histogram('Z_hat', Z_hat)
        tf.summary.histogram('D_', D_)
        tf.summary.histogram('D', D)

        with tf.name_scope('sampler'):
            g = self._generator(Z, None, False, True, 'Generator')
            l = self._generator(X, None, False, True, 'Encoder')
            d  = self._discriminator(X, Z, None, False, True)
            d_map = self._discriminator(X, l, None, False, True)

        d_loss, g_loss, e_loss = self._calculate_losses(X, Z, L, G, X_hat, Z_hat, D_, D, E)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Generator' in var.name]
        e_vars = [var for var in t_vars if 'Encoder' in var.name]

        with tf.name_scope('Grad_compute'):
            d_grads = self.opt.compute_gradients(d_loss, var_list=d_vars)
            g_grads = self.opt.compute_gradients(g_loss, var_list=g_vars)
            e_grads = self.opt.compute_gradients(e_loss, var_list=e_vars)

        return [g, l, d, d_map], [d_loss, g_loss, e_loss], [d_grads, g_grads, e_grads]

    def _tower_outputs(self, outputs):
        self.G, self.L, self.D, self.D_MAP = outputs

    def _get_input_source(self, batch_size, add_far=False):
        if self.dataset_name is 'radial_gauss':
            return input_pipe.radial_gaussians(batch_size, radius=1.0)
        elif self.dataset_name is 'radial_gauss2':
            return input_pipe.radial_gaussians2(batch_size, r1=1.0, r2=2.0)
        elif self.dataset_name is 'rect':
            return input_pipe.rect(batch_size)
        elif self.dataset_name is 'swiss':
            return input_pipe.swiss(batch_size, 2.)
        elif self.dataset_name is 'line':
            return input_pipe.line_1d(batch_size, 5, .01, .5, add_far=add_far)

    def _setup_input_pipe(self):
        print("Creating input sources...")
        with tf.name_scope('Train_batch'):
            self.x_batch_op1 = tf.random_normal([self.batch_size, self.x_len], stddev=.01)
            self.x_batch_op2 = self._get_input_source(self.batch_size, True)
            self.z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.batch_size, name='Z')
            self.e_batch_op = input_pipe.generate_epsilon_batch(self.batch_size, name='EPSILON')
            self.sample_x_batch_op = self._get_input_source(self.sampler_batch_size, True)
            self.sample_z_batch_op = input_pipe.generate_noise_batch([self.z_len], self.sampler_batch_size, name='Sample_Z')
        return self.batch_size * 200

    def _get_data_dict(self, counter):
        x_batch_op = self.x_batch_op2
        # epoch = (counter - 1) / self.total_batches
        # x_batch_op = self.x_batch_op1 if epoch < 50 else self.x_batch_op2
        x_batch, z_batch, e_batch = self.sess.run([x_batch_op, self.z_batch_op, self.e_batch_op])
        data_dict = {self.X: x_batch,
                     self.Z: z_batch,
                     self.EPSILON: e_batch}
        return data_dict

    def generate_sample(self, counter):
        z_batch, x_batch = self.sess.run([self.sample_z_batch_op, self.sample_x_batch_op])

        N_POINTS = 128
        RANGE = 5.5 if self.x_len is 1 else 2.5
        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
        points = points.reshape([-1, 2])
        x_grid = y_grid = np.linspace(-RANGE, RANGE, N_POINTS)

        if self.x_len is 1 and self.z_len is 1:
            # 2D plot
            g, l = self.sess.run([self.G, self.L], feed_dict={self.X: x_batch,
                                                              self.Z: z_batch})
            disc_map = self.sess.run(self.D, feed_dict={self.X: points[:, :1],
                                                        self.Z: points[:, 1:]})
            ax = plt.subplot(111)
            plt.contourf(x_grid, y_grid, disc_map.reshape([N_POINTS, N_POINTS]).transpose(), 20, cmap=plt.cm.Blues)
            plt.colorbar()
            plt.scatter(g, z_batch, c='green', marker='.', label='(G(z),z)')
            plt.scatter(x_batch, l, c='orange', marker='.', label='(x,E(x))')
            ax.set(xlabel="X", ylabel="Z")

        # elif self.x_len is 2 and self.z_len is 1:
        #     # 3D plot
        #     g, l = self.sess.run([self.G, self.L], feed_dict={self.X: x_batch,
        #                                                       self.Z: z_batch})
        #     disc_map = self.sess.run(self.D_MAP, feed_dict={self.X: points})
        #     ax = plt.subplot(111, projection='3d')
        #     ax.contourf(x_grid, y_grid, disc_map.reshape([N_POINTS, N_POINTS]).transpose(), 20, offset=disc_map.min(), cmap=plt.cm.Blues)
        #     ax.scatter(g[:, 0], g[:, 1], z_batch, c='green', marker='.', label='(G(z),z)')
        #     ax.scatter(x_batch[:, 0], x_batch[:, 1], l, c='orange', marker='.', label='(x,E(x))')
        #     ax.set(xlabel='X1', ylabel='X2', zlabel='Z')

        elif self.x_len is 2:
            # 3D plot
            g, l = self.sess.run([self.G, self.L], feed_dict={self.X: x_batch,
                                                              self.Z: z_batch})
            disc_map = self.sess.run(self.D_MAP, feed_dict={self.X: points})
            for i in range(self.z_len):
                ax = self.fig.add_subplot(1, self.z_len, i + 1, projection='3d')
                min_point = np.concatenate([g, l], axis=0).min()
                ax.contourf(x_grid, y_grid, disc_map.reshape([N_POINTS, N_POINTS]).transpose(), 20, offset=min_point, cmap=plt.cm.Blues)
                ax.scatter(g[:, 0], g[:, 1], z_batch[:, i], c='green', marker='.', label='(G(z),z)')
                ax.scatter(x_batch[:, 0], x_batch[:, 1], l[:, i], c='orange', marker='.', label='(x,E(x))')
                ax.set(xlabel='X1', ylabel='X2', zlabel='Z{}'.format(i))

        plt.legend()
        samplefile_path = os.path.join(self.samples_path, '{}.png'.format(counter))
        self.fig.savefig(samplefile_path)
        plt.clf()
        self.sample_imgs.append(samplefile_path)
        return [samplefile_path]

    def _post_train(self):
        imgs = []
        for path in self.sample_imgs:
            img = cv2.imread(path)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        imgs = np.array(imgs)
        samplefile_path = os.path.join(self.samples_path, 'combined.png')
        utils.save_images_h(samplefile_path, imgs, 10)


class vizDCGAN(vizBIGAN):

    def _discriminator(self, X, batch_size=None, is_training=True, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            normalizer_params = {'is_training': is_training, 'decay': 0.9}
            d0 = layers.fully_connected(X, self.df_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="d0")
            print(d0.name, d0.get_shape())
            d1 = layers.fully_connected(d0, self.df_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="d1")
            print(d1.name, d1.get_shape())
            d2 = layers.fully_connected(d1, self.df_dim, activation_fn=self.activation_fn, normalizer_fn=self.g_bn_fn, normalizer_params=normalizer_params, weights_initializer=self.w_init, scope="d2")
            print(d2.name, d2.get_shape())
            d = layers.fully_connected(d2, 1, activation_fn=None, scope="d")
            print(d.name, d.get_shape())
            return d

    def _build_train_tower(self, inputs, batch_size=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X, Z, E = inputs

        print(X.name, X.get_shape())
        print(Z.name, Z.get_shape())

        G  = self._generator(Z, None, True, reuse)
        D_ = self._discriminator(G, None, True, reuse)
        D  = self._discriminator(X, None, True, True)

        tf.summary.histogram('X', X)
        tf.summary.histogram('Z', Z)
        tf.summary.histogram('G', G)
        tf.summary.histogram('D_', D_)
        tf.summary.histogram('D', D)

        with tf.name_scope('sampler'):
            g = self._generator(Z, None, False, True)
            d = self._discriminator(X, None, False, True)

        d_loss, g_loss = self._calculate_losses(X, G, D_, D, E)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Generator' in var.name]

        with tf.name_scope('Grad_compute'):
            d_grads = self.opt.compute_gradients(d_loss, var_list=d_vars)
            g_grads = self.opt.compute_gradients(g_loss, var_list=g_vars)

        return [g, d], [d_loss, g_loss], [d_grads, g_grads]

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

    def _calculate_losses(self, X, G, D_, D, EPSILON):
        with tf.name_scope('Losses'):
            d_loss, g_loss = self._adversarial_loss(D, D_)
            d_total_loss = [d_loss]
            g_total_loss = [g_loss]

            if self.grad_pen:
                gp_loss = self._gradient_penalty(X, G, EPSILON) * self.gp_lambda
                d_total_loss.append(gp_loss)

            d_total_loss = tf.add_n(d_total_loss, name='d_total_loss')
            g_total_loss = tf.add_n(g_total_loss, name='g_total_loss')

        tf.add_to_collection('losses', d_total_loss)
        tf.add_to_collection('losses', g_total_loss)
        tf.summary.scalar('d_total_loss', d_total_loss)
        tf.summary.scalar('g_total_loss', g_total_loss)

        return d_total_loss, g_total_loss

    def _gradient_penalty(self, X, G, EPSILON):
        with tf.name_scope('grad_penalty'):
            x_inter = ops.interpolate(X, G, EPSILON, name='x_inter')
            d_inter = self._discriminator(x_inter, None, True, True)
            grad = tf.gradients(d_inter, x_inter, name="grads")[0]
            n_dim = len(grad.get_shape())
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[i for i in range(1, n_dim)]))
            tf.summary.histogram('slopes', slopes)
            gp_loss = tf.reduce_mean((slopes - 1.)**2, name='gp_loss')
            tf.add_to_collection('losses', gp_loss)
            tf.summary.scalar('gp_loss', gp_loss)
        return gp_loss

    def generate_sample(self, counter):
        x_batch, z_batch = self.sess.run([self.sample_x_batch_op, self.sample_z_batch_op])

        N_POINTS = 128
        RANGE = 2.5
        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
        points = points.reshape([-1, 2])
        x_grid = y_grid = np.linspace(-RANGE, RANGE, N_POINTS)

        gen, disc_map = self.sess.run([self.G, self.D], feed_dict={self.X: points,
                                                                   self.Z: z_batch})

        plt.subplot(111)
        plt.contourf(x_grid, y_grid, disc_map.reshape([N_POINTS, N_POINTS]).transpose(), 20, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.scatter(gen[:, 0], gen[:, 1], c='green', marker='.', label='Pr')
        plt.scatter(x_batch[:, 0], x_batch[:, 1], c='orange', marker='.', label='Pr')

        samplefile_path = os.path.join(self.samples_path, '{}.png'.format(counter))
        self.fig.savefig(samplefile_path)
        plt.clf()
        self.sample_imgs.append(samplefile_path)
        return [samplefile_path]


class vizFIXED_GEN(vizDCGAN):

    def _build_train_tower(self, inputs, batch_size=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X, Z, E = inputs

        print(X.name, X.get_shape())
        print(Z.name, Z.get_shape())

        G  = X + (1. * tf.random_normal(tf.shape(X)))
        D_ = self._discriminator(G, None, True, reuse)
        D  = self._discriminator(X, None, True, True)

        tf.summary.histogram('X', X)
        tf.summary.histogram('Z', Z)
        tf.summary.histogram('G', G)
        tf.summary.histogram('D_', D_)
        tf.summary.histogram('D', D)

        with tf.name_scope('sampler'):
            d  = self._discriminator(X, Z, None, False, True)

        d_loss, g_loss = self._calculate_losses(X, G, D_, D, E)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]

        with tf.name_scope('Grad_compute'):
            d_grads = self.opt.compute_gradients(d_loss, var_list=d_vars)

        return [d], [d_loss], [d_grads]

    def _tower_outputs(self, outputs):
        self.D = outputs

    def _build_train_ops(self, losses, grads):
        d_loss = losses
        d_grads = grads
        # Apply the gradients to adjust the shared variables.
        with tf.name_scope('Train_ops'):
            d_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Discriminator' in v.name]
            with tf.control_dependencies(d_update_ops):
                self.d_train_op = self.opt.apply_gradients(d_grads, name='D')

        with tf.name_scope('Total_loss'):
            self.d_total_loss = tf.add_n(d_loss, name='D') / self.num_gpus

    def generate_sample(self, counter):
        x_batch = self.sess.run([self.sample_x_batch_op])

        N_POINTS = 128
        RANGE = 10.5
        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
        points = points.reshape([-1, 2])
        x_grid = y_grid = np.linspace(-RANGE, RANGE, N_POINTS)

        disc_map = self.sess.run(self.D, feed_dict={self.X})

        plt.contour(x_grid, y_grid, disc_map.reshape([N_POINTS, N_POINTS]).transpose())
        plt.colorbar()
        plt.scatter(x_batch[0], x_batch[1], c='orange', marker='.', label='Pr')

        samplefile_path = os.path.join(self.samples_path, '{}.png'.format(counter))
        self.fig.savefig(samplefile_path)
        plt.clf()
        self.sample_imgs.append(samplefile_path)
        return [samplefile_path]

# End
