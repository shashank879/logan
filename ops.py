import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def transform(images, name='img_trans'):
    with tf.name_scope(name):
        return images / 127.5 - 1.0


def inverse_transform(images, name='inv_img_trans'):
    with tf.name_scope(name):
        max = tf.abs(tf.reduce_max(images))
        min = tf.abs(tf.reduce_min(images))
        return images * (127.5 / tf.maximum(max, min)) + 127.5


def cdf_nd(x, axes, offset=0., scale=1., name='cdf_nd'):
    """ CDF of Normal distribution
    """

    with tf.variable_scope(name):
        m, v = tf.nn.moments(x, axes=axes)
        x1 = (x - m + offset) / tf.sqrt(2 * v * scale)
        x2 = 0.5 * (1. + tf.erf(x1))
    return x2


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            else:
                print(v.name, "This is the culprit")

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def interpolate(x1, x2, epsilon, name='interpolate'):
    with tf.name_scope(name):
        shape = x1.get_shape().as_list()[1:]
        p = 1
        for s in shape:
            p *= s
        e = tf.matmul(epsilon, tf.ones((1, p)), name='e')
        e = tf.reshape(e, [-1] + shape)
        return tf.add(e * x1, (1. - e) * x2, name='x_inter')


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def layer_norm(inputs, decay=0.999, epsilon=0.001, is_training=True, reuse=None, scope='layer_norm'):
    """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """

    shape = inputs.get_shape().as_list()
    # n = shape[1:]
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse()

        s = tf.get_variable('scale', shape=shape[1:], dtype=tf.float32, initializer=tf.ones_initializer())
        b = tf.get_variable('bias', shape=shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
        pop_mean = tf.get_variable('pop_mean', shape=shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', shape=shape[1:], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)

        if is_training:
            axes = [i for i in range(len(shape) - 1)]
            batch_mean, batch_var = tf.nn.moments(inputs, axes)
            # batch_mean /= n
            # batch_var /= n**0.5
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                normalised_inputs = (inputs - batch_mean) / tf.sqrt(batch_var + epsilon)
                return normalised_inputs * s + b
        else:
            normalised_inputs = (inputs - pop_mean) / tf.sqrt(pop_var + epsilon)
            return normalised_inputs * s + b


def conv3d(inputs, num_outputs, kernel_size, stride=1, padding='SAME',
           activation_fn=lrelu, normalizer_fn=None, normalizer_params=None,
           weights_initializer=tf.truncated_normal_initializer, weights_regularizer=None,
           biases_initializer=tf.zeros_initializer, biases_regularizer=None,
           reuse=None, variables_collections=None, outputs_collections=None, trainable=True, scope='conv3d'):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse()

        n = inputs.get_shape()[4]
        if type(kernel_size) is not list:
            kernel_size = [kernel_size for i in range(3)]
        if type(stride) is not list:
            stride = [stride for i in range(3)]
        w = tf.get_variable('weights', shape=kernel_size + [n, num_outputs], initializer=weights_initializer, regularizer=weights_regularizer)
        c = tf.nn.conv3d(inputs, w, [1] + stride + [1], padding, name='conv')

        if normalizer_fn:
            c = normalizer_fn(c, **normalizer_params)
        else:
            b = tf.get_variable('biases', shape=[num_outputs], initializer=biases_initializer, regularizer=biases_regularizer)
            c = c + b

        if activation_fn:
            c = activation_fn(c)

        return c


def deconv3d(inputs, num_outputs, kernel_size, stride=1, padding='SAME',
             activation_fn=lrelu, normalizer_fn=None, normalizer_params=None,
             weights_initializer=tf.truncated_normal_initializer, weights_regularizer=None,
             biases_initializer=tf.zeros_initializer, biases_regularizer=None,
             reuse=None, variables_collections=None, outputs_collections=None, trainable=True, scope='deconv3d'):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse()

        # t, h, w, n = inputs.get_shape().as_list()[1:]
        # if type(kernel_size) is not list:
        #     kernel_size = [kernel_size for i in range(3)]
        # if type(stride) is not list:
        #     stride = [stride for i in range(3)]
        # weights = tf.get_variable('weights', shape=kernel_size + [num_outputs, n], initializer=weights_initializer, regularizer=weights_regularizer)
        # out_shape = tf.constant([None, t * stride[0], h * stride[1], w * stride[2], num_outputs])
        c = tf.layers.conv3d_transpose(inputs, num_outputs, kernel_size, stride, padding, kernel_initializer=weights_initializer, kernel_regularizer=weights_regularizer, name='conv')

        if normalizer_fn:
            c = normalizer_fn(c, **normalizer_params)
        else:
            b = tf.get_variable('biases', shape=[num_outputs], initializer=biases_initializer, regularizer=biases_regularizer)
            c = c + b

        if activation_fn:
            c = activation_fn(c)

        return c


def shuffle_batches(inputs):
    with tf.name_scope('shuffle_batches'):
        batch_size = inputs.get_shape().as_list()[0]
        inputs = [inputs] if type(inputs) is not list else inputs
        orig_shapes = []
        c = []
        sizes = []
        for x in inputs:
            orig_shapes.append(x.get_shape().as_list()[1:])
            x_c = tf.reshape(x, shape=(batch_size, -1))
            c.append(x_c)
            sizes.append(x_c.get_shape().as_list()[1])
        c = tf.concat(c, axis=1)
        c_shuffle = tf.random_shuffle(c)
        i = 0
        outputs = []
        for s in sizes:
            outputs.append(c_shuffle[batch_size, i: s])
            i += s

        return outputs if len(outputs) > 1 else outputs[0]


def image_softmax(logits, name='img_softmax'):
    with tf.name_scope(name):
        h, w, c = logits.get_shape().as_list()[1:4]
        x = tf.reshape(logits, [-1, h * w, c])
        x = tf.nn.softmax(x, dim=1)
        return tf.reshape(x, [-1, h, w, c])


def norm(x, axis, keep_dims=False):
    with tf.name_scope('norm'):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=keep_dims))


def instance_norm(x, decay=0.9, scale=True, is_training=True, scope='instance_norm'):
    with tf.variable_scope(scope):
        epsilon = 1e-5
        axes = [1, 2] if len(x.get_shape()) == 4 else [1]
        mean, var = tf.nn.moments(x, axes, keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out


def general_fc(inputlayer, n, stddev=0.02, name="fc", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        out = layers.fully_connected(inputlayer, n, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            out = instance_norm(out)
            # conv = layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                out = tf.nn.relu(out, "relu")
            else:
                out = lrelu(out, relufactor, "lrelu")
        return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        conv = layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            # conv = instance_norm(conv)
            conv = layers.batch_norm(conv, decay=0.9, scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv


def general_deconv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=2, s_w=2, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        conv = layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            # conv = instance_norm(conv)
            conv = layers.batch_norm(conv, decay=0.9, scope="batch_norm")
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv


def build_resnet_block(inputres, dim, type='same', name="resnet"):
    assert type in ['up', 'down', 'same'], 'Invalid type'
    input_dim = inputres.get_shape().as_list()[-1]
    is_skip = (input_dim == dim) and type is 'same'

    def conv_avg_pool(x, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv_avg_pool", do_norm=True, do_relu=True, relufactor=0):
        with tf.variable_scope(name):
            c1 = general_conv2d(x, o_d, f_h, f_w, 1, 1, stddev, padding, 'conv2d', do_norm, do_relu, relufactor)
            c2 = tf.nn.avg_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        return c2

    if type is 'up':
        shortcut_fn = general_deconv2d
        conv1 = general_deconv2d
        conv2 = general_conv2d
    elif type is 'down':
        shortcut_fn = conv_avg_pool
        conv1 = general_conv2d
        conv2 = conv_avg_pool
    elif type is 'same':
        shortcut_fn = general_conv2d
        conv1 = general_conv2d
        conv2 = general_conv2d

    s = 1 if type is 'same' else 2
    with tf.variable_scope(name):
        out_res = inputres
        # out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv1(out_res, dim, 3, 3, stddev=0.02, padding="SAME", name="c1")
        # out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2(out_res, dim, 3, 3, stddev=0.02, padding="SAME", name="c2", do_relu=False)

        if is_skip:
            shortcut = inputres
        else:
            shortcut = shortcut_fn(inputres, dim, 3, 3, s, s, .02, 'SAME', name="shortcut", do_norm=False, do_relu=False)
        return tf.nn.relu(out_res + shortcut)

# End
