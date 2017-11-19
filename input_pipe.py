import tensorflow as tf
import ops, utils
import os
from PIL import Image
import numpy as np
from tensorflow.contrib import distributions as ds
from sklearn import datasets


PATHS_FILENAME = '{}_paths-{}.txt'


def create_paths_file_if_required(sess, dir, im_size, timesteps, ext='jpg', precheck=False, forceRefresh=False):
    paths_filename = PATHS_FILENAME.format(ext, timesteps)
    if forceRefresh:
        return utils.create_path_list(dir, timesteps, ext, precheck, paths_filename)
    else:
        try:
            with open(os.path.join(dir, paths_filename)) as f:
                lines = [line.strip() for line in f]
            return lines
        except:
            print("Cannot find paths file...")
            return utils.create_path_list(dir, timesteps, ext, precheck, paths_filename)


def _create_record_file(sess, filename, dir, timesteps, im_size, ext='jpg', shuffle_data=True):
    filepaths = create_paths_file_if_required(sess, dir, im_size, timesteps, shuffle_data=shuffle_data, forceRefresh=False)
    filename_queue = tf.train.string_input_producer(filepaths, capacity=512)

    # Read sample image to get frame size
    ratio = 1
    with Image.open(filepaths[0]) as img:
        s = img.size[0]
        if s % im_size == 0:
            ratio = int(s / im_size)
    read_input = ImageReader(filename_queue, ext, ratio=ratio)
    image = read_input.image

    with tf.variable_scope('Preprocessing'):
        s = int(s / ratio)
        image = tf.cast(image, dtype=tf.float32)
        image = image[:timesteps * s, :, :]
        image.set_shape([timesteps * s, s, 3])
        if im_size != image.get_shape()[1]:
            image = tf.image.resize_images(image, size=tf.constant([im_size * timesteps, im_size]))
        image = tf.reshape(image, shape=[timesteps, im_size, im_size, 3])

    writer = tf.python_io.TFRecordWriter(filename)
    print('Creating tfrecords file...')

    tf.train.start_queue_runners()
    for i in range(len(filepaths)):
        img = sess.run(image)
        feature = {'s': ops.int64_feature(im_size),
                   'image_raw': ops.bytes_feature(img.tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        # print(filepaths[i], '...done')
    writer.close()

    return filepaths


def _get_records_filename(dir, im_size, timesteps):
    return os.path.join(dir, 'img_data_{}_{}.tfrecords'.format(im_size, timesteps))


def create_record_file_if_required(sess, dir, im_size, timesteps, shuffle_data=True, forceRefresh=False):
    filename = _get_records_filename(dir, im_size, timesteps)
    if forceRefresh:
        return _create_record_file(sess, filename, dir, timesteps, im_size, shuffle_data)
    else:
        try:
            with open(filename):
                pass
            paths_filename = PATHS_FILENAME.format(timesteps)
            with open(os.path.join(dir, paths_filename)) as f:
                lines = [line.strip() for line in f]
            return lines
        except:
            return _create_record_file(sess, filename, dir, timesteps, im_size, shuffle_data)


def ImageReader(filename_queue, ext, ratio=1):
    class FrameRecord:
        pass
    result = FrameRecord()
    with tf.name_scope('Read_image'):
        reader = tf.WholeFileReader()
        result.key, value = reader.read(filename_queue)
        if ext is 'jpg':
            image = tf.image.decode_jpeg(value, ratio=ratio)
        elif ext is 'png':
            image = tf.image.decode_png(value)
    result.image = image
    return result


def generate_frame_batch(filepaths, dir, im_size, timesteps, batch_size, ext='jpg', shuffle_data=True, name=None):
    """ Construct a queued batch of images
        Args:
            timesteps: number of timesteps.
            batch_size: Number of images per batch.
        Returns:
            images: Images. 5-D tensor of [batch_size, timesteps, height, width, 3] size.
    """

    # Create a queue that produces the filepaths to read.
    filename_queue = tf.train.string_input_producer(filepaths, capacity=8 * batch_size, shuffle=shuffle_data)

    # Read sample image to get frame size
    ratio = 1
    with Image.open(filepaths[0]) as img:
        s = img.size[0]
        if s % im_size == 0:
            ratio = int(s / im_size)

    # Read examples from files in the filename queue.
    read_input = ImageReader(filename_queue, ext, ratio=ratio)
    image = read_input.image

    with tf.variable_scope('Preprocessing'):
        s = int(s / ratio)
        image = image[:timesteps * s, :, :]
        image.set_shape([timesteps * s, s, 3])
        image = tf.reshape(image, shape=[timesteps, s, s, 3])
        if im_size != image.get_shape()[0]:
            image = tf.image.resize_images(image, size=tf.constant([im_size, im_size]))
    num_preprocess_threads = 16

    # image_block = tf.train.batch([image],
    #                              batch_size=timesteps,
    #                              num_threads=num_preprocess_threads,
    #                              capacity=8 * timesteps)

    # image_block = tf.transpose(image_block, [1, 0, 2, 3, 4])
    # image = tf.reshape(image_block, shape=[-1, im_size, im_size, 3])
    # First pull in 'timsteps' number of videos
    image_batch = tf.train.batch([image],
                                 batch_size=batch_size,
                                 num_threads=num_preprocess_threads,
                                 capacity=8 * batch_size, name=name)

    return image_batch


def generate_image_batch(filepaths, im_size, batch_size, ext='jpg', shuffle_data=True, name=None):
    """ Construct a queued batch of images
    """

    # Create a queue that produces the filepaths to read.
    filename_queue = tf.train.string_input_producer(filepaths, capacity=8 * batch_size, shuffle=shuffle_data)

    with Image.open(filepaths[0]) as img:
        h, w = img.size[:2]

    # Read examples from files in the filename queue.
    read_input = ImageReader(filename_queue, ext)
    image = read_input.image

    with tf.variable_scope('Preprocessing'):
        image = tf.image.resize_images(image, size=tf.constant([im_size, im_size]))
        image.set_shape((im_size, im_size, 3))
    num_preprocess_threads = 16

    image_batch = tf.train.batch([image],
                                 batch_size=batch_size,
                                 num_threads=num_preprocess_threads,
                                 capacity=8 * batch_size, name=name)

    return image_batch


def read_tfrecords(filepaths, dir, im_size, timesteps, batch_size, shuffle_data=True, name=None):
    tfrecords_filename = _get_records_filename(dir, im_size, timesteps)
    q = tf.train.string_input_producer([tfrecords_filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(q)
    features = tf.parse_single_example(serialized_example, features={'s': tf.FixedLenFeature([], tf.int64),
                                                                     'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [timesteps, im_size, im_size, 3])
    image_batch = tf.train.batch([image],
                                 batch_size=batch_size,
                                 num_threads=16,
                                 capacity=8 * batch_size, name=name)

    return image_batch


def generate_noise_batch(z_shape, batch_size, interpolations=None, min=-1, max=1, name=None):
    """ Generate a random_normal Tensor of given params

        Args:
            z_len: length of Z vector
            batch_size: --
            interpolations: int if interpolations of Z are required
    """

    n = int(np.prod(z_shape))
    Z = tf.random_uniform([n if interpolations else 1, n], min, max)

    if interpolations:
        m = tf.linspace(0., n - 1, n)
        m = tf.matmul(tf.reshape(m, shape=[n, 1]), tf.ones(shape=[1, interpolations]))
        m = tf.reshape(m, shape=[n * interpolations])
        m = tf.cast(m, tf.int32)
        m = tf.one_hot(m, n)
        Z = tf.concat([Z] * interpolations, axis=1)
        Z = tf.reshape(Z, shape=[-1, n])
        l = tf.reshape(tf.linspace(-1., 1., interpolations), shape=[1, -1])
        l = tf.concat([l] * n * n, axis=0)
        l = tf.transpose(tf.reshape(l, shape=[n, n * interpolations]), perm=[1, 0])
        Z = (m * l) + (1 - m) * Z
        batch_size *= interpolations

    Z = tf.reshape(Z, shape=[-1] + z_shape)

    num_preprocess_threads = 16
    batch = tf.train.batch([Z],
                           batch_size=batch_size,
                           num_threads=num_preprocess_threads,
                           enqueue_many=True,
                           capacity=16 * batch_size, name=name)
    return batch


def test_noise_batch():
    Z = generate_noise_batch([4], 12, 3)
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    q = tf.train.start_queue_runners(sess=sess)
    print(Z.eval())
    coord.join(q)
    coord.request_stop()


def generate_epsilon_batch(batch_size, name=None):
    """ Generate a random_normal Tensor of given params

        Args:
            z_len: length of Z vector
            batch_size: --
    """
    E = tf.random_uniform([1])

    num_preprocess_threads = 16
    batch = tf.train.batch([E],
                           batch_size=batch_size,
                           num_threads=num_preprocess_threads,
                           capacity=16 * batch_size, name=name)
    return batch


def generate_linspace_batch(start, stop, n, batch_size, name=None):
    """ Generate a random_normal Tensor of given params

        Args:
            z_len: length of Z vector
            batch_size: --
    """
    L = tf.linspace(start, stop, n, name)

    num_preprocess_threads = 16
    batch = tf.train.batch([L],
                           batch_size=batch_size,
                           num_threads=num_preprocess_threads,
                           capacity=16 * batch_size, name=name)
    return batch


""" Toy datasets
"""


# 1. radial gaussians
def radial_gaussians(batch_size, n_mixture=8, std=0.01, radius=1.0, add_far=False):
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.cos(thetas), radius * np.sin(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)


# 2. radial gaussians 2
def radial_gaussians2(batch_size, n_mixture=8, std=0.01, r1=1.0, r2=2.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    x1s, y1s = r1 * np.sin(thetas), r1 * np.cos(thetas)
    x2s, y2s = r2 * np.sin(thetas), r2 * np.cos(thetas)
    xs = np.vstack([x1s, x2s])
    ys = np.vstack([y1s, y2s])
    cat = ds.Categorical(tf.zeros(n_mixture * 2))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)


# 3. rect
def rect(batch_size, std=0.01, nx=5, ny=5, h=2, w=2):
    x = np.linspace(- h, h, nx)
    y = np.linspace(- w, w, ny)
    p = []
    for i in x:
        for j in y:
            p.append((i, j))
    cat = ds.Categorical(tf.zeros(len(p)))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in p]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)


# 4. swiss roll
def swiss(batch_size, size=1., std=0.01):
    x, _ = datasets.make_swiss_roll(1000)
    norm = x[:, ::2].max()
    xs = x[:, 0] * size / norm
    ys = x[:, 2] * size / norm
    cat = ds.Categorical(tf.zeros(len(x)))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)


# 5. 1D
def line_1d(batch_size, n_mixture=5, std=0.01, d=1.0, add_far=False):
    xs = np.linspace(-d, d, n_mixture, dtype=np.float32)
    p = [0.] * n_mixture
    if add_far:
        xs = np.concatenate([np.array([-10 * d]), xs, np.array([10 * d])], 0)
        p = [0.] + p + [0.]
    cat = ds.Categorical(tf.constant(p))
    comps = [ds.MultivariateNormalDiag([xi], [std]) for xi in xs.ravel()]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)

# End
