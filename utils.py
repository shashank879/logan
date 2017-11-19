import cv2
import numpy as np
import os, math, asyncio, glob, csv
import tensorflow as tf
from stat import S_ISDIR
import ops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_sure_path_exits(path):
    os.makedirs(path, exist_ok=True)


def resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)


def center_crop(image):
    h = image.shape[0]
    w = image.shape[1]
    d = int((w - h) / 2)
    return image[:, d:(w - d)] if d > 0 else image[-d:(h + d), :]


def save_images_h(path, images, imagesPerRow=16):
    n, h, w, c = images.shape
    rows = int(n / imagesPerRow) + 1 * ((n % imagesPerRow) != 0)
    img = np.zeros((h * rows, w * imagesPerRow, c), dtype=np.float32)
    for idx, image in enumerate(images):
        i = idx % imagesPerRow
        j = idx // imagesPerRow
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_images_v(path, images, imagesPerColumn=16):
    n, h, w, c = images.shape
    colums = int(n / imagesPerColumn)
    img = np.zeros((h * imagesPerColumn, w * colums, c), dtype=np.float32)
    for idx, image in enumerate(images):
        i = idx // imagesPerColumn
        j = idx % imagesPerColumn
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def extract_recons(img_path, save_folder, indices, offset):
    img = cv2.imread(img_path)
    s = int(img.shape[0] / 3)
    for i, j in enumerate(indices):
        slice = img[s:, s * j:s * (j + 1), :]
        cv2.imwrite(os.path.join(save_folder, str(i + offset) + '.png'), slice)


def extract_gens(img_path, save_folder, indices, offset):
    img = cv2.imread(img_path)
    s = int(img.shape[0] / 3)
    for i, j in enumerate(indices):
        slice = img[:s, s * j:s * (j + 1), :]
        cv2.imwrite(os.path.join(save_folder, str(i + offset) + '.png'), slice)


def read_images(imgpaths):
    imgs = []
    for path in imgpaths:
        img = cv2.imread(path)
        if img is None:
            print(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
    return np.array(imgs)


def plot_inception_scores(outfile, csvfiles, labels):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set(xlabel='epochs', ylabel='score')
    for i, path in enumerate(csvfiles):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array(list(reader))[1:, :]
            x = np.linspace(0, len(data) - 1, len(data))
            ax.plot(x, data[:, 0], label=labels[i])
    ax.legend()
    fig.savefig(outfile)
    plt.close()


def plot_inception_scores_mov_avg(outfile, csvfiles, labels):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set(xlabel='epochs', ylabel='score')
    last_values = []
    for i, path in enumerate(csvfiles):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array(list(reader)[1:], dtype=np.float32)[:, 0]
            data = np.convolve(data, [1. / 20] * 20, 'valid')
            x = np.linspace(0, len(data) - 1, len(data))
            ax.plot(x, data, label=labels[i])
            last_values.append(data[-1])
    ax.legend()
    fig.savefig(outfile)
    plt.close()
    return last_values


def save_train_curve(outfile, d_csv, d__csv):
    with open(d_csv) as f:
        reader = csv.reader(f)
        d = np.array(list(reader)[25:], dtype=np.float32)[:, 2]
    with open(d__csv) as f:
        reader = csv.reader(f)
        d_ = np.array(list(reader)[25:], dtype=np.float32)[:, 2]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    fig = plt.figure()
    ax = plt.subplot(111)
    # diff = sigmoid(d) - sigmoid(d_)
    diff = d - d_
    diff_avg = np.convolve(diff, [1. / 40] * 40, 'same')
    d_avg = np.convolve(d, [1. / 40] * 40, 'same')
    logit_sum = d + d_
    x = np.linspace(0, len(diff) - 1, len(diff))
    ax.plot(x, diff, color='orange', alpha=0.5)
    ax.plot(x, diff_avg, color='orange', label='convergence')
    ax.plot(x, d, color='blue', alpha=0.5)
    ax.plot(x, d_avg, color='blue', label='D_fake')
    ax.set(ylim=[0., 2.])
    # ax.plot(x, logit_sum, label='logit_sum')
    ax.legend()
    fig.savefig(outfile)
    plt.close()


def get_gens_of_index(outfile, s, i, j, folder):
    k = 1
    files = []
    while True:
        filename = './training_data/celeb_a/20170910-0050/samples/test/gens/' + str(k * 2000) + '.png'
        try:
            with open(filename):
                files.append(filename)
        except:
            break
        k += 1

    imgs = read_images(files)
    extracted = imgs[:, i * s:(i + 1) * s, j * s:(j + 1) * s, :]
    save_images_h(outfile, extracted, 10)


def plot_bigan_inception_scores(outpath, sessions, labels):
    fig = plt.figure()
    inc_files = [os.path.join(folder, 'logs/inc_score.csv') for folder in sessions]
    ax = plt.subplot(111)
    ax.set(xlabel='epochs', ylabel='score')
    for i, path in enumerate(inc_files):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array(list(reader))[1:, :]
            x = np.linspace(0, len(data) - 1, len(data))
            ax.plot(x, data[:, 0], label=labels[i])
    ax.legend()
    fig.savefig(os.path.join(outpath, 'bi_inc_plots.png'))
    plt.clf()

    diff_files = [os.path.join(folder, 'logs/diff_score.csv') for folder in sessions]
    ax = plt.subplot(111)
    ax.set(xlabel='epochs', ylabel='score')
    for i, path in enumerate(diff_files):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array(list(reader)[1:], dtype=np.float32)[:, 0]
            data = np.convolve(data, [1. / 20] * 20, 'valid')
            print(data.shape, data.dtype)
            x = np.linspace(0, len(data) - 1, len(data))
            ax.plot(x, data, label=labels[i])
    ax.legend()
    fig.savefig(os.path.join(outpath, 'bi_inc_diff_plots.png'))
    plt.close()


def get_best_recons(outfile, imgfile, num=64):
    from skimage.measure import compare_ssim as ssim
    samples = read_images([imgfile])[0]
    s = samples.shape[0] // 2
    n = samples.shape[1] // s
    all_x = []
    all_r = []
    all_d = []
    for i in range(n):
        x = samples[:s, i * s:(i + 1) * s, :]
        r = samples[s:, i * s:(i + 1) * s, :]
        d = -ssim(x, r, multichannel=True)
        all_x.append(x)
        all_r.append(r)
        all_d.append(d)
    all_d = np.array(all_d)
    d_index = np.argsort(all_d)
    out_imgs = []
    for i in range(n):
        if i < num:
            j = d_index[i] - 1
            img = np.concatenate([all_x[j], all_r[j]], 1)
            out_imgs.append(img)
    save_images_h(outfile, np.array(out_imgs), int(np.sqrt(num)))

def get_interpolations(sess, bigan_model, filepaths, num, outfolder):
    imgs = read_images(filepaths)
    b, _, s, _ = imgs.shape
    b = int(b / 2)
    imgs = imgs[:, :s, :, :]
    print('Calculating latent representations...')
    L = sess.run(bigan_model.L, feed_dict={bigan_model.X: imgs})
    imgs1, imgs2 = imgs[:b], imgs[b:]
    l1, l2 = L[:b], L[b:]
    img_inters = []
    e_range = np.linspace(0, 1., num)
    print('Calculating interpolated frames...')
    for i, e in enumerate(e_range):
        l = l1 * (1. - e) + l2 * e
        img = sess.run(ops.inverse_transform(bigan_model.G), feed_dict={bigan_model.Z: l})
        img_inters.append(img)
        print(i)
    img_all = np.concatenate(img_inters, 0)
    save_images_v(os.path.join(outfolder, 'interpolations.png'), img_all, b)
    save_images_v(os.path.join(outfolder, 'imgs1.png'), imgs1, b)
    save_images_v(os.path.join(outfolder, 'imgs2.png'), imgs2, b)


def sftp_walk(remotepath, sftp):
        path = remotepath
        files = []
        folders = []
        for f in sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(os.path.join(remotepath, f.filename))
            else:
                files.append(os.path.join(remotepath, f.filename))
        if files:
            yield path, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder)
            for x in sftp_walk(new_path, sftp):
                yield x


PATHS_FILENAME = 'img_paths.txt'


class Data_loader:
    """ Loads the data from server
    """
    def __init__(self, dir_path, batch_size=100, timesteps=32, out_size=None, refresh_paths_file=False, shuffle_data=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.out_size = out_size
        self.path_filename = os.path.join(dir_path, PATHS_FILENAME)

        if refresh_paths_file:
            create_path_list(self.dir_path, timesteps, shuffle_data)
        else:
            try:
                f = open(self.path_filename)
                i = 0
                for l in f:
                    i += 1
                f.close()
                self.total_batches = int(i / self.batch_size)
            except:
                create_path_list(self.dir_path, timesteps, shuffle_data)

    def __enter__(self):
        self.path_file = open(self.path_filename)

    def __exit__(self, exception_type, exception_value, traceback):
        self.path_file.close()

    def fetch_next_batch(self):
        i = 0
        tasks = []
        while i < self.batch_size:
            filepath = self.path_file.readline().strip()
            tasks.append(_get_formatted_image(filepath, self.timesteps, self.out_size))
            i += 1
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.get_event_loop()
        imgs = loop.run_until_complete(asyncio.gather(*tasks))

        return np.array(imgs)


async def _get_formatted_image(filename, timesteps, out_size):
    img = cv2.imread(filename)
    s = img.shape[1]
    c = img.shape[2]
    # truncate height
    img = img[:timesteps * s, :, :]
    if out_size:
        img = resize(img, (timesteps * out_size[0], out_size[1]))
    imgs = img.reshape([timesteps, out_size[0], out_size[1], c])
    return ops.transform(imgs)


async def _image_ok(filename, timesteps):
    img = cv2.imread(filename)
    sh = img.shape
    return (filename, sh[0] / sh[1] >= timesteps)


def create_path_list(dir, timesteps, ext='jpg', precheck=False, paths_filename=PATHS_FILENAME):
    assert ext in ['jpg', 'png'], 'incorrect extension'
    print("Creating paths file...")
    dir = dir if dir.endswith('/') else (dir + '/')
    img_paths = []
    for filename in glob.iglob(dir + '**/*.{}'.format(ext), recursive=True):
        img_paths.append(filename)
    print("Done collecting paths... found ", len(img_paths))

    if precheck:
        print("Checking for validity of each...")
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.get_event_loop()
        tasks = [_image_ok(fn, timesteps) for fn in img_paths]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
        print("Done validating...")
        ok_files = []
        for i, (filename, ok) in enumerate(results):
            if ok:
                ok_files.append(filename)
        img_paths = ok_files
        print("Found ", len(img_paths), " ok files...")

    f = open(os.path.join(dir, paths_filename), 'w')
    f.write('\n'.join(img_paths))
    f.close()
    print('Paths file created at... ', os.path.join(dir, paths_filename))
    return img_paths


def count_params():
    "print number of trainable variables"
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Model size: %d" % (total_parameters,))


def check_model_for_loose_modules(model_in, model_out):
    """Computes the gradients and checks if any is None
        Args:
        model_in can be a list
    """

    grads_and_vars = tf.gradients(model_out, model_in)
    ok = True
    for gv in grads_and_vars:
        for g, v in gv:
            if g is None:
                print(v.name, "Output not used")
                ok = False
    return ok


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def strftimedelta(delta):
    hours, left = divmod(delta, 3600)
    mins, left = divmod(left, 60)
    secs = left
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(mins), int(secs))


def add_conv_summary():
    with tf.name_scope('Conv_summary'):
        t_vars = tf.trainable_variables()
        summs = []
        for var in t_vars:
            shape = var.get_shape().as_list()
            if len(shape) == 4 and shape[2] == 3:
                c_out = shape[3]
                f = tf.split(var, c_out, axis=3)
                h = int(math.sqrt(c_out))
                while c_out % h is not 0:
                    h -= 1
                w = int(c_out / h)
                rows = []
                for i in range(h):
                    rows.append(tf.concat(f[i * w:(i + 1) * w], axis=1))
                rect = tf.squeeze(tf.concat(rows, axis=0))
                summ = tf.summary.image(var.name[:-2], tf.reshape(rect, shape=[1] + rect.get_shape().as_list()))
                summs.append(summ)
        return tf.summary.merge(summs, name='conv_summ')


def flatten(arr):
    return [j for i in arr for j in i]


# End
