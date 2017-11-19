import tensorflow as tf
import ops, utils, coreutils
import os, datetime, time, json, re
import logging as log
import numpy as np
from abc import ABCMeta, abstractmethod


class _multiGPUmodel(metaclass=ABCMeta):

    def __init__(self, sess, dataset_name, batch_size, num_gpus, config, data_dir=None, output_dir='.', var_summ=True, grad_summ=True, send_email=False):

        self.sess = sess
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.send_email = send_email
        self.config = config
        self.var_summ = var_summ
        self.grad_summ = grad_summ
        self._setup_config(config)
        self.model_name = config['model_name'] if 'model_name' in config is not None else self.__class__.__name__

        self.run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        print("Current run id : ", self.run_id)

        # Required directory names... we will create them as needed
        self.train_dir = output_dir + '/training_data/{}/{}'.format(self.dataset_name, self.run_id)
        self.log_path = self.train_dir + '/logs/'
        self.samples_path = self.train_dir + '/samples/'
        self.saves_path = output_dir + '/saves/{}_{}'.format(self.model_name, self.loss)

        print(self.log_path)
        self._extra_init()
        self._init_log()
        self.Inputs = self._setup_placeholders()
        self.model_built = False

    @abstractmethod
    def _setup_config(self, config):
        pass

    def _extra_init(self):
        pass

    def _init_log(self):
        # Log Config
        utils.make_sure_path_exits(self.log_path)
        log.basicConfig(level=log.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='{}/info.log'.format(self.log_path),
                        filemode='w')

        config = self._get_serializable_config()
        if config:
            with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
                json.dump(config, f, indent=4)

    def _get_serializable_config(self):
        return None

    @abstractmethod
    def _setup_placeholders(self):
        # Create placeholders and return a list of placeholders that are to be devided among the gpus
        pass

    @abstractmethod
    def _build_train_tower(self, inputs, batch_size=None, reuse=False):
        # Inputs are the splits of the inputs returned in _setup_placeholders()
        # Returns a tuple of sampler, list of losses, list of grads
        pass

    def _remove_tower_name_prefix(self, x):
        return re.sub('tower_[0-9]*/', '', x.op.name)

    def additional_summaries(self):
        """ Return additional summaries 'list' if required
        """
        return []

    def build(self):
        """ Builds a multi-tower model
        """
        with tf.device('/cpu:0'):
            assert self.batch_size % self.num_gpus == 0, ('Batch size must be divisible by number of GPUs')

            with tf.name_scope('Input_splits'):
                tower_inputs = [[] for i in range(self.num_gpus)]
                for inp in self.Inputs:
                    splits = tf.split(inp, self.num_gpus, name=inp.name[:-2])
                    for i, s in enumerate(splits):
                        tower_inputs[i].append(s)

            tower_outputs = []
            tower_losses   = []
            tower_grads    = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            # Calculate the loss for one tower of the model. This function
                            # constructs the entire model but shares the variables across
                            # all towers.
                            outputs, losses, grads = self._build_train_tower(tower_inputs[i],
                                                                             int(self.batch_size / self.num_gpus),
                                                                             reuse=i > 0 or self.model_built)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Save summaries from tower_1
                            if i == 0:
                                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            tower_outputs.append(outputs)
                            tower_losses.append(losses)
                            tower_grads.append(grads)

            with tf.name_scope('Concat_outputs'):
                outputs = [[] for _ in tower_outputs[0]]
                for t_outputs in tower_outputs:
                    for i, output in enumerate(t_outputs):
                        outputs[i].append(output)
                self.outputs = []
                for outs in outputs:
                    self.outputs.append(tf.concat(outs, 0))

            with tf.name_scope('Concat_losses'):
                losses = [[] for _ in range(len(tower_losses[0]))]
                for t_losses in tower_losses:
                    for i, loss in enumerate(t_losses):
                        losses[i].append(loss)

            with tf.name_scope('Average_grads'):
                var_grads = [[] for _ in range(len(tower_grads[0]))]
                for t_grads in tower_grads:
                    for i, grad in enumerate(t_grads):
                        var_grads[i].append(grad)
                avg_grads = []
                for v_grads in var_grads:
                    avg_grads.append(ops.average_gradients(v_grads))

            if self.grad_summ:
                # Add histograms for gradients.
                with tf.name_scope('Grad_summary'):
                    grads_summ = []
                    for var_grads in avg_grads:
                        for grad, var in var_grads:
                            if grad is not None:
                                grads_summ.append(tf.summary.histogram(self._remove_tower_name_prefix(var) + '/Grads', grad))
                    summaries.append(tf.summary.merge(grads_summ))

            if self.var_summ:
                # Add histograms for trainable variables.
                t_vars = tf.trainable_variables()
                with tf.name_scope('Var_summary'):
                    vars_summ = []
                    for var in t_vars:
                        vars_summ.append(tf.summary.histogram(self._remove_tower_name_prefix(var), var))
                    summaries.append(tf.summary.merge(vars_summ))

            summaries += self.additional_summaries()

            self._tower_outputs(self.outputs)
            self._build_train_ops(losses, avg_grads)
            self.summary_op = tf.summary.merge(summaries, name='summary_op')
            self.saver = tf.train.Saver()
            self.model_built = True
            utils.count_params()

    def _tower_outputs(self, outputs):
        pass

    @abstractmethod
    def _build_train_ops(self, losses, grads):
        """ Build and store the train ops as properties of 'self'
            Average of the losses and grads returned from _build_train_tower from among the towers
        """
        pass

    def _print_updates(self, delta_time, step, total_batches, epoch, epoch_time):
        sec_left = 0
        eta_str = '--'
        if delta_time > 0:
            left_steps = total_batches - step
            sec_left = left_steps * delta_time
            eta = datetime.datetime.now() + datetime.timedelta(0, sec_left)
            eta_str = eta.strftime("%a, %d %b %H:%M:%S")
        pre = '[Epoch:{:2d}]'.format(epoch)
        elapsed_secs = time.time() - epoch_time
        post = eta_str + ' ({} ; {:.1f}s)'.format(utils.strftimedelta(elapsed_secs), delta_time)
        utils.printProgressBar(step, total_batches, pre, post, length=50)
        return sec_left

    def _load_vars(self):
        """ A function that is called before training, use it to load any pretrained vars
        """
        pass

    @abstractmethod
    def _get_data_dict(self, counter):
        """ Returns a data dictionary for this step
        """
        pass

    @abstractmethod
    def _run_step(self, counter, data_dict):
        """ Run the train step and return the summary string
        """
        pass

    @abstractmethod
    def generate_sample(self, counter):
        """ Use self.sampler to generate samples
        """
        pass

    @abstractmethod
    def _setup_input_pipe(self):
        """ Create input sources and return the number of training examples
        """
        pass

    def train(self, epochs, st_epoch=1, samples_per_epoch=10, saves_per_epoch=1):
        with tf.device('/cpu:0'):
            self.epochs = epochs
            self.samplefile_paths = None  # For saving last samples generated

            self._load_vars()

            print('Run id : ', self.run_id)
            print("[Model : {}]".format(self.model_name))
            n = self._setup_input_pipe()

            print('Creating training summaries...')
            with tf.name_scope('Training_summary'):
                self.ETA_HOURS = tf.placeholder(tf.float32, shape=(), name="ETA_HOURS")
                self.OP_TIME = tf.placeholder(tf.float32, shape=(), name='OP_TIME')
                self.DB_TIME = tf.placeholder(tf.float32, shape=(), name='DB_TIME')
                eta_summary = tf.summary.scalar("eta_hours", self.ETA_HOURS)
                op_time_summay = tf.summary.scalar('op_time', self.OP_TIME)
                db_time_summay = tf.summary.scalar('db_time', self.DB_TIME)
                self.training_summary = tf.summary.merge([eta_summary, op_time_summay, db_time_summay], collections='training_summary')

            print('Creating summary writer...')
            self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

            print('Starting queue runners...')
            self.coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=self.coord)
            print('Starting training for {} epochs...'.format(epochs))
            try:
                start_time = time.time()
                self._train_fn(n, epochs, st_epoch, samples_per_epoch, saves_per_epoch)
                print("Training completed in", utils.strftimedelta(time.time() - start_time), "seconds...")

            except Exception as e:
                if self.send_email:
                    coreutils.send_email('shashank.879@gmail.com',
                                         'Training Error', str(e))
                print(e)
                raise
            self.coord.request_stop()
            self.coord.join(threads)
            self._post_train()
            print('Check log at...', self.log_path)

    def _train_fn(self, n_items, epochs, st_epoch, samples_per_epoch, saves_per_epoch):
        self.total_batches = int(n_items / self.batch_size)
        print('Total batches...', self.total_batches)
        sample_idx = int(self.total_batches / samples_per_epoch)
        save_idx = int(self.total_batches / saves_per_epoch)
        counter = (st_epoch - 1) * self.total_batches + 1
        for epoch in range(st_epoch, epochs + 1):
            epoch_time = time.time()
            self._print_updates(0, 0, self.total_batches, epoch, epoch_time)
            idx = 1
            while idx < self.total_batches + 1:
                if self.coord.should_stop():
                    break

                # Fetch train batch
                st_time = time.time()
                data_dict = self._get_data_dict(counter)
                db_time = time.time() - st_time
                st_time = time.time()
                update_str = self._run_step(counter, data_dict)
                op_time = time.time() - st_time

                secs_left = self._print_updates(op_time, idx, self.total_batches, epoch, epoch_time)
                tr_summ_str = self.sess.run(self.training_summary, feed_dict={self.ETA_HOURS: secs_left / 3600,
                                                                              self.OP_TIME: op_time,
                                                                              self.DB_TIME: db_time})
                self.writer.add_summary(tr_summ_str, global_step=counter)

                log.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, %s" % (epoch, idx, self.total_batches, op_time, update_str))

                # Create samples with error log if required
                if np.mod(idx, sample_idx) == 0:
                    utils.make_sure_path_exits(self.samples_path)
                    self.samplefile_paths = self.generate_sample(counter)
                    log.info("[*] Sample")

                # Save model with avg errors logged
                if np.mod(idx, save_idx) == 0:
                    self.save(counter)
                    log.info('[*] Model saved')

                idx += 1
                counter += 1

            if self.send_email:
                coreutils.send_email('shashank.879@gmail.com',
                                     'Epoch completed',
                                     'Generated samples',
                                     attachments=self.samplefile_paths)

    def _post_train(self):
        pass

    def save(self, step):
        model_dir = self.dataset_name
        chkpt_dir = os.path.join(self.saves_path, model_dir)
        utils.make_sure_path_exits(chkpt_dir)
        self.saver.save(self.sess,
                        os.path.join(chkpt_dir, self.model_name + '.ckpt'),
                        global_step=step)

    def load(self):
        print("[*] Reading checkpoints...")
        model_dir = self.dataset_name
        chkpt_dir = os.path.join(self.saves_path, model_dir)
        chkpt = tf.train.get_checkpoint_state(chkpt_dir)

        if chkpt and chkpt.model_checkpoint_path:
            chkpt_name = os.path.basename(chkpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(chkpt_dir, chkpt_name))
            log.info("[*] Successfully restored... ", chkpt_name)
            return True
        else:
            log.info("[*] Failed to find the checkpoint...")
            return False

# End
