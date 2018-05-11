import os

import tensorflow as tf
import numpy as np
import utils

from vggish import vggish_input, vggish_params
from vggish import vggish_slim
import params

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_boolean(
    'train_vggish', False,
    'If Frue, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', './vggish/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def model(learning_rate=vggish_params.LEARNING_RATE):
    graph = tf.Graph()

    with graph.as_default():
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

        with tf.variable_scope("mymodel"):
            # Add a fully connected layer with 100 units.
            num_units = 100

            fc = slim.fully_connected(embeddings, num_units)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
                fc, params.NUM_CLASSES, activation_fn=None, scope='logits')

            prediction = tf.argmax(logits, name='prediction')

            # Add training ops.
            with tf.variable_scope('train'):
                global_step = tf.Variable(
                    0, name='global_step', trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.GLOBAL_STEP])

                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels = tf.placeholder(
                    tf.float32, shape=(None, params.NUM_CLASSES), name='labels')

                # Cross-entropy label loss.
                xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)
                variable_summaries(loss)

                # Calculate accuracy
                #accuracy = tf.metrics.accuracy(labels=labels, predictions=logits, name="acc")

                #tf.summary.scalar('accuracy', accuracy)
                #variable_summaries(accuracy)

                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')

    return graph, prediction


def train(filenames, file_labels, num_epochs=100, minibatch_size=params.BATCH_SIZE):
    m = len(filenames)

    graph, prediction_op = model()

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with graph.as_default(), tf.Session(graph=graph) as sess:

        # Initialize all variables in the model, and then load the pre-trained
        # VGGish checkpoint.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        all_tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print(all_tensors)
        #accuracy_tensor = sess.graph.get_tensor_by_name('mymodel/train/accuracy_0:0')

        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        # Init summary writer
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./logs/train/fold" + str(params.TEST_FOLD), sess.graph)

        # Init checkpoint saver
        saver = tf.train.Saver()

        tf.global_variables_initializer().run()

        chekpoint = tf.train.latest_checkpoint(checkpoint_dir=params.CHECKPOINT_FOLDER)

        if chekpoint is not None:
            print("Checkpoint exists. Loading from disk..")
            saver.restore(sess, chekpoint)

        for epoch in range(num_epochs):
            minibatch_cost = 0.

            # number of minibatches of size minibatch_size in the train set
            num_minibatches = int(m / minibatch_size)
            minibatches = utils.make_random_batches(filenames, file_labels, minibatch_size)

            # for minibatch in minibatches:
            for minibatch in minibatches:
                filenames_batch, labels_batch = minibatch
                minibatch_X, minibatch_Y = utils.load_data(filenames_batch, labels_batch)
                [summary_str, num_steps, loss, _] = sess.run(
                    [summary, global_step_tensor, loss_tensor, train_op],
                    feed_dict={features_tensor: minibatch_X, labels_tensor: minibatch_Y})

                summary_writer.add_summary(summary_str, num_steps)
                summary_writer.flush()

                minibatch_cost += loss / num_minibatches
                print('Step %d: loss %g ' % (num_steps, loss))

            if epoch % 5 == 0:
                saver.save(sess, params.CHECKPOINT_FOLDER + "/checkpoint.ckpt", epoch)
                print("Checkpoint saved")

        print("Training has finished!")

    return prediction_op


class_map, filenames, labels = utils.read_csv()

_, test_filenames, test_labels = utils.read_csv(test_set=True)

prediction_op = train(filenames, labels, 10)
