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


def model(learning_rate=vggish_params.LEARNING_RATE, training=FLAGS.train_vggish):
    graph = tf.Graph()

    with graph.as_default():
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(training)

        with tf.variable_scope("mymodel"):
            # Add a fully connected layer with 100 units.
            num_units = 100

            # fc = tf.contrib.layers.fully_connected(inputs=embeddings, num_outputs=num_units,
            #                                        activation_fn=tf.nn.sigmoid)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.

            logits = tf.contrib.layers.fully_connected(
                embeddings, params.NUM_CLASSES, activation_fn=None, scope='logits')

            prediction = tf.argmax(logits, axis=1, name='prediction')

            softmax_prediction = tf.nn.softmax(logits, axis=1, name="softmax_prediction")
            softmax_prediction = tf.nn.top_k(softmax_prediction, k=5)

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
                # accuracy = tf.metrics.accuracy(labels=labels, predictions=logits, name="acc")

                # tf.summary.scalar('accuracy', accuracy)
                # variable_summaries(accuracy)

                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')

    return graph, prediction, softmax_prediction


def train(X_train, Y_train, X_test, Y_test, test_fold, num_epochs=100, minibatch_size=params.BATCH_SIZE,
          save_checkpoint=True):
    m = X_train.shape[0]

    graph, prediction_op, softmax_prediction = model(learning_rate=0.001)

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

        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        # Init summary writer
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./logs/train/fold_" + str(test_fold), sess.graph)

        # Init checkpoint saver
        saver = tf.train.Saver()

        tf.global_variables_initializer().run()

        chekpoint = tf.train.latest_checkpoint(checkpoint_dir=params.CHECKPOINT_FOLDER + str(test_fold))

        costs = []

        if chekpoint is not None:
            print("Checkpoint exists. Loading from disk..")
            saver.restore(sess, chekpoint)

        for epoch in range(num_epochs):
            minibatch_cost = 0.
            print("Epoch: %d" % epoch)
            # number of minibatches of size minibatch_size in the train set
            num_minibatches = int(m / minibatch_size)

            minibatches = utils.random_mini_batches(X_train, Y_train, minibatch_size)

            # for minibatch in minibatches:
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                [summary_str, num_steps, loss, _] = sess.run(
                    [summary, global_step_tensor, loss_tensor, train_op],
                    feed_dict={features_tensor: minibatch_X, labels_tensor: minibatch_Y})

                summary_writer.add_summary(summary_str, num_steps)
                summary_writer.flush()

                minibatch_cost += loss / num_minibatches
                print('Step %d: loss %g minibatch_cost: %g' % (num_steps, loss, minibatch_cost))

            costs.append(minibatch_cost)

            avg = sum(costs) / float(len(costs))

            print("Average cost: %g" % avg)

        if save_checkpoint:
            saver.save(sess, params.CHECKPOINT_FOLDER + str(test_fold) + "/checkpoint.ckpt", num_steps)
            print("Checkpoint saved")

        print("Training has finished!")


def main():
    test_fold = 1

    train_data = np.load("dataset/train_data_fold_" + str(test_fold) + ".npy")
    train_labels = np.load("dataset/train_labels_fold_" + str(test_fold) + ".npy")

    test_data = np.load("dataset/test_data_fold_" + str(test_fold) + ".npy")
    test_labels = np.load("dataset/test_labels_fold_" + str(test_fold) + ".npy")

    train(train_data, train_labels, test_data, test_labels, test_fold, 100, save_checkpoint=False)


if __name__ == '__main__':
    main()
