import tensorflow as tf
import numpy as np
from v1 import utils, params

from vggish import vggish_params
from vggish import vggish_slim

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

            conv1 = slim.conv2d(embeddings, 1024, scope="conv1",kernel_size=[3, 3], stride=1, padding='SAME')

            pool1 = slim.avg_pool2d(conv1, scope='pool1',kernel_size=[2, 2], stride=2, padding='SAME')

            pool1 = slim.flatten(pool1)

            fc1 = tf.contrib.layers.fully_connected(inputs=pool1, num_outputs=512,
                                                    activation_fn=None, scope="fc1")

            bn1 = tf.layers.batch_normalization(fc1, 1, name="batch_norm_1")

            fc2 = tf.contrib.layers.fully_connected(inputs=bn1, num_outputs=vggish_params.EMBEDDING_SIZE,
                                                    activation_fn=tf.nn.relu, scope="fc2")

            bn2 = tf.layers.batch_normalization(fc2, 1, name="batch_norm_2")

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.

            logits = tf.contrib.layers.fully_connected(
                bn2, params.NUM_CLASSES, activation_fn=None, scope='logits')

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
                xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
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

            with tf.variable_scope("accuracy"):
                with tf.variable_scope("correct_prediction"):
                    correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))
                with tf.variable_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            tf.summary.scalar("accuracy", accuracy)

    return graph, accuracy, softmax_prediction


def train(X_train, Y_train, X_test, Y_test, test_fold, num_epochs=100, minibatch_size=params.BATCH_SIZE,
          save_checkpoint=True):
    m = X_train.shape[0]

    graph, accuracy_tensor, softmax_prediction = model(learning_rate=0.01)

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

        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        # Init summary writer
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter("./logs/train/fold_" + str(test_fold), sess.graph)

        test_writer = tf.summary.FileWriter("./logs/test/fold_" + str(test_fold), sess.graph)

        # Init checkpoint saver
        saver = tf.train.Saver()

        tf.global_variables_initializer().run()

        chekpoint = tf.train.latest_checkpoint(checkpoint_dir=params.CHECKPOINT_FOLDER + str(test_fold))

        if chekpoint is not None:
            print("Checkpoint exists. Loading from disk..")
            saver.restore(sess, chekpoint)

        for epoch in range(num_epochs):
            minibatch_cost = 0.
            batch_accuracy_average = 0
            print("Epoch: %d" % epoch)
            # number of minibatches of size minibatch_size in the train set

            minibatches = utils.random_mini_batches(X_train, Y_train, minibatch_size)

            num_minibatches = len(minibatches)

            # for minibatch in minibatches:
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                [summary_train, num_steps, loss, _] = sess.run(
                    [merged, global_step_tensor, loss_tensor, train_op],
                    feed_dict={features_tensor: minibatch_X, labels_tensor: minibatch_Y})

                minibatch_cost += loss / num_minibatches

                print('Step %d: loss %g minibatch_cost: %g' % (num_steps, loss, minibatch_cost))

                if epoch % 10 == 0:
                    accuracy = sess.run(accuracy_tensor,
                                        feed_dict={features_tensor: minibatch_X, labels_tensor: minibatch_Y})
                    batch_accuracy_average += accuracy / num_minibatches

                train_writer.add_summary(summary_train, num_steps)
                train_writer.flush()

            summary_test, test_accuracy = sess.run([merged, accuracy_tensor],
                                                   feed_dict={features_tensor: X_test, labels_tensor: Y_test})

            test_writer.add_summary(summary_test, num_steps)

            print("batch cost: %g" % minibatch_cost)

            if epoch % 10 == 0:
                print("batch accuracy: %g" % batch_accuracy_average)

            print("test_acc: %g" % test_accuracy)

            if save_checkpoint and epoch % 200 == 0 and epoch > 0:
                saver.save(sess, params.CHECKPOINT_FOLDER + str(test_fold) + "/checkpoint.ckpt", num_steps)
                print("Checkpoint saved")

        print("Training has finished!")


def calc_acc(prediction_op, input_x, input_y, name, features_tensor, labels_tensor, sess):
    correct_prediction = tf.equal(prediction_op, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name=name)
    variable_summaries(accuracy)
    t_accuracy = accuracy.eval({features_tensor: input_x, labels_tensor: input_y}, sess)
    return t_accuracy


def main():
    test_fold = 1

    train_data = np.load("dataset/train_data_fold_" + str(test_fold) + ".npy")
    train_labels = np.load("dataset/train_labels_fold_" + str(test_fold) + ".npy")

    test_data = np.load("dataset/test_data_fold_" + str(test_fold) + ".npy")
    test_labels = np.load("dataset/test_labels_fold_" + str(test_fold) + ".npy")

    train(train_data, train_labels, test_data, test_labels, test_fold, 1401, save_checkpoint=False)


if __name__ == '__main__':
    main()
