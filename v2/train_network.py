import numpy as np
import scipy as scp
import scipy.io
import tensorflow as tf
from v2 import utils
from v2 import params

# data = scp.io.loadmat("features.mat")

"""
shape of data['features'] array:
    'NUM_FOLDS x NUM_EXAMPLES_IN_FOLD x LAST_LAYER_DIM x SEGMENTS x 1' 
     or 5      x       400            x     1024       x    5     x 1 
    
shape of data['labels'] array:
        NUM_FOLDS x NUM_EXAMPLES_IN_FOLD x 1
        or 5x400x1
"""


def get_folder_name(fold_number, suffix, prefix, test=False):
    prefix = "./logs/" + prefix + ("/test/fold_" if test else "/train/fold_")
    return prefix + str(fold_number) + "/" + suffix


def avg(l):
    return sum(l) / float(len(l))


def transpose_features(features):
    return np.transpose(features, [0, 3, 2, 1])


class Model:
    def __init__(self, cost, labels, features, prediction, top_5, global_step, max_accuracy):
        self.cost = cost
        self.top_5 = top_5
        self.labels = labels
        self.features = features
        self.prediction = prediction
        self.global_step = global_step
        self.max_accuracy = max_accuracy

    def update_accuracy(self, test_accuracy):
        self.max_accuracy = test_accuracy if test_accuracy > self.max_accuracy else self.max_accuracy


def get_model():
    with tf.variable_scope("model"):
        K = 5  # number of segments
        C_t = 50  # number of classes
        n_last_layer = 1024  # number of elements in last layer of base model
        max_accuracy = 0

        features = tf.placeholder(dtype=tf.float32, shape=(None, 1, K, n_last_layer), name='features')
        labels = tf.placeholder(dtype=tf.float32, shape=(None, C_t), name='labels')

        global_step = tf.Variable(0, name='global_step', trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        F1 = tf.layers.conv2d(features, filters=512, kernel_size=[1, 1], padding="same", name="F1",
                              activation=tf.nn.relu)

        F1_dropout = tf.layers.dropout(F1, name="F1_dropout")

        global_pooling = tf.layers.max_pooling2d(F1_dropout, pool_size=[1, 5], strides=1, name="global_pooling")

        pool_dropout = tf.layers.dropout(global_pooling, name="pool_dropout")

        logits = tf.contrib.layers.fully_connected(pool_dropout, num_outputs=C_t, scope="logits",
                                                   activation_fn=None)
        logits = tf.reshape(logits, [-1, 50])

        prediction = tf.argmax(logits, name='prediction', axis=1)

        top_5 = tf.nn.softmax(logits, axis=1, name="top_5")

        top_5 = tf.nn.top_k(top_5, k=5)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='loss')

        cost = tf.reduce_mean(loss, name="cost_op")

    return Model(cost, labels, features, prediction, top_5, global_step, max_accuracy)


def train(train_features, train_labels, test_features, test_labels,
          learning_rate=0.0002, num_epochs=60, mini_batch_size=128, test_fold=5, make_logs=True, save_checkpoint=True):
    graph = tf.Graph()

    with graph.as_default():

        model = get_model()

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(model.cost, name='train_op', global_step=model.global_step)

        with tf.variable_scope("accuracy"):
            with tf.variable_scope("correct_prediction"):
                correct_prediction = tf.equal(model.prediction, tf.argmax(model.labels, 1))
            with tf.variable_scope("accuracy_op"):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.variable_scope("summary"):
            tf.summary.scalar("accuracy", accuracy)
            tf.summary.histogram("accuracy_hist", accuracy)
            tf.summary.scalar("cost", model.cost)
            tf.summary.histogram("cost_hist", model.cost)

        with tf.Session() as sess:

            merged_summary = tf.summary.merge_all()
            suffix = "max_pooling"
            prefix = "final/F_1_with_512"

            saver = tf.train.Saver()

            train_writer = tf.summary.FileWriter(get_folder_name(test_fold, suffix, prefix), sess.graph)
            test_writer = tf.summary.FileWriter(get_folder_name(test_fold, suffix, prefix, True), sess.graph)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(num_epochs):
                minibatch_cost = 0.
                batch_accuracy_average = 0
                print("Epoch: %d" % epoch)

                minibatches = utils.random_mini_batches(train_features, train_labels, mini_batch_size)

                num_minibatches = len(minibatches)

                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch

                    [train_summary, num_steps, cost_val, _, train_accuracy] = sess.run(
                        [merged_summary, model.global_step, model.cost, train_op, accuracy],
                        feed_dict={model.features: minibatch_X, model.labels: minibatch_Y})

                    minibatch_cost += cost_val / num_minibatches

                    print('Step %d: cost %g minibatch_cost: %g' % (num_steps, cost_val, minibatch_cost))

                    batch_accuracy_average += train_accuracy / num_minibatches

                    test_accuracy, test_summary = sess.run([accuracy, merged_summary],
                                                           feed_dict={model.features: test_features,
                                                                      model.labels: test_labels})
                    model.update_accuracy(test_accuracy)

                    # WRITING SUMMARIES

                    max_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag="max_accuracy",
                                                                              simple_value=model.max_accuracy)])
                    if make_logs:
                        train_writer.add_summary(train_summary, num_steps)
                        train_writer.flush()
                        test_writer.add_summary(test_summary, num_steps)
                        test_writer.add_summary(max_accuracy_summary, num_steps)
                        test_writer.flush()

                print("batch cost: %g" % minibatch_cost)

                print("batch accuracy: %g" % batch_accuracy_average)

                print("test_acc: %g" % test_accuracy)

            if save_checkpoint:
                saver.save(sess, params.CHECKPOINT_FILE)

        print("Training of fold " + str(test_fold) + " has finished! TEST ACCURACY: %g" % model.max_accuracy)
        return model


def load_dataset(fold, train):
    path = "dataset/extracted_features/"
    features = np.load(path + ("train" if train else "test") + "_features_fold_" + str(fold) + ".npy")
    labels = np.load(path + ("train" if train else "test") + "_labels_fold_" + str(fold) + ".npy")
    return features, labels


def train_fold(fold):
    train_features, train_labels = load_dataset(fold, True)
    test_features, test_labels = load_dataset(fold, False)

    train_labels = utils.convert_array_to_one_hot(train_labels, 50)
    test_labels = utils.convert_array_to_one_hot(test_labels, 50)

    train_features = transpose_features(train_features)
    test_features = transpose_features(test_features)

    print("TRAINING FOLD " + str(fold))
    return train(train_features, train_labels, test_features, test_labels, test_fold=fold)


def train_all_folds():
    fold_accuracies = {}
    average_acc = 0
    for fold in range(1, 6):
        model = train_fold(fold)
        fold_accuracies[fold] = model.max_accuracy
        average_acc = avg(fold_accuracies.values())
        print("Average accuracy over folds: " + str(average_acc))
    return average_acc


if __name__ == '__main__':
    avg = train_fold(params.TEST_FOLD)
