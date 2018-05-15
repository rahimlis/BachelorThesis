import numpy as np
import tensorflow as tf
import argparse
import params
from train_network import model
from vggish import vggish_params

parser = argparse.ArgumentParser(description='Read model and calculate accuracy on test set')
parser.add_argument('--checkpoint', type=str, help='Trained model checkpoint')
parser.add_argument('--test_fold', type=int, help='Number of test fold')


def calculate_accuracy(checkpoint, test_fold):
    graph, prediction, softmax_prediction = model(False)

    with graph.as_default(), tf.Session(graph=graph) as sess:
        model_var_names = [v.name for v in tf.global_variables()]

        model_vars = [v for v in tf.global_variables() if v.name in model_var_names]

        # Use a Saver to restore just the variables selected above.
        saver = tf.train.Saver(model_vars, name='model_load_pretrained',
                               write_version=1)

        checkpoint_path = params.CHECKPOINT_FOLDER + str(test_fold) + checkpoint

        saver.restore(sess, checkpoint_path)

        test_data = np.load("dataset/test_data_fold_" + str(test_fold) + ".npy")
        train_data = np.load("dataset/train_data_fold_" + str(test_fold) + ".npy")
        test_labels = np.load("dataset/test_labels_fold_" + str(test_fold) + ".npy")
        train_labels = np.load("dataset/train_labels_fold_" + str(test_fold) + ".npy")

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')

        correct_prediction = tf.equal(prediction, tf.argmax(test_labels, 1))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        test_accuracy = accuracy.eval({features_tensor: test_data, labels_tensor: test_labels})
        #train_accuracy = accuracy.eval({features_tensor: train_data, labels_tensor: train_labels})

        print("Test Accuracy:", test_accuracy)
        print("Train Accuracy:", train_accuracy)


calculate_accuracy("/checkpoint.ckpt-32128", 1)

# if __name__ == '__main__':
#   args = parser.parse_args()
# calculate_accuracy(**vars(args))

# argmax, top5 = make_prediction("esc50/audio/2-119161-C-8.wav", "/checkpoint.ckpt-12500", 1)
