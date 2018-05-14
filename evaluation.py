import operator

import tensorflow as tf
import numpy as np

import utils
from train_network import model
import params
from vggish import vggish_params
from vggish import vggish_input


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
        train_accuracy = accuracy.eval({features_tensor: train_data, labels_tensor: train_labels})

        print("Test Accuracy:", test_accuracy)
        print("Train Accuracy:", train_accuracy)


def make_prediction(wav_file, checkpoint, test_fold):
    graph, prediction, softmax_prediction = model(False)

    class_map, _, _ = utils.read_csv()

    with graph.as_default(), tf.Session(graph=graph) as sess:
        model_var_names = [v.name for v in tf.global_variables()]

        model_vars = [v for v in tf.global_variables() if v.name in model_var_names]

        # Use a Saver to restore just the variables selected above.
        saver = tf.train.Saver(model_vars, name='model_load_pretrained',
                               write_version=1)

        checkpoint_path = params.CHECKPOINT_FOLDER + str(test_fold) + checkpoint

        saver.restore(sess, checkpoint_path)

        input_data = vggish_input.wavfile_to_examples(wav_file)

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

        argmax_predict = sess.run(prediction, feed_dict={features_tensor: input_data})
        top_5_predict = sess.run(softmax_prediction, feed_dict={features_tensor: input_data})

        print(argmax_predict)

        print("\n".join(format_top5(class_map, top_5_predict)))

        return argmax_predict, top_5_predict


def format_top5(class_map, top5):
    result_dicts = []
    for probabilities, predictions in zip(top5[0], top5[1]):
        dict = {}
        for cl, p in zip(predictions, probabilities):
            dict[class_map[cl]] = round(p, 6)

        sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

        result_dicts.append(str(sorted_dict))

    return result_dicts


# calculate_accuracy("/checkpoint.ckpt-12500", 1)

argmax, top5 = make_prediction("esc50/audio/2-119161-C-8.wav", "/checkpoint.ckpt-12500", 1)
