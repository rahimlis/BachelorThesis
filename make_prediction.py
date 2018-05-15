import operator

import tensorflow as tf
import numpy as np
import argparse
import utils
from train_network import model
import params
from vggish import vggish_params
from vggish import vggish_input

parser = argparse.ArgumentParser(description='Read file, process audio and give predictions')
parser.add_argument('--wav_file', type=str, help='File to read and process')
parser.add_argument('--checkpoint', type=str, help='Trained model checkpoint')
parser.add_argument('--test_fold', type=int, help='File to read and process')


def make_prediction(wav_file, checkpoint):
    graph, prediction, softmax_prediction = model(False)

    class_map, _, _ = utils.read_csv()

    with graph.as_default(), tf.Session(graph=graph) as sess:
        model_var_names = [v.name for v in tf.global_variables()]

        model_vars = [v for v in tf.global_variables() if v.name in model_var_names]

        # Use a Saver to restore just the variables selected above.
        saver = tf.train.Saver(model_vars, name='model_load_pretrained',
                               write_version=1)

        checkpoint_path = params.CHECKPOINT_FOLDER + checkpoint

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


if __name__ == '__main__':
    args = parser.parse_args()
    make_prediction(**vars(args))

# argmax, top5 = make_prediction("esc50/audio/2-119161-C-8.wav", "/checkpoint.ckpt-12500", 1)
