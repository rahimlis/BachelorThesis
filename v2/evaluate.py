import numpy as np
import tensorflow as tf
import torch
import operator
from v2 import params
from v2.train_network import get_model, transpose_features
from v2 import feat_extractor as extractor
from v2 import utils

def make_prediction(wav_file):
    graph = tf.Graph()
    class_map, _, _ = utils.read_csv()

    extracted_features = extractor.extract_from_file(wav_file)

    extracted_features = transpose_features(extracted_features)

    with graph.as_default(), tf.Session() as sess:
        model = get_model()

        model_var_names = [v.name for v in tf.global_variables()]

        model_vars = [v for v in tf.global_variables() if v.name in model_var_names]

        # Use a Saver to restore just the variables selected above.
        saver = tf.train.Saver(model_vars, name='model_load_pretrained',
                               write_version=1)

        checkpoint_path = params.CHECKPOINT_FILE

        saver.restore(sess, checkpoint_path)

        prediction, top_5 = sess.run([model.prediction, model.top_5], feed_dict={model.features: extracted_features})

        print("\n".join(format_top5(class_map, top_5)))

        return prediction, top_5


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
    prediction, top5 = make_prediction("./files/1-977-A-39.wav")
