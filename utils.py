import params
import csv
import numpy as np
import math
from vggish import vggish_input


def read_csv(filepath=params.CLASS_LABELS_FILE, test_set=False):
    """
    reads CSV meta document containing filenames, folders and corresponding categories
    :param filepath:
    :return: tuple containing class map, filenames and labels
    """
    with open(filepath) as f:
        next(f)  # skip header

        reader = csv.reader(f)
        _class_map = {}
        filenames = []
        labels = []

        for row in reader:

            # map category target to its description
            _class_map[int(row[2])] = row[3]

            # add only rows which are either in test or training folders
            if (not test_set and row[1] != params.TEST_FOLD) or (test_set and row[1] == params.TEST_FOLD):
                filenames.append(params.PATH_TO_FILES + row[0])  # appends filenames
                labels.append(int(row[2]))

    return _class_map, filenames, np.array(labels, dtype=int)


def make_random_batches(filenames, labels, mini_batch_size=params.BATCH_SIZE):
    m = len(filenames)  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    X = np.array(filenames)
    Y = np.array(labels)
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def load_data(filenames_batch, labels_batch):
    minibatch_X = None
    minibatch_Y = None

    for filename, label in zip(filenames_batch, labels_batch):
        examples_of_wav_file = np.array(vggish_input.wavfile_to_examples(filename))

        # Convert label to one-hot vector, and then repeat itfor each sub-example created from vggish_input.wavfile_to_examples
        label = np.repeat(convert_scalar_to_one_hot(label), examples_of_wav_file.shape[0], axis=0)

        if minibatch_X is not None:
            minibatch_X = np.append(minibatch_X, examples_of_wav_file, axis=0)
            minibatch_Y = np.append(minibatch_Y, label, axis=0)
        else:
            minibatch_X = np.array(examples_of_wav_file)
            minibatch_Y = np.array(label)

    return minibatch_X, minibatch_Y


def convert_array_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def convert_scalar_to_one_hot(scalar, C=params.NUM_CLASSES):
    return convert_array_to_one_hot(np.array([scalar]), C)
