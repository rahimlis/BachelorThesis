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


def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, input size) (m, Hi, Wi)
    Y -- true "label" vector (containing ones in the index of class), of shape (number of examples, n_y)
    mini_batch_size - size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


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


def convert_dataset_to_numpy_array(test_set=False, print_progress=True):
    _, filenames, labels = read_csv(test_set=test_set)
    dataset_X = None
    dataset_Y = None
    i = 0
    total = len(filenames)
    for filename, label in zip(filenames, labels):

        if print_progress:
            print("Converting " + filename + " " + str(i) + "/" + str(total))
        i += 1

        examples_of_wav_file = np.array(vggish_input.wavfile_to_examples(filename))

        # Convert label to one-hot vector, and then repeat it
        # for each sub-example created from vggish_input.wavfile_to_examples
        label = np.repeat(convert_scalar_to_one_hot(label), examples_of_wav_file.shape[0], axis=0)

        if dataset_X is not None:
            dataset_X = np.append(dataset_X, examples_of_wav_file, axis=0)
            dataset_Y = np.append(dataset_Y, label, axis=0)
        else:
            dataset_X = np.array(examples_of_wav_file)
            dataset_Y = np.array(label)

    save_to_file(dataset_X, params.TEST_DATA_FILE if test_set else params.TRAINING_DATA_FILE)
    save_to_file(dataset_Y, params.TEST_LABELS_FILE if test_set else params.TRAINING_LABELS_FILE)


def save_to_file(np_arr, path):
    print("File saved on path: " + path)
    np.save(path, np_arr)


def convert_array_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def convert_scalar_to_one_hot(scalar, C=params.NUM_CLASSES):
    return convert_array_to_one_hot(np.array([scalar]), C)


