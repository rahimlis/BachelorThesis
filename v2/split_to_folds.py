import numpy as np

import scipy as scp
import scipy.io

from v2 import utils

data = scp.io.loadmat("dataset/features.mat")

labels = data['labels'].reshape((2000, 1)).astype(int)
features = data['features'].reshape((2000, 1024, 5, 1))


def split_to_folds(np_arr, prefix):
    """
    Splits the full array of ESC-50 by extracting test fold and combining other folds into one array.
    Full array of ESC-50 contains 2000 examples organized into 5 folds. This function gets consecutively
    each fold and produces:

        test_[LABEL OR TRAINING DATA]_fold_[NUMBER OF TEST FOLD].npy
        train_[LABEL OR TRAINING DATA]_fold_[NUMBER OF TEST FOLD].npy (except test fold all others are combined here)

    :param np_arr: full NumPy array of ESC-50 dataset converted into mel spectrogram
    :param prefix: defines prefix for distinquishing labels and training data
    """
    total_examples = np_arr.shape[0]
    num_folds = 5
    test_length = int(total_examples / num_folds)

    for i in range(num_folds):
        test_up = i * test_length
        test_down = (i + 1) * test_length

        tr_up_1 = 0
        tr_down_1 = test_up
        tr_up_2 = test_down
        tr_down_2 = total_examples

        tr_fold = split_arr(arr=np_arr, up1=tr_up_1, down1=tr_down_1, up2=tr_up_2, down2=tr_down_2)
        test_fold = slice_arr(np_arr, test_up, test_down)

        print("\n\nSaving " + prefix + " fold " + str(i + 1))

        np.save("dataset/train_" + prefix + "_fold_" + str(i + 1) + ".npy", tr_fold)
        np.save("dataset/test_" + prefix + "_fold_" + str(i + 1) + ".npy", test_fold)

        print("Test: [" + str(test_up) + " : " + str(test_down)
              + "] Train: [" + str(tr_up_1) + " : " + str(tr_down_1) + "], ["
              + str(tr_up_2) + " : " + str(tr_down_2) + "]")


def split_arr(arr, up1, down1, up2, down2):
    s1 = slice_arr(arr, up1, down1)
    s2 = slice_arr(arr, up2, down2)

    if s1 is None and s2 is None:
        raise RuntimeError()

    if s1 is None:
        return s2

    if s2 is None:
        return s1

    return np.concatenate((s1, s2))


def slice_arr(arr, up, down):
    s1 = None
    if up != down:
        s1 = arr[up:down]
    return s1


split_to_folds(labels, "labels")
split_to_folds(features, "features")
