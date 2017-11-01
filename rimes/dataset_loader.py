import h5py
import numpy as np


def load_from_file(file_name):
    """
    Returns inputs [amount_of_inputs, max_seq_length, input_dim] and targets
    [amounts_of_targets, max_target_seq_length, target_dim] as numpy array.
    :param file_name:
    :return:
    """
    f = h5py.File(file_name, 'r')
    inputs_raw = f['inputs'].value
    seq_lengths = f['seqLengths'].value
    classes = f['targets']['labels']['classes'].value
    targets_raw = f['targets']['data']['classes'].value

    inputs_processed = []
    targets_processed = []

    inputs_pointer = 0  # Used to point to location in list where we are
    targets_pointer = 0

    for i in range(0, len(seq_lengths)):
        # First go over the inputs
        curr_input = []
        for input in range(inputs_pointer, inputs_pointer+seq_lengths[i][0]):
            curr_input.append(np.asarray(inputs_raw[input]))
        inputs_processed.append(np.asarray(curr_input))
        inputs_pointer += seq_lengths[i][0]

        # Then go over the targets
        curr_target = []
        for output in range(targets_pointer, targets_pointer+seq_lengths[i][1]):
            curr_target.append(np.asarray(classes[int(targets_raw[output])]))
        targets_pointer += seq_lengths[i][1]
        targets_processed.append(np.asarray(curr_target))

    f.close()

    return np.asarray(inputs_processed), np.asarray(targets_processed)


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def get_max_seq_length(arr):
    max_length = 0
    for i in range(0, len(arr)):
        val = len(arr[i])
        if val > max_length:
            max_length = val
    return max_length


class BatchManager:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self._current_pos = 0
        self.inputs, self.targets = shuffle_in_unison(self.inputs, self.targets)

    def new_epoch(self):
        self.inputs, self.targets = shuffle_in_unison(self.inputs, self.targets)

    def next_batch(self, batch_size, pad=True):
        """
        Returns the next batch of inputs and outputs. Padding with 0 of correct dims in both input and output
        so that batch size is the same.
        Input and output are of shape [batch_size, max_batch<in/out>put_seq_length, dims]
        :param batch_size:
        :param pad:
        :return:
        """
        inputs_batch = self.inputs[self._current_pos:self._current_pos + batch_size]
        targets_batch = self.targets[self._current_pos:self._current_pos + batch_size]

        # Pad if needed
        if pad is True:
            # First do input padding
            max_length_i = get_max_seq_length(inputs_batch)
            zero_i = np.zeros_like(inputs_batch[0][0])
            for i in range(0, len(inputs_batch)):
                for j in range(0, max_length_i):
                    if j >= len(inputs_batch[i]):
                        inputs_batch[i] = np.append(inputs_batch[i], [zero_i], axis=0)

            # Then do targets
            max_length_t = get_max_seq_length(targets_batch)
            zero_t = np.zeros_like(targets_batch[0][0])
            for i in range(0, len(targets_batch)):
                for j in range(0, max_length_t):
                    if j >= len(targets_batch[i]):
                        targets_batch[i] = np.append(targets_batch[i], zero_t)

            # Hack to get dims right
            inputs_batch = np.asarray(inputs_batch.tolist())
            targets_batch = np.asarray(targets_batch.tolist())

        self._current_pos += batch_size
        return inputs_batch, targets_batch

"""
Example usage
i, t = load_from_file('train.0010')
bm = BatchManager(i, t)
ib, tb = bm.next_batch(5)

"""

