import h5py
import numpy as np
import random
import collections


def handle_ascii(s):
    if all(ord(c) < 128 for c in s) is True:
        return s
    else:
        return ''


def load_from_file(file_name, max_length_input, max_length_target):
    """
    Returns inputs [amount_of_inputs, max_seq_length, input_dim] and targets
    [amounts_of_targets, max_target_seq_length, target_dim] as numpy array.
    :param file_name:
    :param max_length_input: Max length to use. Set 0 to auto determine max length.
    :return:
    """
    f = h5py.File(file_name, 'r')
    inputs_raw = f['inputs'].value
    seq_lengths = f['seqLengths'].value
    classes = f['targets']['labels']['classes'].value
    targets_raw = f['targets']['data']['classes'].value

    # We now use pre determined lengths
    if max_length_input == 0:
        max_length_input, max_length_target, _ = np.asarray(seq_lengths).max(axis=0)
    dims = inputs_raw.shape[1]

    inputs_array = np.zeros((len(seq_lengths), max_length_input, dims))
    inputs_lengths = np.asarray(seq_lengths)[:, 0]
    targets_array = np.zeros((len(seq_lengths), max_length_target), dtype=np.str_)
    targets_lengths = np.asarray(seq_lengths)[:, 1]

    inputs_pointer = 0  # Used to point to location in list where we are
    targets_pointer = 0

    for i in range(0, len(seq_lengths)):
        # First go over the inputs
        for input in range(inputs_pointer, inputs_pointer + seq_lengths[i][0]):
            curr_index = input - inputs_pointer
            for dim in range(0, dims):
                inputs_array[i][curr_index][dim] = inputs_raw[input][dim]
        inputs_pointer += seq_lengths[i][0]

        # Then go over targets
        for output in range(targets_pointer, targets_pointer + seq_lengths[i][1]):
            curr_index = output - targets_pointer
            targets_array[i][curr_index] = handle_ascii(classes[int(targets_raw[output])])
        targets_pointer += seq_lengths[i][1]

    f.close()

    return inputs_array, inputs_lengths, targets_array, targets_lengths


def shuffle_in_unison(a):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def shuffle_in_unison_multiple(a):
    """
    A is list of numpy arrays that will be shuffled in unison.
    :param a:
    :return:
    """
    shuffled_a = []
    for i in range(0,len(a)):
        shuffled_a.append(np.empty(a[i].shape, dtype=a[i].dtype))
    permutation = np.random.permutation(len(a[0]))
    for old_index, new_index in enumerate(permutation):
        for i in range(0, len(a)):
            shuffled_a[i][new_index] = a[i][old_index]
    return shuffled_a


def get_max_seq_length(arr):
    max_length = 0
    for i in range(0, len(arr)):
        val = len(arr[i])
        if val > max_length:
            max_length = val
    #print 'Max length:' + str(max_length)
    return max_length


def sort_based_on_b(a, b):
    # First make array of lengths of b (len_b)
    len_b = np.zeros_like(b.tolist())
    for i in range(len(b)):
        len_b[i] = len(b[i])
    # Then sort len_b and get permutation
    len_b = np.asarray(len_b.tolist())
    perm = len_b.argsort()
    # Apply permutation to a and b
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    for old_index, new_index in enumerate(perm):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[old_index] = b[new_index]
    return shuffled_a, shuffled_b

"""
def sparse_tuple_from(sequences, dtype=np.int32):
    """"""Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """"""
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape
"""

class BatchManager:
    def __init__(self, inputs, inputs_lengths, targets, targets_lengths, pad):
        #np.random.seed(1)
        self.inputs = inputs
        self.targets = targets
        self.inputs_lengths = inputs_lengths
        self.targets_lengths = targets_lengths
        self._current_pos = 0
        self.lookup = []
        self.init_integer_encoding()
        #self.new_epoch()
        self.pad = pad

    def init_integer_encoding(self):
        # Go over all outputs and create lookup dictionary
        for target in range(len(self.targets)):
            for letter in range(len(self.targets[target])):
                if self.targets[target][letter] not in self.lookup:
                    self.lookup.append(self.targets[target][letter])

    def lookup_letter(self, letter):
        if letter is self.pad:
            return self.get_size_vocab() + 1
        return self.lookup.index(letter)

    def get_size_vocab(self):
        return len(self.lookup)

    def get_letter_from_index(self, index):
        if 0 <= index < len(self.lookup):
            return self.lookup[index]
        else:
            return ''

    def offset(self, data, filler, amount=1, position=0, length_vector=None):
        """
        Offsets either at the start or the end of the numpy array the filler object.
        :param data:
        :param filler:
        :param amount:
        :param position:
        :return:
        """
        for i in range(0, amount):
            data = np.insert(data, position, filler, axis=1)
            if length_vector is not None:
                length_vector = np.add(length_vector, np.ones(length_vector.shape))
        return data, length_vector

    def new_epoch(self):
        self.inputs, self.inputs_lengths, self.targets, self.targets_lengths = shuffle_in_unison_multiple(
            [self.inputs, self.inputs_lengths, self.targets, self.targets_lengths]
        )
        self._current_pos = 0

    def next_batch(self, batch_size, convert_outputs_to_ints=True):
        """
        Returns the next batch of inputs and outputs. Padding with 0 of correct dims in both input and output
        so that batch size is the same.
        Input and output are of shape [batch_size, max_batch<in/out>put_seq_length, dims]
        :param batch_size:
        :param pad:
        :return:
        """
        inputs_batch = np.copy(self.inputs[self._current_pos:self._current_pos+batch_size])
        targets_batch_raw = np.copy(self.targets[self._current_pos:self._current_pos+batch_size])

        input_lengths = np.copy(self.inputs_lengths[self._current_pos:self._current_pos+batch_size])
        target_lengths = np.copy(self.targets_lengths[self._current_pos:self._current_pos + batch_size])

        if convert_outputs_to_ints is True:
            targets_batch = np.zeros_like(targets_batch_raw, dtype=np.int32)
            for target in range(0, targets_batch.shape[0]):
                for char in range(0, targets_batch[target].shape[0]):
                    if char <= target_lengths[target]:
                        targets_batch[target][char] = self.lookup_letter(targets_batch_raw[target][char])
                    else:
                        targets_batch[target][char] = self.lookup_letter(self.pad)
        else:
            targets_batch = targets_batch_raw

        targets_batch = np.reshape(targets_batch, (-1))
        targets_batch = targets_batch[0:target_lengths[0]+1]

        self._current_pos += batch_size

        if self._current_pos >= len(self.inputs):
            self.new_epoch()

        return inputs_batch, input_lengths, targets_batch, target_lengths


# Example usage
"""
i, i_l, t, t_l = load_from_file('train.0010')
bm = BatchManager(i, i_l, t, t_l, 'EOS', 'PAD')
print bm.lookup

for i in range(2):
    ib, il, tb, tl = bm.next_batch(1)
    print tb
    print tl

"""