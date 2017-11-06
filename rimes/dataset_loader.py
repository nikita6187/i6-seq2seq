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
    #print 'Max length:' + str(max_length)
    return max_length


class BatchManager:
    def __init__(self, inputs, targets):
        np.random.seed(1)
        self.inputs = inputs
        self.targets = targets
        self._current_pos = 0
        self.lookup = []
        self.init_integer_encoding()
        self.new_epoch()


    def init_integer_encoding(self):
        # Go over all outputs and create lookup dictionary
        for target in range(len(self.targets)):
            for letter in range(len(self.targets[target])):
                if self.targets[target][letter].lower() not in self.lookup:
                    self.lookup.append(self.targets[target][letter].lower())

    def lookup_letter(self, letter):
        return self.lookup.index(letter.lower())

    def get_size_vocab(self):
        return len(self.lookup)

    def new_epoch(self):
        self._current_pos = 0
        self.inputs, self.targets = shuffle_in_unison(self.inputs, self.targets)

    def get_letter_from_index(self, index):
        if 0 <= index < len(self.lookup):
            return self.lookup[index]
        else:
            return ''

    def next_batch(self, batch_size, pad=True, pad_outout_extra=3):
        """
        Returns the next batch of inputs and outputs. Padding with 0 of correct dims in both input and output
        so that batch size is the same.
        Input and output are of shape [batch_size, max_batch<in/out>put_seq_length, dims]
        :param batch_size:
        :param pad:
        :return:
        """
        inputs_batch = np.copy(self.inputs[self._current_pos:self._current_pos + batch_size])
        targets_batch = np.copy(self.targets[self._current_pos:self._current_pos + batch_size])

        # @TODO: bucketing of sequences

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
            max_length_t = get_max_seq_length(targets_batch) + pad_outout_extra
            #zero_t = np.zeros_like(targets_batch[0][0], dtype=np.int32)
            zero_t = np.full_like(targets_batch[0][0], -1, dtype=np.int32)
            for i in range(0, len(targets_batch)):
                for j in range(0, max_length_t):
                    if j >= len(targets_batch[i]):
                        targets_batch[i] = np.append(targets_batch[i], zero_t)

            # Hack to get dims right
            inputs_batch = np.asarray(inputs_batch.tolist())
            targets_batch = np.asarray(targets_batch.tolist())

        self._current_pos += batch_size

        targets_batch_final = np.zeros_like(targets_batch, dtype=np.int32)

        # Convert targets to integers
        for item in range(len(targets_batch)):
            for letter in range(len(targets_batch[item])):
                targets_batch_final[item][letter] = self.lookup_letter(targets_batch[item][letter])

        # Auto run new epoch if needed
        if self._current_pos >= len(self.inputs) - batch_size:
            self.new_epoch()

        return inputs_batch, targets_batch_final



# Example usage
"""
i, t = load_from_file('train.0010')
bm = BatchManager(i, t)
bm.lookup.append('-1')
print bm.lookup
for i in range(5):
    ib, tb = bm.next_batch(5)
    print tb
"""

