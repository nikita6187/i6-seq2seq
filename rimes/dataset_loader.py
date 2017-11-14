import h5py
import numpy as np
import random
import collections


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


class BatchManager:
    def __init__(self, inputs, targets, buckets):
        #NOTE: Error happens if bucket_size < batch_size
        np.random.seed(1)
        self.inputs, self.targets = sort_based_on_b(inputs, targets)
        self.inputs_buckets = []
        self.targets_buckets = []
        self._current_pos = [0] * (len(buckets)+1)
        self.lookup = []
        self.init_integer_encoding()
        self.new_epoch()
        self.buckets = buckets
        self.slice_data_into_buckets()

    def slice_data_into_buckets(self):
        # Slices the input/target arrays into the assigned buckets
        i = 0
        for bucket in self.buckets:
            current_arr_in = []
            current_arr_tg = []
            while len(self.targets[i]) <= bucket:
                current_arr_in.append(self.inputs[i])
                current_arr_tg.append(self.targets[i])
                i += 1
            #print 'Bucket ' + str(bucket) + ' is of size ' + str(len(current_arr_in))
            self.inputs_buckets.append(np.asarray(current_arr_in))
            self.targets_buckets.append(np.asarray(current_arr_tg))

        # Add the leftover sequences
        last_bucket_in = []
        last_bucket_tg = []
        while i < len(self.targets):
            last_bucket_in.append(self.inputs[i])
            last_bucket_tg.append(self.targets[i])
            i += 1
        self.inputs_buckets.append(np.asarray(last_bucket_in))
        self.targets_buckets.append(np.asarray(last_bucket_tg))


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

        for i in range(len(self.inputs_buckets)):
            min_t = 30
            max_t = 0
            #@todo finish testting
            #for t in range(len(self.inputs_buckets))

        for i in range(len(self.inputs_buckets)):
            self._current_pos[i] = 0
            self.inputs_buckets[i], self.targets_buckets[i] = shuffle_in_unison(self.inputs_buckets[i], self.targets_buckets[i])

    def get_letter_from_index(self, index):
        if 0 <= index < len(self.lookup):
            return self.lookup[index]
        else:
            return ''

    def next_batch(self, batch_size, pad=True, pad_outout_extra=1):
        """
        Returns the next batch of inputs and outputs. Padding with 0 of correct dims in both input and output
        so that batch size is the same.
        Input and output are of shape [batch_size, max_batch<in/out>put_seq_length, dims]
        :param batch_size:
        :param pad:
        :return:
        """
        current_bucket_index = random.randrange(0, len(self.buckets)+1)
        current_in = np.copy(self.inputs_buckets[current_bucket_index])
        current_target = np.copy(self.targets_buckets[current_bucket_index])
        inputs_batch = np.copy(current_in[self._current_pos[current_bucket_index]:self._current_pos[current_bucket_index] + batch_size])
        targets_batch = np.copy(current_target[self._current_pos[current_bucket_index]:self._current_pos[current_bucket_index] + batch_size])

        # Prevent returning objects smaller than batch size, by hoping the next random number doesn't hit small bucket
        # NOTE: Could lead to fatal error if unlucky
        if targets_batch.shape[0] < batch_size:
            return self.next_batch(batch_size, pad=pad, pad_outout_extra=pad_outout_extra)

        # Currently error with shuffling

        #print len(self.buckets)
        #print current_bucket_index
        #print targets_batch

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
            print 'Max length targets ' +  str(max_length_t) + ' with random ' + str(current_bucket_index)
            #zero_t = np.zeros_like(targets_batch[0][0], dtype=np.int32)
            zero_t = np.full_like(targets_batch[0][0], -1, dtype=np.int32)
            #print zero_t
            for i in range(0, len(targets_batch)):
                for j in range(0, max_length_t):
                    if j >= len(targets_batch[i]):
                        targets_batch[i] = np.append(targets_batch[i], [zero_t], axis=0)

            # Hack to get dims right
            inputs_batch = np.asarray(inputs_batch.tolist())
            targets_batch = np.asarray(targets_batch.tolist())

        #print self._current_pos[0]
        self._current_pos[current_bucket_index] += batch_size

        targets_batch_final = np.zeros_like(targets_batch, dtype=np.int32)

        # Convert targets to integers
        for item in range(len(targets_batch)):
            for letter in range(len(targets_batch[item])):
                targets_batch_final[item][letter] = self.lookup_letter(targets_batch[item][letter])

        # Auto run new epoch if needed
        if self._current_pos[current_bucket_index] >= len(self.inputs_buckets[current_bucket_index]) - batch_size:
            self.new_epoch()

        return inputs_batch, targets_batch_final

"""
a = np.asarray([[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 1, 3], [1, 1, 1, 1, 4], [1, 1, 1, 1, 5], [1, 1, 1, 1, 6]])
b = np.asarray([['a'], ['b', 'b'], ['c', 'c', 'c'], ['d','d','d','d'], ['e', 'e', 'e', 'e', 'e'], ['f', 'f', 'f', 'f', 'f', 'f']])
bm = BatchManager(a, b, buckets=[2, 4])
bm.lookup.append('-1')
print bm.lookup
for i in range(5):
    ib, tb = bm.next_batch(5)
    print ib
    print tb
"""
# Example usage

#i, t = load_from_file('train.0010')
#bm = BatchManager(i, t, buckets=[5, 10, 15])
#bm.lookup.append('-1')
"""
#a = ['s', 'a', 'l', 'u', 't', 'a', 't', 'i', 'o', 'n', 's']
k = 0
for x in range(len(t)):
    if len(t[x]) == len(a):
        s = 0
        for x2 in range(len(t[x])):
            if t[x][x2] == a[x2]:
                s += 1
        if s == len(a):
            k += 1
            print t[x]
            print x
print k
"""
#print bm.lookup
#for i in range(5000):
#    ib, tb = bm.next_batch(32)
#    print tb

