import numpy as np
import random


def batch(x, max_sequence_length=10):
    """
	See https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/helpers.py
	:param x:
	:return:
	"""
    sequence_lengths = [len(i)+1 for i in x]
    max_length = max_sequence_length
    for i in x:
        while len(i) < max_length:
            i.append(0)
    sequence_array = np.asarray(x).T
    return sequence_array, sequence_lengths


def generate_random_lists(amount=10000, min_size=5, max_size=10, min_n=2, max_n=9):
    """
	Generates a specified amount of lists in a list, with each list being random in length and filled with random
	numbers.
    :type min_size: object
	:param amount:
	:param min_size:
	:param max_size:
	:param min_n:
	:param max_n:
	:return:
	"""
    rlist = []
    for i in range(amount):
        length = random.randrange(min_size, max_size)
        tlist = []
        for l in range(length):
            tlist.append(random.randrange(min_n, max_n))
        rlist.append(tlist)
    return rlist
