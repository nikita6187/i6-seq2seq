import matplotlib.pyplot as plt
import sys

# USAGE:
# Param 1: Path to source file


def main():

    data1 = []

    with open(sys.argv[1]) as f:
        for line in f:
            data1.append(float(line.split(' ')[1].split('\r')[0]))

    data2 = []
    with open(sys.argv[2]) as f:
        for line in f:
            data2.append(float(line.split(' ')[1].split('\r')[0]))

    plt.plot(data2, 'r')
    plt.plot(data1, 'b')
    plt.show()


if __name__ == '__main__':
    main()
