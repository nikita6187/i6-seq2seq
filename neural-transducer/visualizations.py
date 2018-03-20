import matplotlib.pyplot as plt
import sys

# USAGE:
# Param 1: Path to source file


def main():

    data = []

    for i in range(len(sys.argv) - 1):
        data1 = []
        with open(sys.argv[i + 1]) as f:
            for line in f:
                data1.append(float(line.split(' ')[1].split('\r')[0]))
        data.append(data1)

    colours = ['r', 'b', 'g']

    for i in range(len(data)):
        plt.plot(data[i], colours[i])
    plt.show()


if __name__ == '__main__':
    main()
