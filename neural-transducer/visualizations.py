import matplotlib.pyplot as plt
import sys

# USAGE:
# Param 1: Path to source file


def main():

    data = []

    with open(sys.argv[1]) as f:
        for line in f:
            data.append(float(line.split(' ')[1].split('\r')[0]))

    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    main()
