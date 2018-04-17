import matplotlib.pyplot as plt
import sys

# USAGE:
# Param 1: Over how much to average over
# Param 2..n: Path to source files


def main():

    data = []

    average_over = int(sys.argv[1])

    for i in range(2, len(sys.argv) - 1):
        data1 = []
        with open(sys.argv[i + 1]) as f:
            new_data = 0
            for i in range(average_over):
                for line in f:
                    new_data += float(line.split(' ')[1].split('\r')[0])
            new_data /= average_over
            data.append(new_data)

    colours = ['r', 'b', 'g']

    for i in range(len(data)):
        plt.plot(data[i], colours[i])
    plt.show()


if __name__ == '__main__':
    main()
