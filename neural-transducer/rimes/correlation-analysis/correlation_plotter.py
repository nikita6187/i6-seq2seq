import matplotlib.pyplot as plt
import sys

# USAGE:
# Pass the as first parameter the path to the tuple list file
# Pass as the second argument the amount from behind to show
p = sys.argv[1]
x_val = []
y_val = []
with open(p, "rb") as fp:

    all_lines = fp.readlines()[0]

    all_lines = all_lines.split('(')

    all_lines = [line.split(')') for line in all_lines]

    for i in all_lines:
        if i is not ['']:
            tmp = i[0].split(",")
            try:
                x_val.append(float(tmp[0]))
                y_val.append(float(tmp[1]))
            except:
                pass
    print 'Total data: ' + str(len(x_val))

amount_behind = int(sys.argv[2])
plt.plot(x_val[-amount_behind:-1], y_val[-amount_behind:-1], 'go')
plt.show()