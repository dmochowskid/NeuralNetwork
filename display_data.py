import sys
import numpy as np
from Helpers import *


def main(argv):
    print(len(argv))
    if len(argv) != 2:
        usage()
        return

    data = np.loadtxt(argv[1], delimiter=',', skiprows=1)
    graph = create_graph_for_data(np.asmatrix(data))
    plt.show()


def usage():
    print("Usage:")
    print("display_data.py path")
    print("path - path to the data in csv")


if __name__ == "__main__":
    main(sys.argv)
