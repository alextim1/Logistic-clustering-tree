
import main
import numpy as np
import matplotlib.pyplot as plt


class A(object):
    def __init__(self, a):
        self._a = a

    @property
    def a(self):
        return self._a


if __name__ == '__main__':

    matr = main.pointsField(10, 8, 6, 10)


    m = main.matr_adj(matr)


    rt = main.route(matr)


    tree = main.Cluster_Tree(m, rt, rt[0])

    maximum = [0]
    main.cost_balancing(tree, 100, maximum)

    main.show_cluster_tree(tree, matr, maximum[0])

    print(main.check_sum(tree))
    print(rt[0])

    area = np.pi * 10


    for p in matr:
        if p['id'] == rt[0]:
            col = [[0,0,0]]
        else:
            col = [p['color']]
        plt.scatter(p['xy'][0], p['xy'][1], s=area, c=col, alpha=0.9)

    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()