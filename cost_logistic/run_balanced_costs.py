
import main
import numpy as np
import matplotlib.pyplot as plt





if __name__ == '__main__':

    matr = main.pointsField(20, 8, 4, 100)


    m = main.matr_adj(matr)


    rt = main.route(matr)


    getting_on_cach = {}

    tree = main.Cluster_Tree(m, rt, rt, rt[0], 100, getting_on_cach)




    main.show_cluster_tree(tree, matr)

    print(main.check_sum(tree))
    print(rt[0])

    area = np.pi * 10


    maximum = 0
    plt.subplot(131)
    for p in matr:
        maximum = max(p['color'][0], maximum)

    for p in matr:
        if p['id'] == rt[0]:
            col = [[0,0,1]]
            area = 100
            al = 1
        else:
            col = [p['color']]
            col[0][0] = col[0][0]/maximum
            area = 10
            al = 0.7
        plt.scatter(p['xy'][0], p['xy'][1], s=area, c=col, alpha=al)

    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')


    xx = []
    yy = []

    for p in rt:
        xy = [ppt['xy'] for ppt in matr if ppt['id']==p ]
        xx.append(xy[0][0])
        yy.append(xy[0][1])

    plt.subplot(132)
    plt.plot(xx,yy)
    plt.show()