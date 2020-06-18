
import main
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)





def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)






if __name__ == '__main__':

    matr = main.pointsField(20, 8, 4, 100)


    m = main.matr_adj(matr)


    rt = main.route(matr)




    tree = main.Cluster_Tree(m, rt, rt, 0)

    maximum = [0]
    main.cost_balancing(tree, 100, maximum)

    main.show_cluster_tree(tree, matr)

    print(main.check_sum(tree))
    print(rt[0])

    c1 = 'blue'
    c2 = 'yellow'

    xx = []
    yy = []

    for p in rt:
        xy = [ppt['xy'] for ppt in matr if ppt['id'] == p]
        xx.append(xy[0][0])
        yy.append(xy[0][1])

    ####################### Visualise

    fig = plt.figure()
    adjustFigAspect(fig, aspect=1)
    ax = fig.add_subplot(111)

    al = 0.3
    ax.plot(xx, yy, alpha=al)

    maximum = 0
    minimum = 100
    for p in matr:
        maximum = max(p['color'], maximum)
        minimum = min(p['color'], minimum)

    for p in matr:
        if p['id'] == rt[0]:
            col = [[0, 0, 1]]
            area = 100
            al = 0.3
        else:
            col = colorFader(c1, c2, (p['color'] - minimum) / (maximum - minimum))
            area = 10 * p['color']
            al = 0.9
        ax.scatter(p['xy'][0], p['xy'][1], s=area, c=col, alpha=al)

    ax.set_title('route')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.show()
    input()