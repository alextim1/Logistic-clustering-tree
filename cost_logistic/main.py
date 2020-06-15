import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math

import warnings
warnings.filterwarnings("ignore")


def scattering(center, n, radius):

    listArray = [(radius*r*np.cos(theta) + center[0], radius*r*np.sin(theta) + center[1]) for (r,theta) in zip(np.random.default_rng().normal(0, 0.1, n), np.arange(0, np.pi, np.pi/n))]

    return np.array(listArray)

def pointsField(scaleDistance, scaleRadius, nClusters, pointsRate):

    # centers = np.block([[np.random.random(nClusters)], [np.random.random(nClusters)]])*scaleDistance
    # radiuses = np.random.random(nClusters)*scaleRadius
    # #nPoints = np.random.randint(1,pointsRate,nClusters)
    # nPoints = [6,6,6,6]
    #
    # clust = np.array([0,0])
    #
    # for (c,r,n) in zip(centers.T, radiuses, nPoints):
    #     sc = scattering(c, n ,r)
    #     clust = np.block([[clust], [sc]])

    clust = np.array([0,0])
    sc = np.array([[8.59, 10.32],
                    [8.35, 10.27],
                    [8.45, 10.33],
                    [8.44, 10.28],
                    [8.42, 10.36],
                    [8.53, 10.27],
                    [16.77, 15.26],
                    [16.42, 14.98],
                    [16.53, 14.59],
                    [16.90, 15.97],
                    [16.76, 15.36],
                    [7.92, 12.32],
                    [16.38, 16.17],
                    [7.37, 11.98],
                    [7.83, 12.10],
                    [7.96, 10.64],
                    [17.64, 2.22],
                    [16.91, 1.83],
                    [7.58, 12.54],
                    [7.53, 13.06],
                    [16.61, 4.52],
                    [17.58, 5.24],
                    [17.76, 5.91],
                    [17.52, 6.26]])

    # sc = np.array([[8.59, 10.32],
    #                [16.61, 4.52],
    #                [17.58, 5.24],
    #                [17.76, 5.91],
    #                [17.52, 6.26]])

    clust = np.block([[clust], [sc]])

    points = [{'id': i, 'xy': cl} for (cl,i) in zip(clust[1:,:], range(len(clust)-1))]
    # add storage as last
    points.append({'id': len(points), 'xy': clust[1,:]})

    return points

def matr_adj(points):
    matr = [[np.linalg.norm(a['xy'] - b['xy']) for b in points] for a in points]
    return np.array(matr)

def route(points):
    rt = np.arange(len(points))
    #
    # np.random.shuffle(rt)

    # sh1 = [4,5,6,7]
    # sh2 = [10, 11, 12, 13]
    # sh3 = [16, 17, 18, 19]
    #
    # np.random.shuffle(sh1)
    # np.random.shuffle(sh2)
    # np.random.shuffle(sh3)
    #
    # rt = [0,1,2,3] + sh1 + [8, 9] + sh2 + [14, 15] + sh3 + [20, 21, 22, 23]
    #
    # print(rt)
    #
    # rt = np.array(rt)


    return rt


def show_cluster_tree(cluster_tree, points):
    print(cluster_tree.points)
    print(cluster_tree.weighted_cost)
    if cluster_tree.subroutes != None:
        for cl in cluster_tree.subroutes:
            show_cluster_tree(cl, points)
    else:
        matches = next(x for x in points if x['id'] == cluster_tree.points[0])
        matches['color'] = cluster_tree.weighted_cost





def check_sum(cluster_tree):
    buf = 0
    if cluster_tree.subroutes != None:
        for cl in cluster_tree.subroutes:
            buf = buf + check_sum(cl)
        return  buf
    else:
        return cluster_tree.weighted_cost

#################################################################


class Cluster_Tree(object):


    def __init__(self, matr, adj, route, total_route, weighted_cost):
        self._points = route


        self._weighted_cost = weighted_cost



        if len(route) == 1:
            self._subroutes = None
        else:
            self._subroutes = self.list_of_clusters(matr, adj, route, total_route, self._weighted_cost)




############PROPERTIES#########################
    @property
    def weighted_cost(self):
        return self._weighted_cost

    @weighted_cost.setter
    def weighted_cost(self, val):
        self._weighted_cost = val

    @property
    def points(self):
        return self._points


    @property
    def subroutes(self):
        return self._subroutes


#############METHODS#####################################

    def costs_by_route(self, adj, route):
        costs = [0]
        if len(route) == 1:
            return costs

        for i in range(len(route) - 1):
            costs.append(adj[route[i]][route[i + 1]])

        return costs



    def cost_calc(self, matr, adj, route, total_route, boundaries):
        ind = list(total_route).index(route[0])



        storage_weight = 0.01


        shoulder_weight_prev = 0.5
        shoulder_weight_post = 0.5

        if len(route) == 1 and ind == 0:
            #start point case
            point_cost = storage_weight*adj[route[0]][total_route[ind + 1]]

        elif len(route) == 1 and ind == len(total_route)-1:
            #end point case
            point_cost = storage_weight*adj[route[0]][total_route[ind - 1]]

        elif len(route) == 1:

            prev = shoulder_weight_prev*(not boundaries[0])*adj[route[0]][total_route[ind - 1]]
            post = shoulder_weight_post*(not boundaries[1])*adj[route[0]][total_route[ind + 1]]

            point_cost = prev+post

        else:
            point_cost = 0




        X = []
        Y = []
        for p in route:
            X.append(matr[p]['xy'][0])
            Y.append(matr[p]['xy'][1])

        cg = np.array([np.mean(X), np.mean(Y)])
        storage = np.array([matr[total_route[0]]['xy'][0], matr[total_route[0]]['xy'][1]])

        cg_distance = np.linalg.norm(storage - cg)


        cost = point_cost + np.sum(self.costs_by_route(adj, route)) + cg_distance

        return cost



    def list_of_clusters(self, matr, adj, route, total_route, weighted_cost, min_samples=2):

        total_costs = np.cumsum(self.costs_by_route(adj, route))

        eps = total_costs[-1]

        total_costs_2D = np.array([[tc, 0] for tc in total_costs])

        check = np.array([0, 0])
        clustering = []

        while all(check == 0):
            clustering = DBSCAN(eps, min_samples).fit(total_costs_2D)

            eps = 0.95*eps
            check = np.array(clustering.labels_)


        clusters = []

        cl_index = []

        subroutes = []

        lengths = []

        costs = []

        checksum = np.sum(np.abs(np.array(list(set(clustering.labels_)))))

        for l in clustering.labels_:
            if l == -1:
                cl_index.append(len(cl_index) + checksum)
            else:
                cl_index.append(l)

        for ind in set(cl_index):
            subroute = route[cl_index == ind]

            front = subroute[0]==route[0]
            tail = subroute[-1]==route[-1]



            subroutes.append(subroute)

            lengths.append(len(subroute))

            cost = self.cost_calc(matr, adj, subroute, total_route, (front, tail))

            costs.append(cost)


        #Rescaling cluster

        costs = np.array(costs)
        lengths = np.array(lengths)



        costs = costs*lengths

        total_cost = np.sum(costs)

        for (subroute, cost) in zip(subroutes, costs):
            clusters.append(Cluster_Tree(matr, adj, subroute, total_route, weighted_cost*cost/total_cost))


        return clusters







