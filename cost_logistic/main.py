import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math



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
                    [17.64, 6.22],
                    [16.91, 5.83],
                    [7.58, 12.54],
                    [7.53, 13.06],
                    [16.61, 4.52],
                    [17.58, 5.24],
                    [17.76, 5.91],
                    [17.52, 6.26]])

    clust = np.block([[clust], [sc]])

    points = [{'id': i, 'xy': cl} for (cl,i) in zip(clust[1:,:], range(len(clust)-1))]
    # add storage as last
    points.append({'id': len(points), 'xy': clust[1,:]})

    return points

def matr_adj(points):
    matr = [[np.linalg.norm(a['xy'] - b['xy']) for b in points] for a in points]
    return matr

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
    print(cluster_tree.cost)
    if cluster_tree.subroutes != None:
        for cl in cluster_tree.subroutes:
            show_cluster_tree(cl, points)
    else:
        matches = next(x for x in points if x['id'] == cluster_tree.points[0])
        matches['color'] = cluster_tree.cost


def cost_balancing(cluster_tree, total_cost, maximum):
    cluster_tree.weighted_cost = total_cost

    if cluster_tree.subroutes != None:
        total_self_cost = np.sum([cl.cost for cl in cluster_tree.subroutes])
        for cl in cluster_tree.subroutes:
            cost_balancing(cl, total_cost*cl.cost/total_self_cost, maximum)
    else:
        if total_cost > maximum[0]:
            maximum[0] = total_cost


def check_sum(cluster_tree):
    buf = 0
    if cluster_tree.subroutes != None:
        for cl in cluster_tree.subroutes:
            buf = buf + check_sum(cl)
        return  buf
    else:
        return cluster_tree.cost



class Cluster_Tree(object):
    def __init__(self, adj, route, total_route, cost_of_point):
        self._points = route

        # ind = list(total_route).index(route[0])
        #
        # if ind == 0:
        #     previous_point = total_route[0]
        # else:
        #     previous_point = total_route[ind - 1]
        #
        #
        # if len(route) == 1:
        #     self._cost = parent_cost/parent_n_of_points
        # else:
        #     self._cost = np.sum(self.costs_by_route(adj, route))  +  adj[previous_point][route[0]]

        self._cost = cost_of_point

        self._weighted_cost = 0



        if len(route) == 1:
            self._subroutes = None
        else:
            self._subroutes = self.list_of_clusters(adj, route, total_route)

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
    def cost(self):
        return self._cost

    @property
    def subroutes(self):
        return self._subroutes

    def costs_by_route(self, adj, route):
        costs = [0]
        if len(route) == 1:
            return costs

        for i in range(len(route) - 1):
            costs.append(adj[route[i]][route[i + 1]])

        return costs

    def list_of_clusters(self, adj, route, total_route, min_samples=2):

        total_costs = np.cumsum(self.costs_by_route(adj, route))

        eps = total_costs[-1]

        total_costs_2D = np.array([[tc, 0] for tc in total_costs])

        check = np.array([0, 0])
        clustering = []

        while all(check == 0):
            clustering = DBSCAN(eps, min_samples).fit(total_costs_2D)
            #eps = max(eps - 0.01, 0.00001)
            eps = 0.9*eps
            check = np.array(clustering.labels_)


        clusters = []

        cl_index = []

        subroutes = []

        checksum = np.sum(np.abs(np.array(list(set(clustering.labels_)))))

        for l in clustering.labels_:
            if l == -1:
                cl_index.append(len(cl_index) + checksum)
            else:
                cl_index.append(l)

        for ind in set(cl_index):
            subroute = route[cl_index == ind]
            subroutes.append(subroute)

        subroutes.sort(key = lambda el: el[0])

        internal_transfers = 0

        for i in range(len(subroutes) - 1):
            internal_transfers += adj[subroutes[i][-1]][subroutes[i+1][0]]

        cl_ratios = np.array([1/(len(sbr)*len(subroutes)) for sbr in subroutes])
        cl_ratios_normalized = cl_ratios/np.sum(cl_ratios)

        # print(cl_ratios_normalized)
        # print(np.sum(cl_ratios_normalized))
        # print(internal_transfers)
        # print(subroutes)

        for (subroute, r) in zip(subroutes, cl_ratios_normalized):

            delta = internal_transfers*r/len(subroute)

            clusters.append(Cluster_Tree(adj, subroute, total_route, self._cost + delta))


        return clusters






