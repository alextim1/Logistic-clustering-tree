import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math



def scattering(center, n, radius):

    listArray = [(radius*r*np.cos(theta) + center[0], radius*r*np.sin(theta) + center[1]) for (r,theta) in zip(np.random.default_rng().normal(0, 0.1, n), np.arange(0, np.pi, np.pi/n))]

    return np.array(listArray)

def pointsField(scaleDistance, scaleRadius, nClusters, pointsRate):

    centers = np.block([[np.random.random(nClusters)], [np.random.random(nClusters)]])*scaleDistance
    radiuses = np.random.random(nClusters)*scaleRadius
    #nPoints = np.random.randint(1,pointsRate,nClusters)
    nPoints = [6,6,6,6]

    clust = np.array([0,0])

    for (c,r,n) in zip(centers.T, radiuses, nPoints):
        sc = scattering(c, n ,r)
        clust = np.block([[clust], [sc]])

    points = [{'id': i, 'xy': cl} for (cl,i) in zip(clust[1:,:], range(len(clust)-1))]

    return points

def matr_adj(points):
    matr = [[np.linalg.norm(a['xy'] - b['xy']) for b in points] for a in points]
    return matr

def route(points):
    # rt = np.arange(len(points))
    #
    # np.random.shuffle(rt)

    sh1 = [4,5,6,7]
    sh2 = [10, 11, 12, 13]
    sh3 = [16, 17, 18, 19]

    np.random.shuffle(sh1)
    np.random.shuffle(sh2)
    np.random.shuffle(sh3)

    rt = [0,1,2,3] + sh1 + [8, 9] + sh2 + [14, 15] + sh3 + [20, 21, 22, 23]

    print(rt)

    rt = np.array(rt)
    return rt


def show_cluster_tree(cluster_tree, points, total_cost):
    print(cluster_tree.points)
    print(cluster_tree.weighted_cost)
    if cluster_tree.subroutes != None:
        for cl in cluster_tree.subroutes:
            show_cluster_tree(cl, points, total_cost)
    else:
        matches = next(x for x in points if x['id'] == cluster_tree.points[0])
        matches['color'] = [cluster_tree.weighted_cost/total_cost, 0.1, 0]


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
        return cluster_tree.weighted_cost

#################################################################
#
# class Cluster(object):
#
#     #Ctor
#     def __init__(self, matrix, points_ind):
#         self._points = matrix
#         self._center_of_gravity = np.array([np.sum(matrix[:,0]), np.sum(matrix[:,1])])/matrix.shape[0]
#         self._claster_rate = (self.std_euclidian() + np.linalg.norm(self._center_of_gravity))/self._points.shape[0]
#         self._cluster_cost = 0
#         self._points_ind = points_ind
#
#     #Properties
#     @property
#     def center_of_gravity(self):
#         return self._center_of_gravity
#
#     @property
#     def points(self):
#         return self._end_points
#
#     @property
#     def cluster_rate(self):
#         return self._claster_rate
#
#     @property
#     def cluster_cost(self):
#         return self._cluster_cost
#
#     @cluster_cost.setter
#     def cluster_cost(self, var):
#         self._cluster_cost = var
#
#
#
#     #Private methods
#     def std_euclidian(self):
#         return np.sqrt((np.sum((self._points[:,0] - self._center_of_gravity[0])**2) + np.sum((self._points[:,1] - self._center_of_gravity[1])**2))/self._points.shape[0])
#
#
#
#
#     #Public methods
#     def individual_cost_calc(self):
#         norms = [np.linalg.norm(a - self._center_of_gravity) for a in self._points]
#         sigmaN = np.sum(norms)
#
#         if sigmaN == 0:
#             ind_cost = [(self._cluster_cost, self._cluster_cost)]
#         else:
#             ind_cost = [(self._cluster_cost/len(norms), norm*self._cluster_cost/sigmaN) for norm in norms]
#
#         for (end_point, cost) in zip(self._end_points, ind_cost):
#             end_point["cost"] = cost
#
#         return
#

class Cluster_Tree(object):
    def __init__(self, adj, route, start_point):
        self._points = route
        self._cost = np.sum(self.costs_by_route(adj, route)) + adj[start_point][route[0]]
        self._weighted_cost = 0


        if len(route) == 1:
            self._subroutes = None
        else:
            self._subroutes = self.list_of_clusters(adj, route, start_point)

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

    def list_of_clusters(self, adj, route, start_point, eps=3, min_samples=3):

        total_costs = np.cumsum(self.costs_by_route(adj, route))

        total_costs_2D = np.array([[tc, 0] for tc in total_costs])

        check = np.array([0, 0])
        clustering = []

        while all(check == 0):
            clustering = DBSCAN(eps, min_samples).fit(total_costs_2D)
            eps = max(eps - 0.01, 0.00001)
            check = np.array(clustering.labels_)


        clusters = []

        cl_index = []

        checksum = np.sum(np.abs(np.array(list(set(clustering.labels_)))))

        for l in clustering.labels_:
            if l == -1:
                cl_index.append(len(cl_index) + checksum)
            else:
                cl_index.append(l)

        for ind in set(cl_index):
            subroute = route[cl_index == ind]
            clusters.append(Cluster_Tree(adj, subroute, start_point))


        return clusters




# if __name__ == '__main__':
#
#     matr = pointsField(10, 8, 6, 20)
#     clustering = DBSCAN(eps=1, min_samples=3).fit(matr)
#     colors = {col:[np.random.random(), np.random.random(),np.random.random()] for col in set(clustering.labels_)}
#     colors = [colors.get(k) for k in clustering.labels_]
#
#     area = np.pi * 10
#
#     print(np.block([matr, np.array(clustering.labels_).reshape((len(clustering.labels_), 1))]))
#
#     clr=list_of_clusters(matr)
#     print([cl.cluster_rate for cl in clr])
#     print([cl.points for cl in clr])
#
#     plt.scatter(matr[:,0], matr[:,1], s=area, c=colors, alpha=0.5)
#
#     plt.title('Scatter plot pythonspot.com')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()


