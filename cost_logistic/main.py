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
    print(cluster_tree.weighted_cost)
    if cluster_tree.subroutes != None:
        for cl in cluster_tree.subroutes:
            show_cluster_tree(cl, points)
    else:
        matches = next(x for x in points if x['id'] == cluster_tree.points[0])
        matches['color'] = [cluster_tree.weighted_cost, 0.1, 0]





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
    def __init__(self, adj, route, total_route, start_point, weighted_cost, getting_on_cach):
        self._points = route


        self._getting_on = self.getting_on(adj,route,total_route)



        self._weighted_cost = weighted_cost



        if len(route) == 1:
            self._subroutes = None
        else:
            self._subroutes = self.list_of_clusters(adj, route, total_route, start_point, self._weighted_cost, self._getting_on, getting_on_cach)

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

    def costs_by_route(self, adj, route):
        costs = [0]
        if len(route) == 1:
            return costs

        for i in range(len(route) - 1):
            costs.append(adj[route[i]][route[i + 1]])

        return costs

    def getting_on(self, adj, route, total_route):
        ind = list(total_route).index(route[0])

        if ind == 0:
            previous_point = total_route[0]
        else:
            previous_point = total_route[ind - 1]

        res = adj[previous_point][route[0]]

        return res

    def cost_calc(self, adj, route, total_route, parent_getting_on, getting_on_cach):
        ind = list(total_route).index(route[0])

        if ind == 0:
            previous_point = total_route[0]
        else:
            previous_point = total_route[ind - 1]

        key1 = str(previous_point)
        key2 = str(route[0])

        kombined_key = key1 + '_' + key2
        cached = getting_on_cach.get(kombined_key)

        if cached == None:
            getting_on = adj[previous_point][route[0]]
            getting_on_cach[kombined_key] = True
        else:
            getting_on = 0

        #getting_on = adj[previous_point][route[0]]

            ######## idea 1

            # if len(route) == 1:
            #     self._cost = parent_cost/parent_n_of_points
            # else:
            #     self._cost = np.sum(self.costs_by_route(adj, route))  +  adj[previous_point][route[0]]    #+ adj[start_point][route[0]]/storage_weight

            ######## idea 2 +
            # self._cost = max(len(route)*parent_cost/parent_n_of_points, np.sum(self.costs_by_route(adj, route))  +  adj[previous_point][route[0]])

            ######## idea 3 -
            # self._cost = np.sum(self.costs_by_route(adj, route)) + (len(route) * parent_cost / parent_n_of_points)*(adj[previous_point][route[0]])

            ######### idea 4 ++
            # self._cost = (len(route) * parent_cost / parent_n_of_points) + np.sum(self.costs_by_route(adj, route)) + adj[previous_point][route[0]]

            ######### idea 4a - prority of parent cost
            # w = 0.75
            # self._cost = (len(route) * parent_cost / parent_n_of_points) + w*(np.sum(self.costs_by_route(adj, route)) + adj[previous_point][route[0]])

            ######### idea 4aa - low weight of road to the cluster
            # w = 0.75
            # self._cost = (len(route) * parent_cost / parent_n_of_points) + np.sum(self.costs_by_route(adj, route)) + w*adj[previous_point][route[0]]

            ######### idea 5 +-
            # self._cost = np.mean([len(route) * parent_cost / parent_n_of_points, np.sum(self.costs_by_route(adj, route)) + adj[previous_point][route[0]]])

        ########### idea 6 caching cost for getting to the cluster
        w = 1
        cost = parent_getting_on + np.sum(self.costs_by_route(adj, route)) + w*getting_on

        return cost



    def list_of_clusters(self, adj, route, total_route, start_point, weighted_cost, parent_getting_on, getting_on_cach, min_samples=2):

        total_costs = np.cumsum(self.costs_by_route(adj, route))

        eps = total_costs[-1]

        total_costs_2D = np.array([[tc, 0] for tc in total_costs])

        check = np.array([0, 0])
        clustering = []

        while all(check == 0):
            clustering = DBSCAN(eps, min_samples).fit(total_costs_2D)

            eps = 0.9*eps
            check = np.array(clustering.labels_)


        clusters = []

        cl_index = []

        subroutes = []

        costs = []

        checksum = np.sum(np.abs(np.array(list(set(clustering.labels_)))))

        for l in clustering.labels_:
            if l == -1:
                cl_index.append(len(cl_index) + checksum)
            else:
                cl_index.append(l)

        for ind in set(cl_index):
            subroute = route[cl_index == ind]
            print(subroute)
            subroutes.append(subroute)
            cost = self.cost_calc(adj, subroute, total_route, parent_getting_on, getting_on_cach)
            print(cost)
            costs.append(cost)

        print(costs)
        total_cost = np.sum(costs)

        for (subroute, cost) in zip(subroutes, costs):
            clusters.append(Cluster_Tree(adj, subroute, total_route, start_point, weighted_cost*cost/total_cost, getting_on_cach))


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


