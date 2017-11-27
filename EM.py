# -*- coding: utf-8 -*-
"""
Author: VincentGum
"""
import numpy as np

class EM():

    MAX_ITER = 0  # only run for less than 500 times
    DATA = np.array([1, 1])  # the training data
    LAST_CEN = []  # the clusters' centers for the last iteration
    CURR_CEN = []  # the clusters' centers for current iteration
    M = np.zeros((1, 1))  # a matrix having each row representing each object, and each col for each cluster
    CLUSTERS = np.array([1, 1])  # an array containing cluster labels for each object
    C_NUM = 0
    iteration = 0
    SSE = []

    def __init__(self, DATA, CURR_CEN, MAX_ITER=50):
        """

        :param DATA: this is the training data should be inputted,
               take the given data set for example, it is shaped 695 * 6(regardless to user_id)
               for each row is an single object, each col is an attribute
        :param CURR_CEN: is a list containing N np.array, representing N cluster center.
               initial clusters' centers should be given

        Given DATA and CURR_CEN, let's say that we have N objects and C clusters
        """
        self.DATA = DATA
        self.CURR_CEN = CURR_CEN
        self.CLUSTERS = np.zeros((DATA.shape[0], 1))
        self.C_NUM = len(CURR_CEN)
        self.MAX_ITER = MAX_ITER

    def build_matrix(self):
        """
        Initialize the M matrix, shaped N * C

        """
        self.M = np.zeros((self.DATA.shape[0], len(self.CURR_CEN)))

    def normalization(self, l):
        """

        :param l: a list
        :return: the sum of all the item in the list
        """
        to_one = 0
        for i in l:
            to_one += i
        return to_one

    def e_step(self):
        """
        Do the e_step, and update the M matrix
        :return: none
        """
        objs = self.DATA
        rows, cols = objs.shape

        # rows == 695 iterate all the object in the data set
        for row in range(rows):

            # get the object we are going to assign the cluster to
            obj = objs[row, :]
            dist_inv = []

            # set the flag to indicate whether the object just share the same position with the cluster
            # because it would lead to the distance between them like '0' and (1 / 0) is meanless
            have_one = 0

            # len(LAST_CEN) == 2; calculate the dist between object and each cluster center
            for c in range(len(self.CURR_CEN)):

                diff = self.CURR_CEN[c] - obj
                ecu_sq = 0
                for i in diff:
                    ecu_sq += i ** 2
                if ecu_sq != 0:
                    dist_inv.append(1 / ecu_sq)
                # when the point is located tight at the center, take care of it specially
                elif ecu_sq == 0:
                    dist_inv.append(-1)
                    have_one = 1
                    break

            # update the M matrix for each point
            if have_one == 0:
                to_one = self.normalization(dist_inv)

                for c in range(len(self.CURR_CEN)):
                    self.M[row, c] = (dist_inv[c] / to_one)
                self.CLUSTERS[row, 0] = dist_inv.index(max(dist_inv))

            elif have_one == 1:
                for c in range(len(self.CURR_CEN)):
                    self.M[row, c] = 0
                self.M[row, dist_inv.index(-1)] = 1
                self.CLUSTERS[row, 0] = dist_inv.index(-1)

    def m_step(self):
        """
        do the m_step to update the clusters' centers
        """
        m = self.M

        # iterate all the clusters' centers
        for c in range(self.C_NUM):

            item = []
            column = list(m[:,c])

            for i in column:
                item.append(i ** 2)
            item = np.array(item / (self.normalization(item)))
            cur_cen = np.dot(item, self.DATA)

            self.CURR_CEN[c] = cur_cen

    def cal_SSE(self):

        objs = self.DATA
        rows, cols = objs.shape
        c_sse = []
        for c in range(len(self.CURR_CEN)):
            temp = []
            for row in range(rows):
                temp.append(np.sum((objs[row, ] - self.CURR_CEN[c]) ** 2))
            temp = np.array(temp)
            c_sse.append(np.dot(temp, (self.M[:, c])))
        self.SSE.append(np.sum(np.array(c_sse)))

    def train(self):
        """
        train the EM model and stop the training if(epsilon >= 0.001 or cur_iter < 3)
        """
        epsilon = 1
        cur_iter = 0
        self.build_matrix()

        while (epsilon > 0.001) and (cur_iter < self.MAX_ITER):

            self.LAST_CEN = []

            for i in self.CURR_CEN:
                self.LAST_CEN.append(i)

            # the main step to train
            self.e_step()
            self.m_step()
            self.cal_SSE()


            last_cur = self.LAST_CEN

            cur_iter += 1
            self.iteration = cur_iter
            # calculate the distance between clusters' center before and after
            l1 = []
            for i in range(len(self.CURR_CEN)):
                a = abs(last_cur[i] - self.CURR_CEN[i])
                # print(last_cur[i])
                # print(self.CURR_CEN[i])
                l1.append(max(list(a)))
            epsilon = self.normalization(l1)

            # print out the training information
            print('step %d: epsilon %f' % (cur_iter, epsilon))
