# -*- coding: utf-8 -*-
import collections
import numpy as np
import numpy.linalg as la
import random
import math
import time
import scipy.spatial.distance as ssd
from sklearn.cluster import KMeans

def loadTxtData(fileName, types=[int,int,float], sep=','):
    dataSets = []
    with open(fileName) as file:
        for lines in file:
            tmp = lines[:].strip().split(sep)
            line = []
            for i in range(len(types)):
                line.append(types[i](tmp[i]))
            dataSets.append(line)
    return dataSets

def writeListData(fileName, listData, type='w'):
    if len(listData) == 0:
        raise('listData is null')
    with open(fileName, type) as output_file:
        for i in listData:
            output_file.writelines(str(i)[1:-1] + '\n')
def euclideanDist(u, v):
    distance = ssd.euclidean(u, v)
    sim = 1.0 / (1 + distance)
    return sim
def loadRelaxData(fileName, n, types=float, sep=','):
    dataSets = []
    with open(fileName) as file:
        for lines in file:
            tmp = lines[:].strip().split(sep)
            line = []
            for i in range(n):
                line.append(types(tmp[i]))
            dataSets.append(line)
    return dataSets
def NormCalc(a):
    #计算二范数
    return la.norm(a, 2)
def Kxyfunc(x, y):
    if x != 0:
        result = x
    result = y
    return result

def pro_dcd(vector, k):
    remain = []
    for i in range(len(vector)):
        if i == k :
            continue
        remain.append(vector[i])
    remainvec = np.array(remain)
    return remainvec

def processed_traindata(traindata,codelen, usernum, itemnum):
    u_dict = {}
    i_dict = {}
    for i in range(usernum):
        u_dict[i] = []
    for j in range(itemnum):
        i_dict[j] = []
    for rating in traindata:
        userid = rating[0]-1
        itemid = rating[1]-1
        rate = rating[2]
        u_dict[userid].append([itemid, rate])
        i_dict[itemid].append([userid, rate])
    return u_dict, i_dict

def anchor_selector(B, D, usernum, itemnum, codelen, u_clusterlabel, u_centroids, i_clusterlabel, i_centroids, trainData):
    X=np.zeros((usernum, codelen))
    Y=np.zeros((itemnum, codelen))
    for i in trainData:
        userid = i[0]-1
        itemid = i[1]-1
        userlable = u_clusterlabel[userid]
        useranchor = u_centroids[userlable]
        X[userid] = useranchor
        itemlable = i_clusterlabel[itemid]
        itemanchor = i_centroids[itemlable]
        Y[itemid] = itemanchor
    return X.T, Y.T

def updateB(B, D, X, U, u_dict, codelen, alpha1, alpha2, Lambda):
    '''
    输出： matrix(codelen * num)
    '''
    for userid in u_dict.keys():
        ui = np.array(U[userid])
        changed = True
        while(changed):
            changed = False
            for k in range(codelen):
                for j in range(len(u_dict[userid])): 
                    itemid = u_dict[userid][j][0]
                    rate = u_dict[userid][j][1]
                    bi = B[:, userid]
                    useranchor = X[:, userid]
                    dj = D[:, itemid]
                    bi_remain = pro_dcd(bi, k)   
                    dj_remain = pro_dcd(dj, k)
                    x = np.sum(bi_remain)
                    pre_rating = np.dot(dj_remain.T, bi_remain) 
                    gap = rate/5.0 - 0.5 - pre_rating/(2*codelen)                     
                    alter_bi_k = np.dot(-1/codelen*gap, dj[k]) + 2 * alpha1 * (bi[k] - useranchor[k])+ 2 * alpha2 * (bi[k]-ui[k])+ 2 * Lambda * bi[k]
                bi_k = bi[k]
                temp = Kxyfunc(bi_k, alter_bi_k) 
                temp = np.sign(temp)
                if temp != bi_k:
                    changed = True                 
                bi[k] = temp

    return B


def updateD(B, D, Y, V, i_dict, codelen, beta1, beta2, Lambda):
    '''
    输出： matrix(codelen * num)
    '''    
    for itemid in i_dict.keys():
        vj = np.array(V[:,itemid])
        changed = True
        while(changed):
            changed = False
            for k in range(codelen):
                for i in range(len(i_dict[itemid])):
                    userid = i_dict[itemid][i][0]
                    rate = i_dict[itemid][i][1]
                    bi = B[:, userid]
                    dj = D[:, itemid]
                    itemanchor  = Y[:, itemid]
                    bi_remain = pro_dcd(bi, k)   
                    dj_remain = pro_dcd(dj, k)
                    y = np.sum(dj_remain)
                    pre_rating = np.dot(bi_remain.T, dj_remain)
                    gap = rate/5.0 - 0.5 - pre_rating/(2*codelen)
                    alter_dj_k = 2*beta1 * (dj[k] - itemanchor[k]) + 2*beta2 * (dj[k]-vj[k])+ 2*Lambda * dj[k] + np.dot(-1/codelen*gap, bi[k]) 
                dj_k = dj[k]
                temp = Kxyfunc(dj_k, alter_dj_k) 
                temp = np.sign(temp)
                if temp != dj_k:
                    changed = True                 
                dj[k] = temp
    return D

def calculateLoss(B, D, X, Y, U, V, traindata, codelen, alpha1, alpha2, beta1, beta2, Lambda):
    loss = 0.0
    for rating in traindata:
        userid = rating[0]-1
        itemid = rating[1]-1
        rate = rating[2]
        bi = B.T[userid]
        dj = D.T[itemid]
        ui = U[userid]
        vj = V.T[itemid]
        residue = rate/5.0 - 0.5 - (np.dot(bi.T, dj)/(2*codelen))
        
        loss += np.power(residue, 2)
    
    norm1 = np.dot(B.T, X)
    norm2 = np.dot(D.T, Y)
    norm3 = np.dot(B.T, U.T)
    norm4 = np.dot(D.T, V)
    loss -= alpha1*np.trace(norm1)
    loss -= beta1*np.trace(norm2)
    loss -= alpha2*np.trace(norm2)
    loss -= beta2*np.trace(norm2)
    loss += Lambda * (pow(NormCalc(np.sum(B.T, axis=0)), 2)+ pow(NormCalc(np.sum(D, axis=0)), 2))
    return loss

def iterator(B, D, X, Y, U, V, usernum,itemnum, traindata, codelen, iterNum, alpha1, alpha2, beta1, beta2, Lambda):
    start = time.time()
    old_error = 0
    new_error = 0
    converge = 0
    u_dict, i_dict = processed_traindata(traindata, codelen, usernum, itemnum)
    for step in range(iterNum): #迭代次数
        print('step: %d' % step)
        converge = new_error - old_error
        if step == 0:
            old_error = calculateLoss(B, D, X, Y, U, V, traindata, codelen, alpha1, alpha2, beta1, beta2, Lambda)
        else:
            old_error = new_error
        B = updateB(B, D, X, U, u_dict, codelen, alpha1, alpha2, Lambda)
        D = updateD(B, D, Y, V, i_dict, codelen, beta1, beta2, Lambda)
        loss = calculateLoss(B, D, X, Y, U, V, traindata, codelen, alpha1, alpha2, beta1, beta2, Lambda)
        new_error = calculateLoss(B, D, X, Y, U, V, traindata, codelen, alpha1, alpha2, beta1, beta2, Lambda)
        print(new_error)
        print('iterator time: %d' % (time.time() - start))
        print(abs(new_error - old_error))
        if abs(new_error - old_error) == converge:
            break
    return B, D 

usernum = 943
itemnum = 1682
codelen = 20
traindata = loadTxtData("train_ratings.csv")
U = np.array(loadRelaxData("mf100kuser2bit.txt", 2))
V = np.array(loadRelaxData("mf100kitem2bit.txt", 1682))
B = np.sign(U).T
D = np.sign(V)
u_estimator = KMeans(n_clusters=25)
u_estimator.fit(U)
u_clusterlabel = u_estimator.labels_ 
u_centroids = u_estimator.cluster_centers_ 
i_estimator = KMeans(n_clusters=50)
i_estimator.fit(V.T)
i_clusterlabel = i_estimator.labels_ 
i_centroids = i_estimator.cluster_centers_ 
X, Y = anchor_selector(B, D, usernum, itemnum, codelen, u_clusterlabel, u_centroids, i_clusterlabel, i_centroids, traindata)
finalB, finalD = iterator(B, D, X, Y, U, V, usernum, itemnum, traindata, codelen, 100, 3.5, 3.5, 3.5, 3.5, 3.5)
print('train model finished')
writeListData("b_usercode01.txt", (finalB.T).tolist())
writeListData("b_itemcode01.txt", finalD.tolist())
