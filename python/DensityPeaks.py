import itertools
import numpy as np
from PairDotsDistance_oushi import PairDotsDistance_oushi

'''
计算数据密度峰值，发现数据指向结构关系
输入：数据xi， 邻居的百分比percent
输出：表面数据指向结构的数据nneigh
'''
def DensityPeaks(xi, precent):
    rowN, colN = xi.shape[0], xi.shape[1]  # 求输入数据矩阵的个数（行数rowN），属性维度（列数colN）
    xiT = xi.T
    xx = PairDotsDistance_oushi(xiT, colN, rowN)  # j计算密度峰值中distance matrix file
    xxT = xx
    ND = np.max(xxT[:, 1]) + 1  # 第二列的最大值
    NL = np.max(xxT[:, 0]) + 1  # 第一列的最大值

    if NL > ND:
        ND = NL
    N = xxT.shape[0]

    dist = np.zeros((90,90))

    for i in range(0, N):
        ii = xxT[i, 0]
        jj = xxT[i, 1]

        dist[int(ii), int(jj)] = xxT[i, 2]
        dist[int(jj), int(ii)] = xxT[i, 2]

    print('average percentage of neighbours (hard coded): %5.6f\n', precent)

    position = round(N * precent / 100, 0)

    sda = np.sort(xxT[:, 2])  # 对距离排序，升序

    dc = sda[int(position)]  # 文章中的截止距离dc

    print('Computing Rho with gaussian kernel of radius: %12.6f\n', dc)

    rho = np.zeros((1, int(ND))) # [i for i in range(1, ND)]

    for i in range(0, int(ND) - 1):
        for j in range(i + 1, int(ND)):
            rho[0, i] = rho[0, i] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
            # print("rho[i]", np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc)))
            rho[0, j] = rho[0, j] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))

    maxd = max(np.max(dist, axis=0))  # 第一个max计算dist的每列最大值，第二个max计算所有最大值；

    rho_sorted = np.sort(rho,axis=1)  # 降序排列每个i的local density。rho_sorted为降序排列后的，
                                  # ordrho为记录排序后原先的位置
    rho_list = list(rho_sorted)
    rho_list = list(itertools.chain.from_iterable(rho_list))
    rho_list.reverse()
    rho_list = np.array(rho_list)
    rho_list = rho_list.reshape(-1, rho_list.shape[0])
    rho_sorted = rho_list
    ordrho = np.argsort(-rho,axis=1)

    # delta代表i个点的距离δi，令最大local density的那个点的delta为-1
    # print("ordrho[0,0]",ordrho[0,0])
    delta = np.zeros((int(ND)))
    delta[ordrho[0,0] - 1] = -1.
    delta = delta.reshape(-1, delta.shape[0])

    # nneigh代表点i的距离最近密度比i大的点j，令最大local density的那个点的nneigh为-1
    nneigh = np.zeros((int(ND)))
    nneigh[ordrho[0,0] - 1] = 0
    nneigh = nneigh.reshape(-1, nneigh.shape[0])

    for ii in range(1, int(ND)):  # ND代表数据个数，判断距离，如果rho(j)>rho(i),delta（i）=min（dist（i,j））
        delta[0, ordrho[0, ii]] = maxd  # local density第二大的点为maxd
        # print("delta[ordrho[0, ii]]",delta[0, ordrho[0, ii]])
        for jj in range(0, ii - 1):  # 按照降序排列以后，ordrho(jj）的密度>ordrho(ii）的密度
            if (dist[ordrho[0, ii], ordrho[0, jj]] < delta[0, ordrho[0, ii]]):
                delta[0, ordrho[0,ii]] = dist[ordrho[0, ii], ordrho[0, jj]]
                nneigh[0, ordrho[0,ii]] = ordrho[0, jj]

    delta[0, ordrho[0, 0]] = maxd  # 该行是否有问题，应为delta(ordrho(1))=maxd
                             # 让密度最大点的距离为距离中最大的点。
    # 以下是将数组转置计算
    rho_sorted = rho_sorted.T
    rho = rho.T
    ordrho = ordrho.T
    nneigh = nneigh.T
    delta = delta.T
    # 数组转置计算完成

    b = np.argmin(nneigh)  # 找出nneigh关系中密度最大值的位置
    nneigh[b,0] = b

    return nneigh