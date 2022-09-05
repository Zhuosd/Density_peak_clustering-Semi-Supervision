import warnings
import itertools
import numpy as np
import scipy.io as scio
from sklearn import neighbors
warnings.filterwarnings("ignore")
from sklearn import *
from ftrl_adp import *
from sklearn.metrics import accuracy_score

# 定义常量
K=3
percent=2

'''
读取mat文件，并将文件的数据切分为不同的数据label
mat文件中所包括的文件标签有：test、initial_label、train、label_train、label_test
'''
# Finish
def read_file(path):
    matpath = path #"./IRIS.mat"
    data = scio.loadmat(matpath)
    test          = data['test']
    initial_label = data['initial_label']
    train         = data['train']
    label_train   = data['label_train']
    label_test    = data['label_test']
    return test, initial_label, train, label_train, label_test

'''
计算矩阵两两数据点的距离的函数，适用密度峰值距离发现数据结构
输入：数据矩阵Dots,行数NDim, 列数numOfDots
输出：距离矩阵DistanceMat,第一列为数据点i, 第二列另一数据点j，第三列为ij间的距离 
'''
# Finish
def list_of_groups(list_info, per_list_len):
    list_of_group = zip(*(iter(list_info),) *per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list
# Finish
def PairDotsDistance(Dots, NDim, numOfDots):
    Len = numOfDots * (numOfDots - 1) / 2
    DistanceMat = []
    matIndex = 0
    for i in range(0, numOfDots - 1):
        for j in range(i + 1, numOfDots):
            DistanceMat.append(i)
            DistanceMat.append(j)
            DistanceMat.append(np.sqrt(np.sum((Dots[:, i] - Dots[:, j])**2)))
        matIndex = matIndex + 1
    Distance = np.array(list_of_groups(DistanceMat, 3))
    return Distance

'''
计算数据密度峰值，发现数据指向结构关系
输入：数据xi， 邻居的百分比percent；
输出：表面数据指向结构的数据nneigh；
'''
# Finish
def DensityPeaks(xi, precent):
    rowN, colN = xi.shape[0], xi.shape[1]  # 求输入数据矩阵的个数（行数rowN），属性维度（列数colN）
    xiT = xi.T
    xx = PairDotsDistance(xiT, colN, rowN)  # j计算密度峰值中distance matrix file
    xxT = xx
    ND = np.max(xxT[:, 1]) + 1  # 第二列的最大值
    NL = np.max(xxT[:, 0]) + 1  # 第一列的最大值

    if NL > ND:
        ND = NL
    N = xxT.shape[0]

    dist = np.zeros((int(ND),int(ND)))

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

'''
基于密度峰值的半监督分类，KNN方法;
输入：训练集xi标签lable_xi；测试集test标签lable_test；类个数class_n;
每类采集的比例ratio,0-1;K近邻的K值；经过密度峰值计算后的指向图关系nneigh;
t 为初始标签点index;
输出：分类精度accuracy，预测标签predict_label;
'''
# Finish
def SSC_DensityPeaks_KNN(train, label_train, test, label_test, t, K, nneigh):
    data = []
    label_data = []
    label_U = []
    # print("train[t,]",t-1)

    # 声明模型变量
    # clf = linear_model.PassiveAggressiveClassifier()      # 0.666
    # clf = tree.ExtraTreeClassifier(random_state=0)        # 0.5
    # clf = ensemble.AdaBoostClassifier()                   # 0.616
    # clf = ensemble.BaggingClassifier()                    # 0.883
    # clf = ensemble.ExtraTreesClassifier()                 # 0.883
    # clf = ensemble.GradientBoostingClassifier()           # 0.700
    # clf = ensemble.RandomForestClassifier()               # 0.650
    # clf = linear_model.RidgeClassifier()                  # 0.633
    # clf = linear_model.RidgeClassifierCV()                # 0.633
    # clf = linear_model.SGDClassifier()                    # 0.666
    # clf = dummy.DummyClassifier()                         # 0.333
    # clf = neural_network.MLPClassifier()                  # 0.933
    # clf = tree.DecisionTreeClassifier()                   # 0.950
    # clf = tree.ExtraTreeClassifier()                      # 0.816
    # clf = neighbors.RadiusNeighborsClassifier(radius=1.6) # 0.763
    clf = neighbors.KNeighborsClassifier(K, weights='distance',algorithm='brute')   # 0.9133 或 0.883  # algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']:
    # clf = neighbors.KNeighborsClassifier(K, weights='uniform',algorithm='brute')    # 0.883


    data.append(train[t-1,])
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[3])
    label_data.append(label_train[t-1])
    label_data = np.array(label_data)
    label_data = label_data.reshape(label_data.shape[1], label_data.shape[3])
    struct = t-1

    struct_record = struct  # 保存每一次添加未标记样本后的数据点结果
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,train.shape[0],1)]) # 90

    t_U = np.setdiff1d(diff_array, t-1)
    U = train[t_U, :]  # 为标记样本集
    # print('label_train(t_U, 0)',label_train[t_U, 0])
    label_U.append(label_train[t_U, 0])  # 未标记样本的标签
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    '''先选择指向下一个节点'''
    # Step 2
    data_neigh = []
    predict_1 = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct)):
            # print("nneigh[int(struct[i])]",nneigh[int(struct[i])])
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)
        # data_neigh = data_neigh.reshape(data_neigh.shape[1],data_neigh.shape[0])

        # data_neigh = np.unique(data_neigh, 'rows')  # 查找唯一
        struct = np.setdiff1d(data_neigh, struct_record)  # 矩阵A-B中有的元素；
        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        clf.fit(data_TR, lable_TR)
        label_data = clf.predict(data)

    struct = struct_record
    length_struct = len(struct)
    del data_neigh

    '''再选择被指向上一个点'''
    # Step 3
    data_neigh = []
    predict_2 = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0 # 查找所有下节点
        for i in range(0, len(struct)):
            number_neigh = np.where(nneigh == struct[i])[0]
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record) # 矩阵A-B；
        length_struct = len(struct)

        # clear data_neigh;%如果清除的话，到最后一次，会出错，因为最后 data_neigh不会生成。
        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)


        data_TR = data  # 中间过渡一下，用于调用KNN
        lable_TR = label_data  # 中间过渡一下，用于调用KNN
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)  # 求数据长度

        clf.fit(data_TR, lable_TR)
        label_data = clf.predict(data)

    clf.fit(data, label_data)
    predict_label_train = clf.predict(U)
    print("输出U_tabel的预测标签及原始标签")
    print("predict_label_train", predict_label_train)
    print("predict_label_train", label_U.reshape(label_U.shape[1],label_U.shape[0]))
    Accuracy_train = clf.score(U, label_U)

    predict_label_test = clf.predict(test)
    print("输出test的预测标签及原始标签")
    print("predict_label_test",predict_label_test)
    print("label_test",label_test.reshape(label_test.shape[1],label_test.shape[0]))
    Accuracy_test = clf.score(test,label_test)

    return Accuracy_test, predict_label_test, Accuracy_train, predict_label_train


def SSC_DensityPeaks_ftrl(train, label_train, test, label_test, t, K, nneigh, decay_choice, contribute_error_rate, classifier):
    data = []
    label_data = []
    label_U = []
    # print("train[t,]",t-1)

    # 声明模型变量
    # clf = linear_model.PassiveAggressiveClassifier()      # 0.666
    # clf = tree.ExtraTreeClassifier(random_state=0)        # 0.5
    # clf = ensemble.AdaBoostClassifier()                   # 0.616
    # clf = ensemble.BaggingClassifier()                    # 0.883
    # clf = ensemble.ExtraTreesClassifier()                 # 0.883
    # clf = ensemble.GradientBoostingClassifier()           # 0.700
    # clf = ensemble.RandomForestClassifier()               # 0.650
    # clf = linear_model.RidgeClassifier()                  # 0.633
    # clf = linear_model.RidgeClassifierCV()                # 0.633
    # clf = linear_model.SGDClassifier()                    # 0.666
    # clf = dummy.DummyClassifier()                         # 0.333
    # clf = neural_network.MLPClassifier()                  # 0.933
    # clf = tree.DecisionTreeClassifier()                   # 0.950
    # clf = tree.ExtraTreeClassifier()                      # 0.816
    # clf = neighbors.RadiusNeighborsClassifier(radius=1.6) # 0.763
    # clf = neighbors.KNeighborsClassifier(K, weights='distance',algorithm='brute')   # 0.9133 或 0.883  # algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']:
    # clf = neighbors.KNeighborsClassifier(K, weights='uniform',algorithm='brute')    # 0.883

    classifier = classifier

    data.append(train[t-1,])
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[3])
    label_data.append(label_train[t-1])
    label_data = np.array(label_data)
    label_data = label_data.reshape(label_data.shape[1], label_data.shape[3])
    struct = t-1

    struct_record = struct  # 保存每一次添加未标记样本后的数据点结果
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,train.shape[0],1)]) # 90

    t_U = np.setdiff1d(diff_array, t-1)
    U = train[t_U, :]  # 为标记样本集
    # print('label_train(t_U, 0)',label_train[t_U, 0])
    label_U.append(label_train[t_U, 0])  # 未标记样本的标签
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    '''先选择指向下一个节点'''
    # Step 2
    data_neigh = []
    predict_1 = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct)):
            # print("nneigh[int(struct[i])]",nneigh[int(struct[i])])
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)
        # data_neigh = data_neigh.reshape(data_neigh.shape[1],data_neigh.shape[0])

        # data_neigh = np.unique(data_neigh, 'rows')  # 查找唯一
        struct = np.setdiff1d(data_neigh, struct_record)  # 矩阵A-B中有的元素；
        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        # clf.fit(data_TR, lable_TR)
        # label_data = clf.predict(data)
        n = len(data_TR)
        for row in range(n):
            indices = [i for i in range(data_TR.shape[1])]
            x = data_TR[row]
            y = lable_TR[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            if p < 0.3:
                p = 0
            else:
                p = 1
            predict_1.append(p)

    struct = struct_record
    length_struct = len(struct)
    del data_neigh

    '''再选择被指向上一个点'''
    # Step 3
    data_neigh = []
    predict_2 = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0 # 查找所有下节点
        for i in range(0, len(struct)):
            number_neigh = np.where(nneigh == struct[i])[0]
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record) # 矩阵A-B；
        length_struct = len(struct)

        # clear data_neigh;%如果清除的话，到最后一次，会出错，因为最后 data_neigh不会生成。
        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)


        data_TR = data  # 中间过渡一下，用于调用KNN
        lable_TR = label_data  # 中间过渡一下，用于调用KNN
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)  # 求数据长度

        # clf.fit(data_TR, lable_TR)
        # label_data = clf.predict(data)
        n = len(data_TR)
        for row in range(n):
            indices = [i for i in range(data_TR.shape[1])]
            x = data_TR[row]
            y = lable_TR[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            if p < 0.3:
                p = 0
            else:
                p = 1
            predict_2.append(p)

    # clf.fit(data, label_data)
    # predict_label_train = clf.predict(U)
    # print("输出U_tabel的预测标签及原始标签")
    # print("predict_label_train", predict_label_train)
    # print("predict_label_train", label_U.reshape(label_U.shape[1],label_U.shape[0]))
    # Accuracy_train = clf.score(U, label_U)

    predict_label_train = []
    n = len(U)
    for row in range(n):
        indices = [i for i in range(U.shape[1])]
        x = U[row]
        y = label_U[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        if p < 0.3:
            p = 0
        else:
            p = 1
        predict_label_train.append(p)
    predict_label_train = np.array(predict_label_train)
    Accuracy_train = accuracy_score(label_U, predict_label_train)

    # predict_label_test = clf.predict(test)
    # print("输出test的预测标签及原始标签")
    # print("predict_label_test",predict_label_test)
    # print("label_test",label_test.reshape(label_test.shape[1],label_test.shape[0]))
    # Accuracy_test = clf.score(test,label_test)
    predict_label_test = []
    n = len(test)
    for row in range(n):
        indices = [i for i in range(test.shape[1])]
        x = test[row]
        y = label_test[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        if p < 0.3:
            p = 0
        else:
            p = 1
        predict_label_test.append(p)
    predict_label_test = np.array(predict_label_test)
    Accuracy_test = accuracy_score(label_test, predict_label_test)

    return Accuracy_test, predict_label_test, Accuracy_train, predict_label_train

if __name__ == '__main__':
    path = "./IRIS.mat"
    percent = 2
    test, initial_label, train, label_train, label_test  = read_file(path)

    nneigh = DensityPeaks(train, percent)
    classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=train.shape[1])
    decay_choices_array = [0, 1, 2, 3, 4]
    contribute_error_rates_array = [0.005, 0.01, 0.02, 0.03, 0.04]
    for decay_choice in decay_choices_array:
        print("decay_choice", decay_choice)
        for contribute_error_rate in contribute_error_rates_array:
            print("contribute_error_rate", contribute_error_rate)
            Accuracy_test, predict_label_test, Accuracy_train, predict_label_train = SSC_DensityPeaks_ftrl(train, label_train,
                                                                                                          test, label_test, initial_label,
                                                                                                          K, nneigh, decay_choice,
                                                                                                          contribute_error_rate, classifier)
            print("Accuracy_train", Accuracy_train)
            print("Accuracy_test" , Accuracy_test)

    print("###########################################")
    print("这是KNN的准确率")
    Accuracy_test, predict_label_test, Accuracy_train, predict_label_train = SSC_DensityPeaks_KNN(train, label_train,
                                                                                                  test, label_test,
                                                                                                  initial_label,
                                                                                                  K, nneigh)
    print("Accuracy_train", Accuracy_train)
    print("Accuracy_test", Accuracy_test)
