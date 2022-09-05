import numpy as np

def SSC_DensityPeaks_KNN(train, label_train, test, label_test, t, K, nneigh):
    data = []
    label_data = []
    label_U = []

    data[1:len(t), :] = train[t, :]
    label_data[1:len(t), 1] = label_train[t]
    struct = t

    struct_record = struct  # 保存每一次添加未标记样本后的数据点结果
    length_data = len(data)
    length_struct = len(struct)
    t_U = np.setdiff((1,np.size(train,1)), t)
    U = train[t_U, :]  # 为标记样本集
    label_U[1, np.size(U, 1), 1] = label_train(t_U, 1)  # 未标记样本的标签

    '''先选择指向下一个节点'''
    data_neigh = []
    struct_record = []
    while (len(struct) > 0):
        for i in range(1, len(struct)):
            data_neigh[i] = nneigh(struct(i))

        data_neigh = np.unique(data_neigh, 'rows')  # 查找唯一
        struct = np.setdiff(data_neigh, struct_record)  # 矩阵A-B中有的元素；
        length_struct = len(struct)

        for i in range(1, length_struct):
            struct_record[length_data + i] = struct(i)

        data_TR = data
        lable_TR = label_data
        data = []
        label_data = []
        for j in range(1, length_struct):
            data[length_data + j, :] = train[struct(j), :]
            label_data[length_data + j] = label_train[j]
        length_data = len(data)
        accuracy, predict_label = KNN_classifier(data_TR, lable_TR, data, label_data, K)
        label_data = predict_label

    struct = struct_record;
    length_struct = len(struct)

    # 再选择被指向上一个点
    while (len(struct) > 0):
        k = 0 # 查找所有下节点
        for i in range(1, len(struct)):
            number_neigh = np.find(nneigh == struct[i])
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(1, length_neigh):
                    data_neigh[k + j] = number_neigh[j]
        k = k + length_neigh

        struct = np.setdiff(data_neigh, struct_record)  # 矩阵A-B；
        length_struct = len(struct)

        # clear data_neigh;%如果清除的话，到最后一次，会出错，因为最后 data_neigh不会生成。

        for i in range(1, length_struct):
            struct_record[length_data + i] = struct(i)

        data_TR = data  # 中间过渡一下，用于调用KNN
        lable_TR = label_data  # 中间过渡一下，用于调用KNN

        for j in range(1, length_struct):
            data[length_data + j, :] = train[struct(j), :]
            label_data[length_data + j] = label_train[j]  # 随便给新加入的训练样本一个标签，为了在调用KNN模型时数据行数一致

        length_data = len(data)  # 求数据长度

        accuracy, predict_label = KNN_classifier(data_TR, lable_TR, data, label_data, K);
        label_data = predict_label

    Accuracy_test, predict_label_test = KNN_classifier(data, label_data, test, label_test, K);
    Accuracy_train, predict_label_train = KNN_classifier(data, label_data, U, label_U, K);

    return Accuracy_test, predict_label_test, Accuracy_train, predict_label_train
