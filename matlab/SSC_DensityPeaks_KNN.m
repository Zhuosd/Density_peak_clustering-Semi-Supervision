
%%%% 基于密度峰值的半监督分类，KNN方法
%%%% 输入：训练集xi标签lable_xi；测试集test标签lable_test；类个数class_n;
%%%% 每类采集的比例ratio,0-1;K近邻的K值；经过密度峰值计算后的指向图关系nneigh；
%%%% t 为初始标签点index
%%%% 输出：分类精度accuracy，预测标签predict_label

function [predict_label_train,predict_label_test]=SSC_DensityPeaks_KNN(train,label_train,test,label_test,t,K,nneigh)

    data(1:length(t),:)=train(t,:);
    label_data(1:length(t),1)=label_train(t);
    struct=t;
    struct_record=struct;%保存每一次添加未标记样本后的数据点结果
    length_data=length(data);
    length_struct=length(struct);
    t_U=setdiff((1:size(train,1)),t);
    
    U=train(t_U,:);%为标记样本集
    label_U(1:size(U,1),1)=label_train(t_U,1);%;%未标记样本的标签
    %fprintf('label_U %12.4', label_U)

    %先选择指向下一个节点
    while(length(struct)>0)
        for i=1:length_struct
            data_neigh(i) = nneigh(struct(i));
        end

        data_neigh = unique(data_neigh,'rows');%查找唯一
        struct=setdiff(data_neigh,struct_record);%矩阵A-B中有的元素； 
        length_struct=length(struct);  

        for i=1:length_struct
            struct_record(length_data+i)=struct(i);
        end

        data_TR=data;%中间过渡一下，用于调用KNN
        lable_TR=label_data;%中间过渡一下，用于调用KNN

        for j=1:length_struct
            data(length_data+j,:)=train(struct(j),:);
            label_data(length_data+j)=label_train(j);%随便给新加入的训练样本一个标签，为了在调用KNN模型时数据行数一致
        end

        length_data=length(data);%求数据长度
        % format long, single(data_TR)
        % format long, single(data)
        % data_TR=long(data_TR);
        %fprintf('data_TR=%d\n',data_TR);
        %fprintf('lable_TR=%d\n',lable_TR);
        %fprintf('data=%d\n',data);
        %fprintf('label_data=%d\n',label_data);
        %fprintf('K=%d\n',K);
        
        predict_label=knnclassify(data, data_TR, lable_TR, K); % accuracy
        label_data=predict_label;
    end

    struct=struct_record;
    length_struct=length(struct); 
    clear data_neigh;

    %再选择被指向上一个点
    while( length(struct)>0 ) 
        k=0;%查找所有下节点 
        for i=1:length_struct
            number_neigh=find(nneigh==struct(i));
            length_neigh=length(number_neigh);
            if length_neigh>0
                for j=1:length_neigh
                    data_neigh(k+j)=number_neigh(j);
                end
            end
            k=k+length_neigh;
        end

        struct=setdiff(data_neigh,struct_record);%矩阵A-B；
        length_struct=length(struct);  
        % clear data_neigh;%如果清除的话，到最后一次，会出错，因为最后 data_neigh不会生成。

        for i=1:length_struct
            struct_record(length_data+i)=struct(i);
        end 

        data_TR=data;%中间过渡一下，用于调用KNN
        lable_TR=label_data;%中间过渡一下，用于调用KNN

        for j=1:length_struct
            data(length_data+j,:)=train(struct(j),:);
            label_data(length_data+j)=label_train(j);%随便给新加入的训练样本一个标签，为了在调用KNN模型时数据行数一致
        end

        length_data=length(data);%求数据长度

        predict_label=knnclassify(data, data_TR,lable_TR,K); % accuracy
        label_data=predict_label;
    end
    predict_label_test=knnclassify(test, data, label_data, K);
    predict_label_train=knnclassify(U, data, label_data, K);
    fprintf('label_test: %d\n', label_test);
    fprintf('predict_label_test: %d\n', predict_label_test);
    fprintf('label_U:    %d\n', label_U);
    fprintf('predict_label_train: %d\n', predict_label_train);
end

















 