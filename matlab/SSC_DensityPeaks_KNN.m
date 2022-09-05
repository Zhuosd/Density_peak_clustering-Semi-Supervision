
%%%% �����ܶȷ�ֵ�İ�ල���࣬KNN����
%%%% ���룺ѵ����xi��ǩlable_xi�����Լ�test��ǩlable_test�������class_n;
%%%% ÿ��ɼ��ı���ratio,0-1;K���ڵ�Kֵ�������ܶȷ�ֵ������ָ��ͼ��ϵnneigh��
%%%% t Ϊ��ʼ��ǩ��index
%%%% ��������ྫ��accuracy��Ԥ���ǩpredict_label

function [predict_label_train,predict_label_test]=SSC_DensityPeaks_KNN(train,label_train,test,label_test,t,K,nneigh)

    data(1:length(t),:)=train(t,:);
    label_data(1:length(t),1)=label_train(t);
    struct=t;
    struct_record=struct;%����ÿһ�����δ�������������ݵ���
    length_data=length(data);
    length_struct=length(struct);
    t_U=setdiff((1:size(train,1)),t);
    
    U=train(t_U,:);%Ϊ���������
    label_U(1:size(U,1),1)=label_train(t_U,1);%;%δ��������ı�ǩ
    %fprintf('label_U %12.4', label_U)

    %��ѡ��ָ����һ���ڵ�
    while(length(struct)>0)
        for i=1:length_struct
            data_neigh(i) = nneigh(struct(i));
        end

        data_neigh = unique(data_neigh,'rows');%����Ψһ
        struct=setdiff(data_neigh,struct_record);%����A-B���е�Ԫ�أ� 
        length_struct=length(struct);  

        for i=1:length_struct
            struct_record(length_data+i)=struct(i);
        end

        data_TR=data;%�м����һ�£����ڵ���KNN
        lable_TR=label_data;%�м����һ�£����ڵ���KNN

        for j=1:length_struct
            data(length_data+j,:)=train(struct(j),:);
            label_data(length_data+j)=label_train(j);%�����¼����ѵ������һ����ǩ��Ϊ���ڵ���KNNģ��ʱ��������һ��
        end

        length_data=length(data);%�����ݳ���
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

    %��ѡ��ָ����һ����
    while( length(struct)>0 ) 
        k=0;%���������½ڵ� 
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

        struct=setdiff(data_neigh,struct_record);%����A-B��
        length_struct=length(struct);  
        % clear data_neigh;%�������Ļ��������һ�Σ��������Ϊ��� data_neigh�������ɡ�

        for i=1:length_struct
            struct_record(length_data+i)=struct(i);
        end 

        data_TR=data;%�м����һ�£����ڵ���KNN
        lable_TR=label_data;%�м����һ�£����ڵ���KNN

        for j=1:length_struct
            data(length_data+j,:)=train(struct(j),:);
            label_data(length_data+j)=label_train(j);%�����¼����ѵ������һ����ǩ��Ϊ���ڵ���KNNģ��ʱ��������һ��
        end

        length_data=length(data);%�����ݳ���

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

















 