clear
format short;
load IRIS.mat;


%------------------�����ܶȷ�ֵ����----------------------------------
percent=2;%% �ܶȷ�ֵ����
nneigh=DensityPeaks(train,percent);%�����ܶȷ�ֵ����ṹ
%fprintf('nneigh: %12.6f\n', nneigh);
%fprintf('train: %12.6f\n', train);
% fprintf('nneigh: %12.6f\n', nneigh);
%------------------�����ܶȷ�ֵ����----------------------------------


%------------------��ල----------------------------------
K=3;%KNN K

[DP_KNN_ac_train,DP_KNN_ac_test]=SSC_DensityPeaks_KNN(train,label_train,test,label_test,initial_label,K,nneigh);
%fprintf('DP_KNN_ac_train: %12.6f\n', DP_KNN_ac_train);
%fprintf('DP_KNN_ac_test:  %12.6f\n', DP_KNN_ac_test);
