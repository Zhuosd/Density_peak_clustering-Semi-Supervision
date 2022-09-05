clear
format short;
load IRIS.mat;


%------------------调用密度峰值计算----------------------------------
percent=2;%% 密度峰值参数
nneigh=DensityPeaks(train,percent);%调用密度峰值计算结构
%fprintf('nneigh: %12.6f\n', nneigh);
%fprintf('train: %12.6f\n', train);
% fprintf('nneigh: %12.6f\n', nneigh);
%------------------调用密度峰值计算----------------------------------


%------------------半监督----------------------------------
K=3;%KNN K

[DP_KNN_ac_train,DP_KNN_ac_test]=SSC_DensityPeaks_KNN(train,label_train,test,label_test,initial_label,K,nneigh);
%fprintf('DP_KNN_ac_train: %12.6f\n', DP_KNN_ac_train);
%fprintf('DP_KNN_ac_test:  %12.6f\n', DP_KNN_ac_test);
