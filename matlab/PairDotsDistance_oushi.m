%%%%计算矩阵两两数据点的距离的函数，适用密度峰值距离发现数据结构
%%%%输入：数据矩阵Dots,行数NDim, 列数numOfDots 
%%%%输出：距离矩阵DistanceMat,第一列为数据点i, 第二列另一数据点j，第三列为ij间的距离 

function DistanceMat = PairDotsDistance(Dots, NDim, numOfDots)
    Len=numOfDots*(numOfDots-1)/2;
    DistanceMat=zeros(3,Len);
    %fprintf('Len= %12.6f\n',Len)
    matIndex=1;
    for i=1:numOfDots-1
        for j=i+1:numOfDots
            DistanceMat(1,matIndex)=i; 
            DistanceMat(2,matIndex)=j;
        %Euclidian Distance
        % fprintf('Dots(:,i)= %12.6f\n',Dots(:,i))
        % fprintf('Dots(:,j)= %12.6f\n',Dots(:,j))
        % fprintf('Dots(:,i)-Dots(:,j)= %12.6f\n',Dots(:,i)-Dots(:,j))
        % fprintf('sqrt(sum((Dots(:,i)-Dots(:,j)).^2))= %12.6f\n',sqrt(sum((Dots(:,i)-Dots(:,j)).^2)))
        DistanceMat(3,matIndex)=sqrt(sum((Dots(:,i)-Dots(:,j)).^2));%欧式距离公式
        matIndex=matIndex+1;
        end
        % fprintf('i=%d\n',i);
    end
end