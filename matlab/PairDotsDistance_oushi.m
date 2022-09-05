%%%%��������������ݵ�ľ���ĺ����������ܶȷ�ֵ���뷢�����ݽṹ
%%%%���룺���ݾ���Dots,����NDim, ����numOfDots 
%%%%������������DistanceMat,��һ��Ϊ���ݵ�i, �ڶ�����һ���ݵ�j��������Ϊij��ľ��� 

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
        DistanceMat(3,matIndex)=sqrt(sum((Dots(:,i)-Dots(:,j)).^2));%ŷʽ���빫ʽ
        matIndex=matIndex+1;
        end
        % fprintf('i=%d\n',i);
    end
end