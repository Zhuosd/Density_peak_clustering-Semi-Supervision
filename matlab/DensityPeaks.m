%%%%计算数据密度峰值，发现数据指向结构关系
%%%%输入：数据xi， 邻居的百分比percent；
%%%%输出：表面数据指向结构的数据nneigh；

function nneigh=DensityPeaks(xi,percent)
    [rowN,colN]=size(xi);%求输入数据矩阵的个数（行数rowN），属性维度（列数colN）
    xx=(PairDotsDistance_oushi(xi',colN,rowN))';%j计算密度峰值中distance matrix file，'代表转置矩阵
    %fprintf('xx: %12.6f\n', xx);
    %xx是距离矩阵了，欧式距离用徐计函数调用，麦哈顿距离不用上面两行

    ND=max(xx(:,2));%第二列的最大值
    %fprintf('ND: %12.6f\n', ND);
    NL=max(xx(:,1));%第一列的最大值
    if (NL>ND)   %ND代表数据个数,求数据个数
      ND=NL;
    end
    N=size(xx,1);
    for i=1:ND          %初始化距离矩阵dist（i,j)
      for j=1:ND
        dist(i,j)=0;
      end
    end
    %fprintf('dist %6f\n',dist);
    %fprintf('xx: %12.6f\n', xx);
    for i=1:N   %距离矩阵dist（i,j)记载了距离信息
      ii=xx(i,1);
      jj=xx(i,2);
      dist(ii,jj)=xx(i,3);
      dist(jj,ii)=xx(i,3);
    end
    
    %  percent=1;%注意此值的取值，相对比较关键，原来为2.0；
    fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent); %是否 %f实数小数，宽度为5，小数点后6位

    position=round(N*percent/100);%截止距离dc的位置，四舍五入round（1.4）=1， 向-取整floor（3.3）=3， 向+取整ceil（4.23）=5，
    sda=sort(xx(:,3));%对距离排序，升序
    dc=sda(position);%文章中的截止距离dc

    fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);

    for i=1:ND
      rho(i)=0.;%对每个Pi（Rho）的local density赋初始值
    end
    %fprintf('rho: %12.6f\n', rho);
    
    % Gaussian kernel 高斯分布计算Rho
    for i=1:ND-1  %计算每个点的local density，rho（i）为i个点的local density
      for j=i+1:ND %这里利用高斯核对距离贡献来计算i个点的local density，
         rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
         %fprintf('exp(-(dist(i,j)/dc)*(dist(i,j)/dc)): %12.6f\n', exp(-(dist(i,j)/dc)*(dist(i,j)/dc)));
         
         rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
         %fprintf('rho(j): %12.6f\n', rho(j));
      end
    end
    
    maxd=max(max(dist));%第一个max计算dist的每列最大值，第二个max计算所有最大值；

    [rho_sorted,ordrho]=sort(rho,'descend');%降序排列每个i的local density。rho_sorted为降序排列后的，ordrho为记录排序后原先的位置
    delta(ordrho(1))=-1.;%delta代表i个点的距离δi，令最大local density的那个点的delta为-1
    %fprintf('delta(j): %12.6f\n', delta);
    nneigh(ordrho(1))=0;%nneigh代表点i的距离最近密度比i大的点j，令最大local density的那个点的nneigh为-1
    %fprintf('nneigh(j): %12.6f\n', nneigh);
    
    for ii=2:ND%,ND代表数据个数，判断距离，如果rho(j)>rho(i),delta（i）=min（dist（i,j））
       delta(ordrho(ii))=maxd;%local density第二大的点为maxd
       %fprintf('delta(ordrho(ii)): %12.6f\n', delta(ordrho(ii)));
       for jj=1:ii-1%按照降序排列以后，ordrho(jj）的密度>ordrho(ii）的密度
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
            nneigh(ordrho(ii))=ordrho(jj);
         end%比如ii=3，jj=1,2的密度都比ii=3的密度大，降序因为是
       end
    end
    
    delta(ordrho(1))=max(delta(:));%该行是否有问题，应为delta(ordrho(1))=maxd？？让密度最大点的距离为距离中最大的点。
    %fprintf('delta(ordrho(1)): %12.6f\n', delta(ordrho(1)));
    %以下是将数组转职计算
    rho_sorted = rho_sorted';
    rho = rho';
    ordrho = ordrho';
    nneigh = nneigh';
    delta = delta';
    %数组转职计算完成

    [a,b] = min(nneigh);%找出nneigh关系中密度最大的值得位置
    %fprintf('min(nneigh): %12.4f\n',min(nneigh))
    %fprintf('a: %12.6f\n', a);
    %fprintf('b: %12.6f\n', b);
    nneigh(b)=b;
end

