%%%%���������ܶȷ�ֵ����������ָ��ṹ��ϵ
%%%%���룺����xi�� �ھӵİٷֱ�percent��
%%%%�������������ָ��ṹ������nneigh��

function nneigh=DensityPeaks(xi,percent)
    [rowN,colN]=size(xi);%���������ݾ���ĸ���������rowN��������ά�ȣ�����colN��
    xx=(PairDotsDistance_oushi(xi',colN,rowN))';%j�����ܶȷ�ֵ��distance matrix file��'����ת�þ���
    %fprintf('xx: %12.6f\n', xx);
    %xx�Ǿ�������ˣ�ŷʽ��������ƺ������ã�����پ��벻����������

    ND=max(xx(:,2));%�ڶ��е����ֵ
    %fprintf('ND: %12.6f\n', ND);
    NL=max(xx(:,1));%��һ�е����ֵ
    if (NL>ND)   %ND�������ݸ���,�����ݸ���
      ND=NL;
    end
    N=size(xx,1);
    for i=1:ND          %��ʼ���������dist��i,j)
      for j=1:ND
        dist(i,j)=0;
      end
    end
    %fprintf('dist %6f\n',dist);
    %fprintf('xx: %12.6f\n', xx);
    for i=1:N   %�������dist��i,j)�����˾�����Ϣ
      ii=xx(i,1);
      jj=xx(i,2);
      dist(ii,jj)=xx(i,3);
      dist(jj,ii)=xx(i,3);
    end
    
    %  percent=1;%ע���ֵ��ȡֵ����ԱȽϹؼ���ԭ��Ϊ2.0��
    fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent); %�Ƿ� %fʵ��С�������Ϊ5��С�����6λ

    position=round(N*percent/100);%��ֹ����dc��λ�ã���������round��1.4��=1�� ��-ȡ��floor��3.3��=3�� ��+ȡ��ceil��4.23��=5��
    sda=sort(xx(:,3));%�Ծ�����������
    dc=sda(position);%�����еĽ�ֹ����dc

    fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);

    for i=1:ND
      rho(i)=0.;%��ÿ��Pi��Rho����local density����ʼֵ
    end
    %fprintf('rho: %12.6f\n', rho);
    
    % Gaussian kernel ��˹�ֲ�����Rho
    for i=1:ND-1  %����ÿ�����local density��rho��i��Ϊi�����local density
      for j=i+1:ND %�������ø�˹�˶Ծ��빱��������i�����local density��
         rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
         %fprintf('exp(-(dist(i,j)/dc)*(dist(i,j)/dc)): %12.6f\n', exp(-(dist(i,j)/dc)*(dist(i,j)/dc)));
         
         rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
         %fprintf('rho(j): %12.6f\n', rho(j));
      end
    end
    
    maxd=max(max(dist));%��һ��max����dist��ÿ�����ֵ���ڶ���max�����������ֵ��

    [rho_sorted,ordrho]=sort(rho,'descend');%��������ÿ��i��local density��rho_sortedΪ�������к�ģ�ordrhoΪ��¼�����ԭ�ȵ�λ��
    delta(ordrho(1))=-1.;%delta����i����ľ����i�������local density���Ǹ����deltaΪ-1
    %fprintf('delta(j): %12.6f\n', delta);
    nneigh(ordrho(1))=0;%nneigh�����i�ľ�������ܶȱ�i��ĵ�j�������local density���Ǹ����nneighΪ-1
    %fprintf('nneigh(j): %12.6f\n', nneigh);
    
    for ii=2:ND%,ND�������ݸ������жϾ��룬���rho(j)>rho(i),delta��i��=min��dist��i,j����
       delta(ordrho(ii))=maxd;%local density�ڶ���ĵ�Ϊmaxd
       %fprintf('delta(ordrho(ii)): %12.6f\n', delta(ordrho(ii)));
       for jj=1:ii-1%���ս��������Ժ�ordrho(jj�����ܶ�>ordrho(ii�����ܶ�
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
            nneigh(ordrho(ii))=ordrho(jj);
         end%����ii=3��jj=1,2���ܶȶ���ii=3���ܶȴ󣬽�����Ϊ��
       end
    end
    
    delta(ordrho(1))=max(delta(:));%�����Ƿ������⣬ӦΪdelta(ordrho(1))=maxd�������ܶ�����ľ���Ϊ���������ĵ㡣
    %fprintf('delta(ordrho(1)): %12.6f\n', delta(ordrho(1)));
    %�����ǽ�����תְ����
    rho_sorted = rho_sorted';
    rho = rho';
    ordrho = ordrho';
    nneigh = nneigh';
    delta = delta';
    %����תְ�������

    [a,b] = min(nneigh);%�ҳ�nneigh��ϵ���ܶ�����ֵ��λ��
    %fprintf('min(nneigh): %12.4f\n',min(nneigh))
    %fprintf('a: %12.6f\n', a);
    %fprintf('b: %12.6f\n', b);
    nneigh(b)=b;
end

