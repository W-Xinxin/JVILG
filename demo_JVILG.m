%% sum_v=1^m av^2 (||PvXv + EvGv - AvWvFv'||_F^2 + beta*|Ev-AvZv|_F^2 +lambda |Zv|_1 )+ phi* ||F^||_Sp^p 
%%  code by xinxin,  21/07/2024   

clear;
clc;

addpath(genpath('./'));

resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/'));
end

datadir='./datasets/';

dataname={'BDGP_fea'};

numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};

for idata = 1
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
    for dataIndex = 1:1:9
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        disp(datafile);
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        cls_num = length(unique(gt));
        k= cls_num;
        tic;
        [X1, ind] = findindex(data, index);

        time1 = toc;
        maxAcc = 0;      
        TempBeta = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4];
%         TempBeta =[1e-4];
        TempLambda= [1];
        TempPhi = [1e-1,1,1e1,5e1,1e2,5e2];        % [1e-1,1,1e1,5e1,1e2,5e2];
%         TempPhi = [1e2];
        TempP = [0.1:0.1:1];                     % for low rank tensor 
%         TempP = [0.1];
        m_list =[1*k,5*k,10*k,20*k,30*k];        % for anchor number;  original m=k;
%         m_list=[1*k];
        dim = 1*k;

        ACC = zeros(length(TempBeta),length(m_list),length(TempPhi),length(TempP));
        NMI = zeros(length(TempBeta),length(m_list),length(TempPhi),length(TempP));
        Purity = zeros(length(TempBeta),length(m_list),length(TempPhi),length(TempP));
        idx = 1;
        lambda2 = 1;
            for LambdaIndex1 = 1 : length(TempBeta)
             lambda1 = TempBeta(LambdaIndex1);  
%              for LambdaIndex2 = 1 : length(TempLambda) 
%              lambda2 = TempLambda(LambdaIndex2);  
              for LambdaIndex2 = 1 : length(m_list) 
                 m = m_list(LambdaIndex2);
               for LambdaIndex3 = 1 : length(TempPhi) 
                  lambda3 = TempPhi(LambdaIndex3);  
                  for LambdaIndex4 = 1 : length(TempP)
                    p = TempP(LambdaIndex4);
                    disp([char(dataname(idata)), char(numname(dataIndex)),'-b1=', num2str(lambda1),'-m=',num2str(m),'-l2=', num2str(lambda2),'-f2=', num2str(lambda3) , '-p=', num2str(p),'-dim=',num2str(dim)]);
                    tic;
                    para.c = cls_num; % K: number of clusters
                    [PreY,obj,iter] = algo_JVILG(X1,gt,m,lambda1,lambda2,lambda3,p,ind); % X,Y,lambda,d,numanchor
                    time2 = toc;
                    tic;
                    for rep = 1 : 10
                        res(rep, : ) = Clustering8Measure(gt, PreY);
                    end
                    time3 = toc;

                    runtime(idx) = time1 + time2 + time3/10; 
                    disp(['runtime:', num2str(runtime(idx))])
                    idx = idx + 1;
                    tempResBest(dataIndex, : ) = mean(res);
                    tempResStd(dataIndex, : ) = std(res);
                    ACC(LambdaIndex1, LambdaIndex2,LambdaIndex3,LambdaIndex4) = tempResBest(dataIndex, 1);
                    NMI(LambdaIndex1, LambdaIndex2,LambdaIndex3,LambdaIndex4) = tempResBest(dataIndex, 2);
                    Purity(LambdaIndex1, LambdaIndex2,LambdaIndex3,LambdaIndex4) = tempResBest(dataIndex, 3);
                    save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-b1=', num2str(lambda1),'-m=',num2str(m),'-l2=', num2str(lambda2),'-f2=', num2str(lambda3) ,'-p=', num2str(p), ...
                        '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd');
                    for tempIndex = 1 : 8
                        if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                            ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                            ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                        end
                    end   
                 end
              end
             end
            end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2,char('CAbCF5_3_') char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC(:))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'PResBest', 'PResStd','iter','TempBeta','TempLambda','TempPhi','TempP','m_list','dim');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end
