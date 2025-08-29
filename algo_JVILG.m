%% sum_v=1^m av^2 (||PvXv + EvGv - AvWvFv'||_F^2 + beta*|Ev-AvZv|_F^2 +lambda |Zv|_1 )+ phi* ||F^||_Sp^p 
%% s.t,  Pv'Pv =I; A'A = Im; Wv >=0;  Fv >=0; 
%%  code by xinxin,  21/07/2024  

function [Index,obj,iter] = algo_JVILG(X, label ,m, beta, lambda, phi,p ,ind)
% labels : ground truth   n *1.
% lambda : the hyper-parameter.
% p      : the parameter of tensor learning
% ind    ： missing index : n *view
% m      : the number of anchors

% Xv     : incomplete data with zero pedding, dv *n  
% A      : dim * m   consistent anchor 
% Wv     : m *k
% Fv     : k *n    
% Pv     : dim * dv   projection matrix

nV = length(X); N = size(X{1},2);
k = length(unique(label));
weight_vector = ones(1,nV)';      % the defult weight_vector of tensor Schatten p-norm
dim =1*k;                         %% projected dimension
missingindex = constructA(ind);

%% ==============Variable Initialization=========%%
for iv = 1:nV
    F{iv} = eye(N, k);
    Y1{iv} = zeros(N, k);      % lagrange multipler for F
    J{iv} = eye(N, k);       % auxiliary variable

    Qa{iv} = zeros(N, k);    % solve A
    Qf1{iv} = zeros(N, k);   % solve F  
    Qj{iv}=  zeros(N, k);    % solve J
    Qe{iv}=  zeros(N, k);    % solve E
    Qp{iv}=  zeros(N, k);    % solve P
    
    di = size(X{iv},1);
    P{iv} = zeros(dim,di);     % projection
   
    existInd{iv} = find(missingindex{iv});    %% the index of existed views 
    missInd{iv} = find(1-missingindex{iv});   %% the index of missing views    
    numMiss{iv} = N - sum(missingindex{iv});  %% the number of missing
    Z{iv} =  zeros(m,numMiss{iv});
    Ev{iv} = zeros(dim,numMiss{iv});    
    W{iv} = zeros(m, k);
    A{iv} = zeros(dim,m);
end

alpha = ones(1,nV)/nV;

%% TBGL  xiawei TPAMI 2023
% disp('--------------Anchor Selection and Bipartite graph Construction----------');
% tic;
% opt1. style = 1;
% opt1. IterMax =50;
% opt1. toy = 0;
% [~, B] = My_Bipartite_Con(X,cls_num,0.5, opt1,10);
% toc;

%% =====================  Initialization =====================  code from TPAMI2023 Xiawei  TBGL_MVC
sX = [N, k, nV];   
Isconverg = 0; iter = 1;
rho = 1e-5; max_rho = 10e12; pho_rho =2;   % pho_rho : refer to AAAI 2024, xiewei TLL-AG
Pstops = 10e-5;

%% =====================Optimization=====================
while(Isconverg == 0)
    %% solve Av
     part1 = 0;
     for iv =1:nV    
         EE = zeros(dim,N);  
         EE(:,missInd{iv}) = Ev{iv};
         Qa{iv} = P{iv}*X{iv} + EE;
         part1 =  Qa{iv} * F{iv}*W{iv}' + beta * Ev{iv}*Z{iv}';
         [Unew,~,Vnew] = svd(part1,'econ');
         A{iv} = Unew*Vnew';
     end
            
    %% solve Wv %% W 每列和为1
     for iv =1:nV
         part1 = A{iv}'* Qa{iv}*F{iv};
         part2 = F{iv}'*F{iv};
         tempW = part1 * pinv(part2);  % 
         for ii= 1: k
            ut = tempW(:,ii);
            W{iv}(:,ii) = EProjSimplex_new(ut');  
         end
     end
             
    %% solve Fv : k
    for iv =1:nV
        Qf1{iv} = J{iv} - Y1{iv}/rho;
        F1 = alpha(iv)^2 * W{iv}'*W{iv} + 0.5*rho* eye(k);
        F2 = alpha(iv)^2* Qa{iv}'* A{iv} * W{iv} + 0.5* rho * Qf1{iv};    
        tempF = F2*pinv(F1);       
        F{iv} = max(tempF,0);
    end
    
    %%  solve J{v}
    for iv =1:nV
        Qj{iv}=(F{iv} + Y1{iv}/rho);
    end
    Q_tensor = cat(3,Qj{:,:});
    Qg = Q_tensor(:);
    [myj, ~] = wshrinkObj_weight_lp(Qg, phi * weight_vector./rho,sX, 0,3,p);
    J_tensor = reshape(myj, sX);
    for iv=1:nV
        J{iv} = J_tensor(:,:,iv);
    end
  
    %% solve Ev{iv}
    for iv = 1:nV
        Qe{iv} = A{iv} * W{iv}*F{iv}'- P{iv}* X{iv};
        tempE1 = Qe{iv}(:,missInd{iv});
        partE =  tempE1 + beta*A{iv}*Z{iv};
        Ev{iv} = partE / (1+ beta);
    end  
    
    %% solve Zv{iv}  %% 不考虑 normalization (Zv1=1, Zv>=0)
    for iv = 1:nV
        eps1 = 0.5* lambda / beta;
        Z{iv} = prox_l1( A{iv}'* Ev{iv}, eps1);  %% code from AAAI2023, ETLSRR
    end
   
    %% solve P{v}    
    for iv =1: nV
        EE = zeros(dim,N);   
        EE(:,missInd{iv}) = Ev{iv};
        Qp{iv} = A{iv} *W{iv}*F{iv}' - EE;
        partP = Qp{iv} * X{iv}'; 
        [U,~,V] = svd(partP,'econ');     
        P{iv} = U * V';
    end
  
    %% solve av
    M = zeros(nV,1);
    for iv = 1:nV
        M(iv) = norm( P{iv}*X{iv} - Qp{iv},'fro')^2 + beta * norm( Ev{iv} - A{iv}*Z{iv},'fro')^2 + lambda * sum(sum(abs(Z{iv})));
    end
    Mfra = M.^-1;
    Qalpha = 1/sum(Mfra);
    alpha = Qalpha * Mfra;
    
    %% solve Y and  penalty parameters
    for iv=1:nV
        Y1{iv} = Y1{iv} + rho*(F{iv}-J{iv});
    end
    rho = min(rho*pho_rho, max_rho);

    %% compute loss value
    term1 = 0;
    for iv = 1:nV
        term1 = term1 + alpha(iv)^2 * M(iv);
        res1(iv) = norm(F{iv}- J{iv}, inf );
    end
    obj(iter) = term1;
    res_max1(iter) = max(res1);
    
%     fprintf('p :%d, loss: %f \n ',iter,obj(iter));     
%     Vw = norm(W{1},'fro')^2;
%     Va = norm(A{1},'fro')^2;
%     Vf = norm(F{1},'fro')^2;
%     Px = norm(P{1}*X{1},'fro')^2; 
%     Ve = norm(Ev{1},'fro')^2; 
%     Vz = norm(Z{1},'fro')^2; 
%     fprintf('W :%f ,A: %f, F:%f, PX:%f, E:%f, Z:%f \n ',Vw, Va, Vf,Px,Ve,Vz);
    
    %% ==============Max Training Epoc==============%%   %% original stop iter=100
    if (iter>10 && res_max1(iter)< Pstops)  || iter > 50   
        Isconverg  = 1;
        SumF = 0;
        Sa = 0;
        for iv = 1: nV
            SumF = SumF + alpha(iv)^2 *F{iv};
            Sa = Sa + alpha(iv)^2;
        end
        SumF = SumF/Sa;
       [~,Index] = max(SumF,[],2);
       res = Clustering8Measure(label, Index); %%result = [ACC nmi Purity Fscore Precision Recall AR Entropy];
       fprintf('p :%d, ACC: %f,NMI: %f, Purity:%f,iter:%d  ',p , res(1),res(2),res(3),iter);   
    end
    iter = iter + 1;
end
