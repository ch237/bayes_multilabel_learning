clear;
rng('default');
load('bibtex.mat');
X_tr = X_tr';
Y_tr = full(Y_tr');
X_te = X_te';
Y_te = full(Y_te');
[L N] = size(Y_tr);
N_te = size(X_te,2);
D = size(X_tr,1);

[a b] = find(Y_tr==1);
id{1,1} = a;
id{1,2} = b;
clear a; clear b;

PolyaGammaTruncation = 1000; 
IsMexOK = false;

% parameter settings
K = 32; % embedding dimension (# of topics)
eta = 1;
r = 0.5*ones(K,1); % negative binomial overdispersion hyperparameter
for k=1:K
    sigma2{k} = 0.1; % variance of the gaussian prior on W
end
a0 = 1e-6;
b0 = 1e-6;
lambda = ones(1,K); % doesn't matter!!

numIters = 100;

% initialize
[W S V] = svds(X_tr,K);
clear S;
clear V;

for k=1:K
    V(:,k) = sampleDirMat(eta*ones(1,L),1);
    wtx = W(:,k)'*X_tr;
    U(k,:) = r(k)*ones(1,N).*(1./(1+exp(-wtx)));
end

tic;
for iter=1:numIters
    VU{1,1} = V;
    VU{1,2} = U';
    zeta_lnk=unormalzetair(VU,id,lambda);
    zeta_ln = sum(zeta_lnk,2);
    M_ln=zeta_ln.*exp(zeta_ln)./(exp(zeta_ln)-1);
    M_lnk=repmat(M_ln,1,K).*(zeta_lnk./repmat(zeta_ln,1,K));
    [S1,S2]=tensorsum(M_lnk,id,[L N]); 
    
    for k=1:K
        wtx = W(:,k)'*X_tr;
        U(k,:) = (r(k)+S1{2}(:,k)').*(1./(1+exp(-wtx)));
        V(:,k) = (eta+S1{1}(:,k)-1)/(sum(S1{1}(:,k))+L*(eta-1));
        psi = X_tr'*W(:,k);
        phi = (0.5*(S1{2}(:,k)+r(k))./psi).*tanh(psi/2);
        d = 0.5*X_tr*(S1{2}(:,k) - r(k)); 
        % use CG to solve for W
        [W(:,k), flags] = cgs(@(x)afun(x,X_tr,phi,sigma2{k}),d,[],10,[],[],W(:,k));  
    end
    
    if iter==1 
        time_trace(iter) = toc;
        tic;
    else
        time_trace(iter) = time_trace(iter-1) + toc;
        tic;
    end     
    
    lam_tr = V*U;
    pr_ytr = 1 - exp(-lam_tr);
    auc_tr = compute_AUC(Y_tr(:),pr_ytr(:),ones(size(Y_tr(:))));
    pred_ytr = double(round(pr_ytr));
    hamm_tr = mean(mean(Y_tr~=pred_ytr));  

    % predict labels for test data
    tmp = 1;
    for k=1:K
        tmp = tmp./((V(:,k)*(exp(W(:,k)'*X_te))+1).^r(k));
    end
    pr_yte = 1 - tmp;  
    pred_yte = double(round(pr_yte));
    hamm_te = mean(mean(Y_te~=pred_yte)); 

    auc_te = compute_AUC(Y_te(:),pr_yte(:),ones(size(Y_te(:))));
    auc1(iter) = auc_tr;
    auc2(iter) = auc_te;

    fprintf('Iteration %d, Train AUC = %f, Train Hamming = %f, Test AUC = %f, Test Hamming = %f, norm(W) = %f\n',iter,auc_tr,hamm_tr,auc_te,hamm_te,sum(sum(W.^2))/(D*K)); 
    auc_record(iter) = auc_te;
end