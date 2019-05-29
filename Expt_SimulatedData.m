clear all;
rand('state',0);randn('state',0);
addpath proximal_gradient_rpca;
addpath RobustSubspaceLearning;
addpath Yall1;

p          = 100;              %Mt, Lt and St are p-dimensioanl vectors
numTrain   = 2000;             %length of training sequence: t0 = 2000.
q = 100;                       %sequence lenght.

num_block  = 1;                %Tt = supp(St). |Tt| =num_block*block_size, 
block_size = 27; 
magnitude  = 100;               %(S_t)_i = 10, for all i in Tt.

gamma_res  = 0.5*magnitude;    % threshold on (I-PtPt')Mt used by adapted-iSVD and iRSL to detect outliers

for mc=1:100
    
    % 100 times monte carlo averaging

    pos      = zeros(num_block,1);
    
for item     = 1:num_block    
    pos_temp = randi([(block_size-1)/2,p-(block_size-1)/2],1,1);
   % make sure initial location of each strip is distinct 
    while (ismember(pos_temp,pos,'rows')==1)
           pos_temp = randi([(block_size-1)/2,p-(block_size-1)/2],1,1);
    end    
    pos(item,:)     = pos_temp;
end  
vp       = {0 1 -1};
motion_p = 0.8;
S        = zeros(p,q); 
for t=1:q
    for item = 1:num_block; 
        pro  = rand(1);
        if pro<=motion_p
                velocity = vp{1};
        elseif pro<=motion_p+ (1-motion_p)/2
                velocity = vp{2};
        else    velocity = vp{3};
        end
    pos(item)    = pos(item) + velocity; 
    x_index      = round( pos(item)- (block_size-1)/2: pos(item) + (block_size-1)/2 );  x_index = x_index (x_index>0 & x_index<p);
    S(x_index,t) = magnitude;
    end
end  

% generate low rank part
r        = 0.2*p; 
r_delta  = 2;
sigma    = zeros(r,1);
sigma(1) =  1e4;
ratio    = (10/sigma(1))^(1/r);
for iter = 2:r
    sigma(iter) = ratio*sigma(iter-1);
end

sigma(r+1:r+r_delta) = [50 60];
Sigma = diag(sigma);

U = gs(randn(p,r+r_delta));
f = 0.5; fd = 0.1; theta = 0.5; 
% training sequence of length t0 = numTrain
x = zeros(r+r_delta,numTrain);
t = 1;
Et= 1:r;
x(Et,t) = sqrt(Sigma(Et,Et))*randn(length(Et),1);
for t      = 2:numTrain 
   x(Et,t) = f*x(Et,t-1) + sqrt(Sigma(Et,Et)./(1-f^2))*randn(length(Et),1);
end
DataTrain  = U*x; clear x;
% generate Lt
AdditionSet    = cell(1,q); DeletionSet = cell(1,q); 
AdditionSet{5} = [r+1:r+2];  %add two new PCs at t=t0+5
DeletionSet{5} = [r-2 r-1];  %two old PCs start to vanish at t=t0+5
x              = zeros(r+r_delta,q);

t  = 1;
Et = 1:r;
Delta_t = AdditionSet{t};
Dt      = DeletionSet{t};
Nt      = union(union(Delta_t,Dt),Et);
x(Et,t) = sqrt(Sigma(Et,Et))*randn(r,1);
for t = 2:q  
    Delta_t = AdditionSet{t}; 
    Dt      = union (Dt, DeletionSet{t});
    Et      = setdiff(Nt,Dt);  
    Nt      = union(union(Delta_t,Dt),Et); 
    
    x(Delta_t,t) = sqrt(theta*Sigma(Delta_t,Delta_t))*randn(length(Delta_t),1);    
    x(Et,t)      = f*x(Et,t-1) + sqrt((1-f^2)*Sigma(Et,Et))*randn(length(Et),1);
    x(Dt,t)      = fd*x(Dt,t-1); 
end

L = U*x; clear x

M = L + S;
O = zeros(size(M));
for t = 1:q
    T = find(abs(S(:,t))~=0);
    O(T,t)   = M(T,t);  %foreground
    EO(mc,t) = norm(O(:,t))^2;
    EL(mc,t) = norm(L(:,t))^2;
    ES(mc,t) = norm(S(:,t))^2;
end

EgyO(mc) = norm(O,'fro').^2;
EgyL(mc) = norm(L,'fro').^2;
EgyS(mc) = norm(S,'fro').^2;

%%PracReProCS
numTrain    = size(DataTrain,2);
[U, Sig, ~] = svd(1/sqrt(numTrain)*DataTrain,0);
evals       = diag(Sig).^2;
energy      = sum(evals);
cum_evals   = cumsum(evals);
ind0        = find(cum_evals < 0.9999*energy);
rhat        = length(ind0);
lambda_min  = evals(rhat);
U0          = U(:, 1:rhat); 


Shat_mod = zeros(p,q); 
Lhat_mod = zeros(p,q); 
Nhat_mod = cell(q,1); 
Fg       = zeros(p,q);
xcheck   = zeros(p,q);
Tpred    = [];
Ut       = U0; 
Pstar    = Ut;
k    = 0;
K    = [];
Kmin = 3;
Kmax = 10;
addition = 0; cnew = []; t_new = []; time_upd = [];
thresh_diff=[]; thresh_base = [];
alpha=20;
for t =1: q

clear opts; 
opts.tol = 1e-3; 
opts.print = 0;
Atf.times = @(x) Projection(Ut,x); Atf.trans = @(y) Projection(Ut,y);
yt = Atf.times(M(:,t));
% decide noise
if t==1
        opts.delta = norm(Atf.times(DataTrain(:,numTrain)));
    else
        opts.delta = norm(Atf.times(Lhat_mod(:, t-1)));
end
if t==1||t==2
    [xp,~]   = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta
    omega(t) = 0.25*sqrt(M(:,t)'*M(:,t)/p);
    That     = find(abs(xp)>=omega(t));
else
        if length(Nhat_mod{t-2})==0
            thresh(t)=0;
        else
        thresh(t) = length(intersect(Nhat_mod{t-1}, Nhat_mod{t-2}))/length(Nhat_mod{t-2});
        lambda1   = length(setdiff(Nhat_mod{t-2}, Nhat_mod{t-1}))/length(Nhat_mod{t-1});
        end
        
    if thresh(t)<0.5
        [xp,~]    = yall1(Atf, yt, opts); 
        omega(t)  = 0.25*sqrt(M(:,t)'*M(:,t)/p);
        That      = find(abs(xp)>=omega(t));
    else
        weights   = ones(p,1); weights(Tpred)=lambda1;
        opts.weights = weights(:);
   
        [xp,flag] = yall1(Atf, yt, opts); 
        [xp,ind]  = sort(abs(xp),'descend');
        Tcheck    = ind(1:round(min(1.4*length(Tpred),0.6*p)));
        xcheck(Tcheck,t)=subLS(Ut, Tcheck, yt);
        omega(t)=0.25*sqrt(M(:,t)'*M(:,t)/p);
        That=find(abs(xcheck(:,t))>=omega(t));
    end
end

    Shat_mod(That,t) = subLS(Ut,That,yt);     
    Lhat_mod(:,t) = M(:,t) - Shat_mod(:,t);
    Fg(That,t)=M(That,t);
    Nhat_mod{t}=That;
    Tpred = That;
    %projection-PCA
    if addition==0    %&& norm( (Lhat(:,t-alpha+1:t) - Phat*(Phat'*Lhat(:,t-alpha+1:t)))./sqrt(alpha) )>thresh
        addition=1;
        t_new = t;
        Pnewhat=[];
        k=0;
    end
        
    if addition==1&&mod(t-t_new+1,alpha)==0
        time_upd = [time_upd,t];           
        D= Lhat_mod(:,t-alpha+1:t)-Pstar*(Pstar'*Lhat_mod(:,t-alpha+1:t)); 
       
        [Pnew_hat,Lambda_new,~] = svd(D./sqrt(alpha),0);
      
        Lambda_new = diag(Lambda_new).^2;
        Lambda_new = Lambda_new(Lambda_new>=lambda_min);
        th=round(rhat/3);
        if size(Lambda_new,1)> th
            Lambda_new=Lambda_new(1:th);
        end
           if numel(Lambda_new)==0
               addition = 0; 
               cnew = [cnew 0];
           else              

               cnew_hat = numel(Lambda_new);
               Pnewhat_old = Pnewhat;
               Pnewhat = Pnew_hat(:,1:cnew_hat); cnew = [cnew cnew_hat];%size(Pnewhat,2)];
               Ut = [Pstar Pnewhat];   
               
               k=k+1;
               
               if k==1 
                   temp=(Pnewhat*(Pnewhat'*Lhat_mod(:,t-alpha+1:t)));
                   thresh_base = [thresh_base norm(temp./sqrt(alpha))];
                   thresh_diff = [thresh_diff norm(temp./sqrt(alpha))];                 
               else
                   temp=(Pnewhat*(Pnewhat'*Lhat_mod(:,t-alpha+1:t)));
                   thresh_base = [thresh_base norm(temp./sqrt(alpha))];
                   
                   temp = (Pnewhat*(Pnewhat'*Lhat_mod(:,t-alpha+1:t)) - Pnewhat_old*(Pnewhat_old'*Lhat_mod(:,t-alpha+1:t)));
                   thresh_diff = [thresh_diff norm(temp./sqrt(alpha))];  
               end
               
               flag = 0;
               if k >= Kmin
                   numK = 3;
                   flag = thresh_diff(end)/thresh_base(end-1)<0.01;
                   for ik = 1:numK-1
                       flag = flag && thresh_diff(end-ik)/thresh_base(end-ik-1)<0.01;
                   end
               end
               
               if  k==Kmax|| (k>=Kmin && flag==1)                  
                   addition =0;
                   K = [K k];
                   Pstar = Ut;            
               end
           end
    end
end

ErrorL_CS(mc) = norm(L-Lhat_mod,'fro').^2;
ErrorS_CS(mc) = norm(S-Shat_mod,'fro').^2;
ErrorO_CS(mc) = norm(O-Fg,'fro').^2;

%% RSL 
Data = [DataTrain M]; imSize=[p 1];
RobSubLearn;
Lhat_RSL = Lhat_RSL(:,numTrain+1:end);
Shat_RSL = M - Lhat_RSL;
Ohat_RSL = zeros(size(M)); 
for t=1:q 
    That = find(abs(Shat_RSL(:,t))>gamma_res);
    Ohat_RSL(That,t) = M(That,t);   
end
ErrorS_RSL(mc) = norm(S-Shat_RSL,'fro')^2;

%% PCP
lambda = 1/sqrt(max(size([DataTrain M])));
[Lhat_PCP Shat_PCP] = proximal_gradient_rpca([DataTrain M], lambda); 
Lhat_PCP = Lhat_PCP(:,numTrain+1:end);
Shat_PCP = Shat_PCP(:,numTrain+1:end);
Ohat_PCP = zeros(size(M));
for t=1:q 
    That = find(abs(Shat_PCP(:,t))>gamma_res);
    Ohat_PCP(That,t) = M(That,t);   
end
ErrorS_PCP(mc) = norm(S-Shat_PCP,'fro').^2;

 %% Grasta
   addpath grasta
   %% GRASTA parameters
OPTIONS.RANK                = rhat;  % the estimated low-rank
OPTIONS.rho                 = 1.8;    
OPTIONS.ITER_MAX            = 20; 
OPTIONS.ITER_MIN            = 20;    % the min iteration allowed for ADMM at the beginning

OPTIONS.USE_MEX             = 0;     % If you do not have the mex-version of Alg 2
                                     % please set Use_mex = 0.                                     

%% Initialize the rough subspace
OPTIONS.CONSTANT_STEP       = 0;   % use adaptive step-size to initialize the subspace
OPTIONS.MAX_LEVEL           = 20;
OPTIONS.MAX_MU              = 10000; % set max_mu large enough for initial subspace training
OPTIONS.MIN_MU              = 1;
FPS_ONLY                    = 1;    % 0:show the training video, 1: suppress the video demostration
TRAIN_FRAME                 = 1;    % 0¡Guse the first #training_size frames¡F 
                                    % 1: random select #training_size frames
                                    
max_cycles                  = 10;      % training cycles
training_size               = 500; %100;   % random chose 50 frames as the training set
TRAINING_SAMPLING           = 1;   % Use how much information to train the first subspace.


imSize=[100,1];
[bgU, status, OPTS]  = bgtraining(DataTrain, imSize, OPTIONS, max_cycles, TRAINING_SAMPLING, training_size,FPS_ONLY,TRAIN_FRAME);
%% Make video -- grasta
OPTS.MAX_ITER               = 20;
OPTIONS.CONSTANT_STEP       = 1e-2; % use the constant step-size
FPS_ONLY                    = 1;    % if you want to measure the FPS performance, please let FPS_ONLY=1
SAMPLING                    = 1;  % Use how much information to track the subspace.
thresh                      = 0.2;
MAX_FRAME                   = -1;   % -1 means seperating all the frames
OPTIONS.USE_MEX             = 0;
[video_grasta_shat,video_grasta_bg, vInfo] = bgfg_seperation_grasta( M, imSize, bgU,  SAMPLING ,status,OPTIONS, OPTS,thresh,FPS_ONLY, MAX_FRAME);
ErrorS_grasta(mc) = norm(S-video_grasta_shat,'fro')^2;



 

%% adapted-iSVD

mu0  = 0;
Sig0 = Sig(1:rhat,1:rhat);
Lhat_iSVD = zeros(size(M));Shat_iSVD = zeros(size(M)); Ohat_iSVD = zeros(size(M));
u=U0;s =Sig0;
sig_add = 5;
for t=1:q
    x = M(:,t);
    r = x - u*(u'*x);    % r=(I-UtUt')Mt
    T2 = find(abs(r)> gamma_res);
    T1 = setdiff([1:p],T2);
    U1=u(T1,:);U2=u(T2,:);    
    x(T2) = U2*s*(pinv(U1*s)*x(T1));
    k = norm(x(T1) - U1*s*(pinv(U1*s)*x(T1)));
    if( k> 0.1)
        j = (x - u*(u'*x))./k;
        l = u'*x;
        Sig = [s,                  l ; 
              zeros(1,size(s,2)), k ]; 
        [ur sr ~] = svd(Sig,0);
        Tr = find(diag(sr)> sig_add);
        ur = ur(:,Tr); s = sr(Tr,Tr);     
        u = [u j]*ur;
    end    
    Lhat_iSVD(:,t) = x;
    Shat_iSVD(:,t) = M(:,t) - Lhat_iSVD(:,t);    
    Ohat_iSVD(T2,t)= M(T2,t);
end

ErrorO_iSVD(mc) = norm(O-Ohat_iSVD,'fro').^2;


%% iRSL
Ohat_iRSL = zeros(size(M));
u = U0; lambda = Sig0;
beta= 2.3849; alpha=0.95;
mu = mu0;

for t=1:size(M,2)
    xp = M(:,t);
    sigma = max(abs(u)*sqrt(lambda),[],2);
    c = beta*sigma;
    x = xp- mu;
  
    r = x- u*(u'*x);      % r=(I-UtUt')Mt
    w = 1./(1+(r./c).^2); % weight each data point according to its reliability
    z = sqrt(w).*x;
    
    mu = mu+(1-alpha)*z;
    
    y1 = u*sqrt(alpha*lambda);
    y2 = sqrt(1-alpha)*z;
    A = [y1 y2];
    B = A'*A;
    [V,D]=eig(B);    
    u = A*V;
  
    [lambda T] = sort(diag(D),'descend');
    lambda = diag(lambda);
    u = u(:,T);
    for k = 1:size(u,2)
       u(:,k) = u(:,k)/norm(u(:,k));
    end
    
    T1 = find(abs(r)>gamma_res);  %detect outliers by thresholding on (I-UtUt')Mt
    Ohat_iRSL(T1,t) = M(T1,t);
end

ErrorO_iRSL(mc) = norm(O-Ohat_iRSL,'fro').^2;

end

E_repro  = mean(ErrorS_CS)/mean(EgyS)
E_RSL    = mean(ErrorS_RSL)/mean(EgyS)
E_PCP    = mean(ErrorS_PCP)/mean(EgyS)
E_grasta = mean(ErrorS_grasta)/mean(EgyS)
E_iSVD   = mean(ErrorO_iSVD)/mean(EgyO)
E_iRSL   = mean(ErrorO_iRSL)/mean(EgyO)
