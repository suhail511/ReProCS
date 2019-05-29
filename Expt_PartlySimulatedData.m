
clear all; clc;
addpath Yall1;
addpath inexact_alm_rpca;
addpath inexact_alm_rpca/PROPACK;
addpath RobustSubspaceLearning;
addpath Data;
addpath MG;
load lake_bgnd.mat
numTrain = 1420;
DataTrain = data(:,1:numTrain);  % training sequence
B = data(:,numTrain+1:end);      % test sequence
imSize = [72 90];
p = size(B,1);q = size(B,2);

%% MC 50 times
MCnum = 50;
DataS = cell(MCnum,1);
DataShat_mod_recur = cell(MCnum,1);
DataShat_mod_proj = cell(MCnum,1);
DataShat_grasta = cell(MCnum,1);
DataShat_PCP = cell(MCnum,1);
DataShat_RSL = cell(MCnum,1);
DataShat_MG = cell(MCnum,1);

for MC = 1: MCnum

MC

%% overlay simulated foreground images Ot (one slowing moving rectangular block) on background sequence Lt
I = zeros(p,q);
%intensity = 200;  % foreground intensity
%intensity=unifrnd(170,230*ones(imSize));
pos = zeros(2,q); % location
vp = zeros(2,q);  % velocity
nv = zeros(2,q);  % bounded noise on velocity

t=1;
w = round([44 24]'/2);  %block size is 45 x 25
pos(:,t) = [w(1)+5, w(2)+30];
vp(:,t)=[0.5 0]'; 
%sigmav = [sqrt(0.005) 0]';
sigmav=[sqrt(0.02) 0]';
x_index = round( pos(1,t)-w(1): pos(1,t)+ w(1) );   x_index = x_index (x_index>0 & x_index<imSize(2));
y_index = round( pos(2,t)-w(2): pos(2,t)+ w(2) );   y_index = y_index (y_index>0 & y_index<imSize(1));
O2d =zeros(imSize); 
intensity=unifrnd(170,230*ones(length(y_index), length(x_index)));
O2d( y_index, x_index) = intensity;
F(:,t) = O2d(:);
for t=2:q
        nv(:,t)  = sigmav.*randn(2,1); nv(:,t) = sign(nv(:,t) ).*min(2*sigmav,abs(nv(:,t) ));
        vp(:,t) = vp(:,t-1) + nv(:,t);        
        pos(:,t) = pos(:,t-1) + vp(:,t);        
        x_index = round( pos(1,t)- w(1): pos(1,t)+ w(1) );  x_index = x_index (x_index>0 & x_index<imSize(2));
        y_index = round( pos(2,t)- w(2): pos(2,t)+ w(2) );  y_index = y_index (y_index>0 & y_index<imSize(1)); 
        intensity=unifrnd(170,230*ones(length(y_index), length(x_index)));
        O2d =zeros(imSize); O2d( y_index, x_index) = intensity;
        F(:,t) = O2d(:);
end
clear O2d;

I = zeros(p,q);S= zeros(p,q);N = cell(q,1);
for t=1:q
    N{t}= find(F(:,t)~=0);   set(t) = numel(N{t});
    I(N{t},t) = F(N{t},t);   I(setdiff(1:p,N{t}),t) = B(setdiff(1:p,N{t}),t);  %image overlay
    S(:,t) = I(:,t) - B(:,t);
end

DataS{MC}=S; 
clear N,t;
%% MG
Data=[DataTrain I];

sizeim=[72 90];

% Data generation
[p1,N] = size(Data);
q1 = 47;

lambda_1 = 9.6957e-04;
% max_iter = 10;
max_iter = 50;

% Grid of K1 values of \lambda_1
K1 = 1; % We already found a good lambda (above), so no need to search
%lambda_1 = logspace(log10(1e-8),log10(1e-2),K1);


% Generate data matrix X
X = Data';
clear Data

% Initialization
O_0 = zeros(N,p1);
S_0 = zeros(N,q1);
mu_0 = mean(X,1)';
temp = speye(p1);
C_0 = temp(:,1:q1); % First q columns of the identity matrix
clear temp

% Batch PCA (classic PCA, lambda is infinity so no outliers)
[C_hat_nr,S_hat_nr,mu_hat_nr,O_hat_nr] = robust_PCA(X,inf,max_iter,q1,O_0,S_0,mu_0,C_0);

% Initialize
errors = zeros(K1,1);
S_hat_r = S_hat_nr;
mu_hat_r = mu_hat_nr;
C_hat_r = C_hat_nr;
% Batch Robust PCA
for i=K1:-1:1
    [C_hat_r,S_hat_r,mu_hat_r,O_hat_r] = robust_PCA(X,lambda_1(i),max_iter,q1,O_0,S_hat_r,mu_hat_r,C_hat_r);
    errors(i) = (1/N)*norm(X-ones(N,1)*mu_hat_r'-S_hat_r*C_hat_r'-O_hat_r,'fro')^2;
end






refiter = 10;
% Parameter \delta
delta = 1e-6;
% Initialize based on latest solution
O_hat_ref = O_hat_r;
C_hat_ref = C_hat_r;
S_hat_ref = S_hat_r;
mu_hat_ref = mu_hat_r;


for i=1:refiter
    % Compute weighted \lambda values
    lambdaweighted = lambda_1./(delta+abs(O_hat_ref));
    % Solve robust PCA problem for given values of \lambdaweight
    [C_hat_ref,S_hat_ref,mu_hat_ref,O_hat_ref] = robust_PCA(X,lambda_1,1,q1,O_hat_ref,S_hat_ref,mu_hat_ref,C_hat_ref);
end


O=O_hat_ref';
O=O(:,1421:1500);
DataShat_MG{MC}=O;


%% ReProCS-proj

mu0 = mean(DataTrain,2);
numTrain=size(DataTrain,2);
MTrain=DataTrain-repmat(mu0,1,numTrain);


M=I-repmat(mu0,1, size(I,2)); %subtract the mean
[U, Sig, ~] = svd(1/sqrt(numTrain)*MTrain,0);
evals = diag(Sig).^2;
energy = sum(evals);
cum_evals = cumsum(evals);
ind0 = find(cum_evals < 0.95*energy);
rhat = length(ind0);
lambda_min=evals(rhat);
U0 = U(:, ind0); 

Shat_mod = zeros(p,q); 
Lhat_mod = zeros(p,q); 
Nhat_mod=cell(q,1); 
Fg = zeros(p,q);
xcheck=zeros(p,q);
Tpred = [];
Ut = U0; 

Pstar = Ut;

k=0;K = [];
Kmin=3;
Kmax=10;
addition =0;cnew=[];t_new=[];time_upd = [];
thresh_diff=[]; thresh_base = [];
alpha=20;

for t =1: q
  
clear opts; opts.tol = 1e-3; opts.print = 0;

Atf.times = @(x) Projection(Ut,x); Atf.trans = @(y) Projection(Ut,y);
yt = Atf.times(M(:,t));

% decide noise
if t==1
        opts.delta = norm(Atf.times(MTrain(:,numTrain)));
    else
        opts.delta = norm(Atf.times(Lhat_mod(:, t-1)));
end



if t==1||t==2
    [xp,~] = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta
    omega(t)=sqrt(M(:,t)'*M(:,t)/p);
    That=find(abs(xp)>=omega(t));

else

        if length(Nhat_mod{t-2})==0
            thresh(t)=0;
        else
        thresh(t)=length(intersect(Nhat_mod{t-1}, Nhat_mod{t-2}))/length(Nhat_mod{t-2});
        lambda1=length(setdiff(Nhat_mod{t-2}, Nhat_mod{t-1}))/length(Nhat_mod{t-1});
        end
        
    if thresh(t)<0.5
        [xp,~] = yall1(Atf, yt, opts); 
        omega(t)=sqrt(M(:,t)'*M(:,t)/p);
        That=find(abs(xp)>=omega(t));
    else
        weights= ones(p,1); weights(Tpred)=lambda1;
        opts.weights = weights(:);
   
        [xp,flag] = yall1(Atf, yt, opts); 
        [xp,ind]=sort(abs(xp),'descend');
        Tcheck=ind(1:round(min(1.4*length(Tpred),0.6*p)));
        xcheck(Tcheck,t)=subLS(Ut, Tcheck, yt);
        omega(t)=sqrt(M(:,t)'*M(:,t)/p);
        That=find(abs(xcheck(:,t))>=omega(t));
    end
end

    Shat_mod(That,t) = subLS(Ut,That,yt);     
    Lhat_mod(:,t) = M(:,t) - Shat_mod(:,t);
    %ShatNorm(t)=norm(Shat_mod(:,t));
    %LhatNorm(t)=norm(Lhat_mod(:,t));
    
    Fg(That,t)=I(That,t);
    Nhat_mod{t}=That;
    Tpred = That;
    Bg(:,t) = Lhat_mod(:,t) + mu0;
    %Ut = recursivePCA(Lhat_mod(:,t), Ut);
    
    
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
%                if  numel(Lambda_new)==1
%                    cnew_hat = 1;
%                else
%                    [~, cnew_hat] = max(abs(diff(Lambda_new))./Lambda_new(1:end-1));% max(abs(diff(log(Lambda_new))));
%                end  
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
                   %abs(thresh_diff(end)-thresh_diff(end-1))/thresh_diff(end-1)<0.01)        
                   addition =0;
                   K = [K k];
                   Pstar = Ut;            
               end
           end
    end
    
end
  DataShat_mod_proj{MC}=Shat_mod; 

  
%% ReProCS_recur
clear rhat t Sig U_temp alpha tau D;
global rhat t Sig U_temp alpha tau D
D = [];

mu0         = mean(DataTrain,2);
numTrain    = size(DataTrain,2);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
M           = I-repmat(mu0,1, size(I,2)); %subtract the mean
[U, Sig, ~] = svd(1/sqrt(numTrain)*MTrain,0);

evals       = diag(Sig).^2;
energy      = sum(evals);
cum_evals   = cumsum(evals);
ind0        = find(cum_evals < 0.95*energy);
rhat        = min(length(ind0),round(numTrain/10));
Sig         = Sig(1:rhat,1:rhat);
alpha       = 20;
tau         = 3*rhat;
U0          = U(:, 1:rhat); 



Shat_mod    = zeros(p,q); 
Lhat_mod    = zeros(p,q); 
Nhat_mod    = cell(q,1); 
Fg          = zeros(p,q);
xcheck      = zeros(p,q);
Tpred       = [];
Ut          = U0; 


for t = 1: q
  
clear opts; 
opts.tol   = 1e-3; 
opts.print = 0;
Atf.times  = @(x) Projection(Ut,x); Atf.trans = @(y) Projection(Ut,y);
yt         = Atf.times(M(:,t));

% decide noise
if t==1
        opts.delta = norm(Atf.times(MTrain(:,numTrain)));
    else
        opts.delta = norm(Atf.times(Lhat_mod(:, t-1)));
end

if t==1||t==2
    [xp,~]   = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta
    omega(t) = sqrt(M(:,t)'*M(:,t)/p);
    That     = find(abs(xp)>=omega(t));
else
        if length(Nhat_mod{t-2})==0
            thresh(t)=0;
        else
            thresh(t)=length(intersect(Nhat_mod{t-1}, Nhat_mod{t-2}))/length(Nhat_mod{t-2});
            lambda1=length(setdiff(Nhat_mod{t-2}, Nhat_mod{t-1}))/length(Nhat_mod{t-1});
        end
        
    if thresh(t)<0.5
        [xp,~]   = yall1(Atf, yt, opts); 
        omega(t) = sqrt(M(:,t)'*M(:,t)/p);
        That=find(abs(xp)>=omega(t));
    else
        weights         = ones(p,1); 
        weights(Tpred)  = lambda1;
        opts.weights    = weights(:);
        [xp,flag]       = yall1(Atf, yt, opts); 
        [xp,ind]        = sort(abs(xp),'descend');
        Tcheck          = ind(1:round(min(1.4*length(Tpred),0.6*p)));
        xcheck(Tcheck,t)= subLS(Ut, Tcheck, yt);
        omega(t)        = sqrt(M(:,t)'*M(:,t)/p);
        That            = find(abs(xcheck(:,t))>=omega(t));
    end
end

    Shat_mod(That,t)    = subLS(Ut,That,yt);     
    Lhat_mod(:,t)       = M(:,t) - Shat_mod(:,t);
    Fg(That,t)          = I(That,t);
    Nhat_mod{t}         = That;
    Tpred               = That;
    Bg(:,t)             = Lhat_mod(:,t) + mu0;
    Ut                  = RecursivePCA(Lhat_mod(:,t), Ut);
end

DataShat_mod_recur{MC}=Shat_mod; 

%% Grasta
   addpath grasta
   %% GRASTA parameters
OPTIONS.RANK                = 5;  % the estimated low-rank
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
                                    
max_cycles                  = 10;    % training cycles
training_size               = 1420;   % random chose 50 frames as the training set
TRAINING_SAMPLING           = 0.3;   % Use how much information to train the first subspace.


[bgU, status, OPTS]  = bgtraining(DataTrain, imSize, OPTIONS, max_cycles, TRAINING_SAMPLING, training_size,FPS_ONLY,TRAIN_FRAME);



%% Make video -- grasta
OPTS.MAX_ITER               = 20;
OPTIONS.CONSTANT_STEP       = 1e-2; % use the constant step-size
FPS_ONLY                    = 1;    % if you want to measure the FPS performance, please let FPS_ONLY=1
SAMPLING                    = 1;  % Use how much information to track the subspace.
thresh                      = 0.2;
MAX_FRAME                   = -1;   % -1 means seperating all the frames
OPTIONS.USE_MEX             = 0;
fprintf('Seperating the whole video sequence by grasta...\n');
[video_grasta_shat,video_grasta_bg, vInfo] = bgfg_seperation_grasta( I, imSize, bgU,  SAMPLING ,status,OPTIONS, OPTS,thresh,FPS_ONLY, MAX_FRAME);

DataShat_grasta{MC}=video_grasta_shat;


%% offline PCP
M=I;
lambda = 1/sqrt(max(size([DataTrain M])));
[Lhat_PCP, Shat_PCP, ~] = inexact_alm_rpca([DataTrain M], lambda);  %large scale problems
Lhat_PCP = Lhat_PCP(:,numTrain+1:end);
Shat_PCP = Shat_PCP(:,numTrain+1:end);

% Ohat_PCP = zeros(size(M));
% for t=1:q 
%     shat = sort(abs(Shat_PCP(:,t)),'descend');
%     tot_egy = shat'*shat;
%     k = 1;
%     while shat(1:k)'*shat(1:k)<=0.9*tot_egy
%         k = k+1;
%     end
%     alpha_obj = shat(k);
%     That = find(abs(Shat_PCP(:,t))>alpha_obj);
%     Ohat_PCP(That,t) = M(That,t);
% end

DataShat_PCP{MC} = Shat_PCP;
%% offline RSL
Data = [DataTrain M];
RobSubLearn;
Lhat_RSL = Lhat_RSL(:,numTrain+1:end);
Shat_RSL = M - Lhat_RSL;

% Ohat_RSL = zeros(size(M));
% for t=1:q 
%     shat = sort(abs(Shat_RSL(:,t)),'descend');
%     tot_egy = shat'*shat;
%     k = 1;
%     while shat(1:k)'*shat(1:k)<=0.9*tot_egy
%         k = k+1;
%     end
%     alpha_obj = shat(k);
%     That = find(abs(Shat_RSL(:,t))>alpha_obj);
%     Ohat_RSL(That,t) = M(That,t);
% end
DataShat_RSL{MC}=Shat_RSL;

end

save('DataS.mat','DataS');
save('DataShat_mod_proj.mat','DataShat_mod_proj');
save('DataShat_mod_recur.mat','DataShat_mod_recur');
save('DataShat_PCP.mat','DataShat_PCP');
save('DataShat_RSL.mat','DataShat_RSL');
save('DataShat_grasta.mat','DataShat_grasta');
save('DataShat_MG.mat','DataShat_MG');