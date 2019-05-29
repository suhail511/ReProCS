% Han Guo, Chenlu Qiu, Namrata Vaswani 
% Copyright: Iowa State University
% Questions: hanguo@iastate.edu
% Reference: An Online Algorithm for Separating Sparse and Low-dimensional Signal Sequences from their Sum,
%            accepted by IEEE Transaction on Signal Processing



clear all; clc;
addpath Yall1;
addpath Data;

load Curtain.mat
p = size(I,1);
q = size(I,2);
b = 0.95;

global rhat t Sig U_temp alpha tau D
D = [];

%%%% training
mu0         = mean(DataTrain,2);
numTrain    = size(DataTrain,2);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
M           = I-repmat(mu0,1, size(I,2)); %subtract the mean
[U, Sig, ~] = svd(1/sqrt(numTrain)*MTrain,0);

evals       = diag(Sig).^2;
energy      = sum(evals);
cum_evals   = cumsum(evals);
ind0        = find(cum_evals < b*energy);
rhat        = min(length(ind0),round(numTrain/10));
Sig         = Sig(1:rhat,1:rhat);
alpha       = 20;
tau         = 3*rhat;
U0          = U(:, 1:rhat); 


%% practical-ReProCS

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
   
