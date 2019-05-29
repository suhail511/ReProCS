function [C,S,mu,O] = robust_PCA(X,lambda_1,max_iter,q,O,S,mu,C)

% Obtain N,p
[N,p] = size(X);

% Initialize
Xo = zeros(N,p);

for i=1:max_iter
    Xo = X-ones(N,1)*mu'-O;
    % Update S
    S = Xo*C;
    % Update C
    [U,Sigma,V] = svd(Xo'*S,'econ');
    C = U*V';
    % Update O
    O = sign(X-ones(N,1)*mu'-S*C').*max(abs(X-ones(N,1)*mu'-S*C')-lambda_1,0);
    % Update mu
    mu = mean(X-O-S*C',1)';
end
