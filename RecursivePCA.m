function Ut = RecursivePCA(Lt, U0)

global D rhat t Sig U_temp alpha tau

if t==1
    U_temp = U0;
end

D = [D Lt];
Ut=U0;
if mod(t, alpha)==0;
    
    C = U_temp'*D;
    E = D-U_temp*C;
    [J K] = qr(E,0);
    Mtemp = [Sig, C;
    zeros(alpha, size(Sig,2)), K];
    [Pr, Sig, ~] = svd(Mtemp,0);
    U_temp = [U_temp J]*Pr;
    Ut = U_temp(:,1:rhat);
    D=[];
end

if mod(t,tau)==0;
    U_temp = U_temp(:,1:rhat);
    Sig = Sig(1:rhat,1:rhat);
end





end

