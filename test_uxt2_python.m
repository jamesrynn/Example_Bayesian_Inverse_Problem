
x = [0.2, 0.4, 0.6, 0.8]';
t = 0.05*[1/3, 2/3, 1]';
lam = [5, 4.5, 7, 2, 3.56]';


% x = [1, 2, 3, 4]';
% t = [0, 0, 0]';
% lam = [1, 0, 1, 0, 1]';



U1 = 99*ones(length(x), length(t), length(lam));
for i = 1:length(lam)
    U1(:,:,i) = uxt(x,t,lam(i));
end


U2 = uxt2(x,t,lam);


U3 = 99*ones(length(x), length(t), length(lam));
for i = 1:length(x)
    for j = 1:length(t)
        for k = 1:length(lam)
            U3(i,j,k) = X(x(i))*exp(-lam(k)*pi*pi*t(j));
        end
    end
end


for k = 1:length(lam)
    k
    U1(:,:,k)
    U2(:,:,k)
    U3(:,:,k)
end



function [U2] = uxt2(x,t,lam)

U2 = 99*ones(length(x),length(t),length(lam));

% Spatially dependent solution component, u(x,0)
uX = 3*sin(pi*x) + sin(3*pi*x);

% Time dependent solution component.
uT = exp(-(pi^2)*t*lam');

for i = 1:length(x)
    U2(i,:,:) = uT*uX(i);
end
end


function [ux] = X(x)
    ux = 3*sin(pi*x) + sin(3*pi*x);
end


