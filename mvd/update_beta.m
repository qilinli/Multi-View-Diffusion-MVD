function beta_new = update_beta(A, W, q, para)

N = size(A, 1);       % number of data points
M = size(para.beta, 1);   % number of views

% Calculate Q
Q = zeros(M, M);
for i = 1:M
    for j = 1:M
        Q(i, j) = sum(sum(W{i} .* W{j}));
    end
end

% Calculate p
p = zeros(M, 1);
AW = cellfun(@(x)( A .* x ), W, 'UniformOutput', false);
for i = 1:M
    p(i, 1)=sum(sum(AW{i}));
end

% H and f for 1/2 * x'Hx + f'x
H = double(para.lambda * eye(M) + 2 * para.mu * Q);
f = double(q/norm(q) - 2 * para.mu * p/norm(p));


% Aeq and Beq for the constrain Aeq * x = beq, sum(beta)=1
Aeq = ones(1, M);
beq = 1;

% lb and ub for lb <= x <= ub, 0 <= beta <= 1
lb = zeros(M, 1);
ub = ones(M, 1);

% Qudratic programming
options = optimset('Display', 'off');
beta_new = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);
end