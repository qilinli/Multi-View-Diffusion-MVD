function S = preprocessingW(W, self_affinity, symmetric, remove_diagonal)

if (nargin < 2)
    % default nromalization is symmetric
    symmetric = True;
end
if (nargin < 3)
    % default self affinity is 1 rather than d
    self_affinity = False;
end

n = size(W, 1);          %%% number of data points
I = eye(n);              %%%% identity matrix of size n

% Pre-processing of weight matrix W
d = sum(W, 2);
D = diag(d + eps);

if strcmp(self_affinity, '0')
    W = W - diag(diag(W));        %%% No self-affinity
elseif strcmp(self_affinity, 'D') %%% use node degree as self-affinity
    W = W - diag(diag(W)) + D;
else
    W = W - diag(diag(W)) + I;    %%% slef-affinity = 1
end

%%% Normalization  %%%%%%%%%%%%%%%%%
if symmetric
    d = sum(W,2);
    D = diag(d + eps);
    S = D^(-1/2)*W*D^(-1/2);      %%% Symmetric normalization is better
else
    S = W ./ repmat(sum(W, 2)+eps, 1, n);
end

%%% if remove self-affinity %%%
if remove_diagonal
    W = W - diag(diag(W));
end