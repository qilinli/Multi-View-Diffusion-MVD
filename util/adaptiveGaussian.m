function W=adaptiveGaussian(data, K, metric)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code is an implementation of adaptive Gaussian affinity used in:
%%% "Affinity Learning via a Diffusion Process for Subspace Clustering"
%%% Note that the self-affinity is defined, and it's not simply 0 or 1
%%% By QILIN LI (li.qilin@postgrad.curtin.edu.au)
%%% Last Update 05/07/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    metric = 'squaredEuclidean';
end

n = size(data, 1); 

if strcmp(metric, 'cosine')
    D = pdist2_fast(data, data, 'cosine');
    D = D.^2;
else
    D = EuDist2(data);
end

D = D - diag(diag(D));   %%% Zero distance to itself
[T, ~] = sort(D, 2);

W = zeros(n,n);
for i = 1:n
    for j = 1:n
        sigma = mean([T(i,2:K+1), T(j,2:K+1)]);
        if sigma == 0, warning("AdaptiveGaussian.m: sigm=0, cannot compute W(%d, %d)!", i, j); end
        W(i,j) = normpdf(D(i,j), 0, 0.35*sigma);
    end
end
W = preprocessingW(W, 'D', 1, 1);


