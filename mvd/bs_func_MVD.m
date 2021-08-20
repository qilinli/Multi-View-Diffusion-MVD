function [A, out_beta, obj] = bs_func_MVD(W, para)

alpha = para.beta/(para.mu + sum(para.beta));
D = cellfun(@(x)( 1./sqrt(sum(x, 2)) ), W, 'UniformOutput', false);
D = cellfun(@(x)( x*x' ), D, 'UniformOutput', false);
[X, Y, V] = cellfun(@(x)(find(x)), W, 'UniformOutput', false);

if para.is_sparse
    S = cellfun(@(x,y)(sparse(double(x)).*double(y)), W, D, 'UniformOutput', false);
else
    S = cellfun(@(x,y)(x.*y), D, W, 'UniformOutput', false);
end

for i = 1:length(S)
    S{i} = knnSparse(S{i}, para.kS);
end

Z = zeros(size(W{1}), 'single');
for i = 1:length(W)
    Z = Z + para.beta(i) .* W{i};
end
Z = knnSparse(Z, para.kZ); %%% knn sparse

A = Z;
A_tmp = zeros([size(Z), length(W)], 'like', A);
obj = zeros(para.max_iter_alternating, 1);

for ii = 1:para.max_iter_alternating
    if para.if_debug
        objj(ii) = bs_compute_objective(A, D, X, Y, V, Z, para);
        fprintf('After %d iteration, obj is %5.4f\n', ii-1, objj(ii) );
    end
    
    % update A by diffusion
    tmp = zeros(para.max_iter_diffusion, 1, 'single');
    for iter = 1:para.max_iter_diffusion
        for v = 1:length(W)
            A_tmp(:, :, v) = alpha(v)*(S{v}*A*S{v}');
        end
        A_new = sum(A_tmp, 3) + (1-sum(alpha))*Z;
        if iter > 1, if abs( tmp(iter-1)-tmp(iter) ) < 1e-2, break; end, end
        A = A_new;
    end
    
    % update weights beta
    q = zeros(length(W), 1, 'single');
    for v = 1:length(W)
        q(v) = bs_compute_H(A, D{v}, X{v}, Y{v}, V{v}, para);
    end
    para.beta = update_beta(A, W, q, para);
    alpha = para.beta/(para.mu + 1);
    
    % calculate the obj
    obj_term1 = para.beta'*q;
    obj_term2 = (para.mu)*sum((A(:)-Z(:)).^2);
    obj_term3 = 0.5*para.lambda*norm(para.beta);
    obj(ii) = obj_term1 + obj_term2 + obj_term3;
    
    if ii > 1, if abs( obj(ii-1)-obj(ii) ) < para.thres, break; end, end
    
    Z = zeros(size(W{1}), 'single');
    for i = 1:length(W)
        Z = Z + para.beta(i) .* W{i};
    end
    Z = knnSparse(Z, para.kZ); %%% knn sparse
end


A = single(A);
A = knnSparse(A, para.kA);
out_beta = para.beta;
end