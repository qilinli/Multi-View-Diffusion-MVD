function objective_value = bs_compute_objective(A, D, X, Y, V, Z, para)

if para.fusion_type==1||para.fusion_type==2||para.fusion_type==3
    objective_value = bs_compute_H(A, D, X, Y, V, para);
elseif para.fusion_type==4
    H = zeros(length(V), 1, 'single');
    for v = 1:length(V)
        H(v) = bs_compute_H(A, D{v}, X{v}, Y{v}, V{v}, para);
    end
    objective_value = (para.beta)'*H;
    objective_value = objective_value + 0.5*(para.lambda)*norm(para.beta);
end
objective_value = objective_value + (para.mu)*sum((A(:)-Z(:)).^2);