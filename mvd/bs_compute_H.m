function H =  bs_compute_H(A, D, X, Y, V, para)

A = A.*D;
if para.fusion_type==3
%     tic;
    H = bs_compute_H2(single(A), int32(X{1})-1, int32(Y{1})-1, single(V{1}),...
        int32(X{2})-1, int32(Y{2})-1, single(V{2}));
%     toc;
elseif para.fusion_type==1||para.fusion_type==2||para.fusion_type==4||para.fusion_type==5
%     tic;
    H = bs_compute_H1(single(A), int32(X)-1, int32(Y)-1, single(V));
%     toc;
end
H = H/2;