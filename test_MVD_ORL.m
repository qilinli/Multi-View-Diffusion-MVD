clear;
addpath('util/');
addpath('mvd/');
%compile_func(0);

%%%%%%%%%%%%%  Hyper-parameters %%%%%%%%%%
para.kSig = 17;
para.kW = 7;
para.kS = 7;
para.kZ = 7;
para.kA = 10;
para.mu = 0.01;
para.lambda = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

para.max_iter_diffusion = 10;
para.max_iter_alternating = 10;
para.if_debug = 0;
para.thres = 1e-3;
para.is_sparse = 0;
para.fusion_type = 4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Load dataset %%%
%%% Use cosine for text data %%%
dataset = "ORL";
load('data/' + dataset + '.mat');
metric = 'squaredEuclidean';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Initialize view-weighting %%%
Y = gt;
class_num = length(unique(Y));
V = length(fea);
para.beta = ones(V, 1) / V;      % init the weighting to be 1/V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Construct initial affinity graph %%%
W = cell(1, V);
for i = 1:V
    X = NormalizeFea(fea{i}, 1);
    W{i} = adaptiveGaussian(X, para.kSig, metric);
    W{i} = knnSparse(W{i}, para.kW);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Run 10 trials for average reuslts and std %%%
for trial = 1:10
    ACC = [];
    NMI = [];
    for ii = 1:length(fea)
        Yii = SpectralClustering(W{ii}, class_num);
        ACC = [ACC; clusteringAcc(Yii, Y)];
        NMI = [NMI; nmi(Yii, Y)];
    end
    
    [A, out_beta, obj] = bs_func_MVD(W, para);
    Y_MVD = SpectralClustering(double(A), class_num);
    acc_MVD = clusteringAcc(Y_MVD, Y);
    nmi_MVD = nmi(Y_MVD, Y);
    
    ACC_SC(trial) = max(ACC);
    NMI_SC(trial) = max(NMI);
    ACC_MVD(trial) = acc_MVD;
    NMI_MVD(trial) = nmi_MVD;
end
fprintf("===dataset:%s, lambda: %d, mu: %f===\n",...
    dataset, para.lambda, para.mu);
fprintf("SC(best): Acc(%.4f ± %.4f), NMI(%.4f ± %.4f)\n",...
    mean(ACC_SC), std(ACC_SC), mean(NMI_SC), std(NMI_SC));
fprintf("MVD     : Acc(%.4f ± %.4f), NMI(%.4f ± %.4f)\n",...
    mean(ACC_MVD), std(ACC_MVD), mean(NMI_MVD), std(NMI_MVD));
fprintf("View-specific [NMI weighting] is:\n");
disp([NMI, out_beta]);
disp("==========================================");
