%% demo_compare_model_selection.m
% One-shot comparison of K selection methods on a synthetic dataset.
% Methods: ELBO (max), ICL (min), Predictive ELBO via K-fold (max).
% Configs: baseline vs enhanced (alpha0-learning + prune + merge).
% No toolboxes required.

clear; clc; rng(1);

%% ------------------- Synthetic mixture --------------------------
N        = 100;          % Number of "samples"
K_true   = 4;            % True cluster count
Nk_min   = 250;          % minimum points per cluster
Nk_max   = 400;          % maximum points per cluster
iso_noise_std = 0.05;    % additional isotropic noise
outlier_frac   = 0.03;   % proportion of uniform outliers
box_width      = 10.0;   % bounding box for cluster means

[X, labels_true, mu_true, Sig_true] = gen_mixture(N, K_true, Nk_min, Nk_max, box_width, iso_noise_std, outlier_frac); %#ok<ASGLU>
T = size(X,2);
fprintf('Generated data: N=%d, K_true=%d, T=%d (%.1f%% outliers)\n', N, K_true, T, 100*outlier_frac);

%% ------------------- Settings --------------------------
Kgrid = 1:12;
opts_base = struct('Kgrid',Kgrid,'restarts',5,'tol',1e-6,'max_iter',1000,'verbose',0,...
                   'diag_cov',false,'learn_alpha0',false,'tau_prune',0.0,'posthoc_merge',false);
opts_enh  = struct('Kgrid',Kgrid,'restarts',5,'tol',1e-6,'max_iter',1000,'verbose',0,...
                   'diag_cov',false,'learn_alpha0',true,'alpha0_min',1e-3,'tau_prune',0.01,'posthoc_merge',true,'merge_thresh',0.5);

kfold = 5;   % predictive free energy folds

%% ------------------- Baseline fit --------------------------
res_base = vb_gmm_freeenergy(X, max(Kgrid), opts_base);
[sel_ELBO_base, sel_ICL_base] = select_by_elbo_icl(res_base, X);

sel_pred_base = kfold_predictive_selection(X, Kgrid, opts_base, kfold);

%% ------------------- Enhanced fit --------------------------
res_enh = vb_gmm_freeenergy(X, max(Kgrid), opts_enh);
[sel_ELBO_enh, sel_ICL_enh] = select_by_elbo_icl(res_enh, X);

sel_pred_enh = kfold_predictive_selection(X, Kgrid, opts_enh, kfold);

%% ------------------- Report --------------------------
fprintf('\n=== K selection (baseline) ===\n');
fprintf('ELBO (max): K* = %d\n', sel_ELBO_base);
fprintf('ICL  (min): K* = %d\n', sel_ICL_base);
fprintf('Predictive ELBO %d-fold (max): K* = %d\n', kfold, sel_pred_base);

fprintf('\n=== K selection (enhanced: α0-learn + prune + merge) ===\n');
fprintf('ELBO (max): K* = %d\n', sel_ELBO_enh);
fprintf('ICL  (min): K* = %d\n', sel_ICL_enh);
fprintf('Predictive ELBO %d-fold (max): K* = %d\n', kfold, sel_pred_enh);

%% ======================= Helpers ==============================

function [X, labels_true, mu_true, Sigma_true] = gen_mixture(N, K, Nk_min, Nk_max, box_width, iso_noise_std, outlier_frac)
mu_true = (rand(N,K)-0.5)*2*box_width;
Sigma_true = cell(1,K);
for k=1:K
    A = randn(N); [Q,~] = qr(A,0);
    s = linspace(0.2, 1.0, N)'.^2;
    Sigma_true{k} = Q*diag(s)*Q';
end
Nk = randi([Nk_min Nk_max], K, 1);
T  = sum(Nk);
X = zeros(N,T); labels_true = zeros(1,T);
idx = 1;
for k=1:K
    C = chol(Sigma_true{k} + 1e-12*eye(N),'lower');
    Z = randn(N, Nk(k));
    Xk = C*Z + mu_true(:,k);
    X(:, idx:idx+Nk(k)-1) = Xk;
    labels_true(idx:idx+Nk(k)-1) = k; idx = idx+Nk(k);
end
X = X + iso_noise_std*randn(size(X));
Nout = round(outlier_frac*T);
if Nout>0
    O = (rand(N,Nout)-0.5)*2*box_width;
    X(:,1:Nout) = O; labels_true(1:Nout) = 0;
end
end

function [sel_ELBO, sel_ICL] = select_by_elbo_icl(res, X) %#ok<INUSD>
% Map each index to the actual K used in that fit (after prune/merge).
K_by_idx = cellfun(@(m) size(m.R,1), res.models_per_K);

% ELBO: choose argmax of F_per_K
[~,i_ELBO] = max(res.F_per_K);
sel_ELBO = K_by_idx(i_ELBO);

% ICL: choose argmin of ICL collected for each fit
ICL_vals = cellfun(@(s) s.ICL, res.info_per_K);
[~,i_ICL] = min(ICL_vals);
sel_ICL = K_by_idx(i_ICL);
end

function K_best = kfold_predictive_selection(X, Kgrid, opts, kfold)
% Split columns into kfold folds.
T = size(X,2);
perm = randperm(T);
fold_sizes = floor(T / kfold) * ones(1,kfold);
fold_sizes(1:mod(T,kfold)) = fold_sizes(1:mod(T,kfold)) + 1;
folds = cell(1,kfold);
s = 1;
for f=1:kfold
    e = s + fold_sizes(f) - 1;
    folds{f} = perm(s:e);
    s = e+1;
end
pred_scores = zeros(1,numel(Kgrid));

for ki = 1:numel(Kgrid)
    Kcand = Kgrid(ki);
    opts_local = opts; opts_local.Kgrid = Kcand;
    score_sum = 0;
    for f=1:kfold
        test_idx  = folds{f};
        train_idx = setdiff(1:T, test_idx);
        Xtr = X(:,train_idx); Xte = X(:,test_idx);
        % fit on train
        res_tr = vb_gmm_freeenergy(Xtr, Kcand, opts_local);
        M = res_tr.model;
        % predictive log-lik on test using VB expectations:
        ll = expected_log_gauss_external(Xte, M.m, M.beta, M.W, M.nu, opts.diag_cov);
        Elogpi = psi(M.alpha) - psi(sum(M.alpha));  % K x 1
        lse = logsumexp(bsxfun(@plus, ll, Elogpi), 1);
        score_sum = score_sum + sum(lse);
    end
    pred_scores(ki) = score_sum;
end
[~,ix] = max(pred_scores);
K_best = Kgrid(ix);
end

function A = logsumexp(Ain, dim)
if nargin<2, dim = 1; end
m = max(Ain,[],dim);
A = m + log(sum(exp(bsxfun(@minus, Ain, m)), dim));
end

function loglik = expected_log_gauss_external(X, m, beta, W, nu, diag_cov)
[N,T] = size(X); K = numel(m);
ElogLambda = zeros(K,1);
quad = zeros(K,T);
for k=1:K
    if diag_cov
        Wk = diag(diag(W{k}));
        ElogLambda(k) = sum(psi(0.5*(nu(k) - (0:(N-1))'))) + N*log(2) + logdet_diag_local(Wk);
        DX = X - m{k};
        quad(k,:) = N/beta(k) + nu(k)*sum((DX.^2).*diag(Wk),1);
    else
        Wk = W{k};
        ElogLambda(k) = sum(psi(0.5*(nu(k) - (0:(N-1))'))) + N*log(2) + logdet_spd_local(Wk);
        DX = X - m{k};
        quad(k,:) = N/beta(k) + nu(k)*sum((DX'*(Wk).*DX'),2)'; 
    end
end
loglik = 0.5*(ElogLambda - N*log(2*pi)) - 0.5*quad;
end

function L = logdet_spd_local(A)
U = chol(A + 1e-12*eye(size(A)),'upper');
L = 2*sum(log(diag(U)));
end
function L = logdet_diag_local(A)
L = sum(log(max(diag(A), realmin)));
end
