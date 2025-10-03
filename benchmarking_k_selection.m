%% benchmarking_k_selection.m
% Benchmark K selection: k-means (silhouette) vs VB-GMM (ELBO, ICL, predictive ELBO)
% Toolbox-free. Requires vb_gmm_freeenergy.m in path.
% The goal of this is to test the performance of VBB clustering with
% different situations (different noises)

clear; clc; rng(1);

% make some threads
if isempty(gcp('nocreate')), parpool('threads'); end

%% ---------------- Experiment grid ----------------
N                = 20;                          % dimensionality (features)
K_true_list      = 3:10;                        % true K values to test
noise_levels     = [0.01 0.02 0.05 0.10 0.15];  % isotropic Gaussian noise sd
outlier_frac     = 0.05;                        % uniform outliers fraction
Nk_range         = [200 1000];                  % per-cluster sample count bounds
box_width        = 8.0;                         % mean spread
replicates       = 5;                           % datasets per (K_true, noise)
Kgrid_eval       = 1:12;                        % K candidates for VB
Kgrid_kmeans     = 2:10;                        % silhouette scan

% VB options
opts_vb = struct( ...
    'Kgrid',        Kgrid_eval, ...
    'restarts',     5, ...
    'tol',          1e-6, ...
    'max_iter',     1000, ...
    'verbose',      0, ...
    'diag_cov',     false, ...
    'whiten',       false, ...           % set true if D≫samples
    'anneal_beta',  [0.3 0.6 0.85 1.0], ... % eterministic annealing
    'learn_alpha_vec', true, ...         % er-component α0 learning (ARD)
    'alpha0_min',   1e-3, ...
    'alpha0_maxit', 30, ...
    'tau_prune',    0.005, ...
    'posthoc_merge', false, ...          % keep false; we now do in-loop merge
    'merge_thresh', 1, ...
    'use_background', 't', ...           % 't' (Student-t), 'uniform', or 'none'
    'bg_df',        4, ...               % df for t background
    'bg_weight',    0.02, ...            % prior mass on background (effective)
    'split_once',   true, ...            % one targeted split pass
    'merge_once',   true ...             % one merge pass
);


% Predictive ELBO folds (reduce if runtime is high)
kfold_pred = 3;

%% --------------- Storage for results ---------------
rows = {};
row_headers = {'trial_id','K_true','noise_sd','method','K_selected','ARI','cent_rmse','cov_err','elapsed_sec'};

trial_id = 0;

%% --------------- Main sweep ------------------------
for noise_sd = noise_levels
    fprintf("Analysing with noise = %d\n", noise_sd)
    for K_true = K_true_list
        fprintf("Testing with K = %d\n", K_true)
        for r = 1:replicates
            fprintf("Run %d\n", r)
            trial_id = trial_id + 1;

            % ----- Generate dataset -----
            [X, labels_true, mu_true, Sigma_true] = gen_mixture(N, K_true, Nk_range(1), Nk_range(2), box_width, noise_sd, outlier_frac);
            T = size(X,2);

            % ----- k-means (silhouette-selected K) -----
            tic;
            [K_km, labels_km, C_km] = kmeans_silhouette_select(X, Kgrid_kmeans); %#ok<ASGLU>
            t_km = toc;
            ari_km = adjusted_rand_index(labels_true, labels_km);
            if K_km == K_true
                mu_km = estimate_centroids_from_labels(X, labels_km, K_km);
                Sigma_km = estimate_covariances_from_labels(X, labels_km, K_km);
                match_km = match_components(mu_true, mu_km);
                cent_err_km = rmse_centres(mu_true, mu_km, match_km);
                cov_err_km  = cov_fro_error(Sigma_true, Sigma_km, match_km);
            else
                cent_err_km = NaN; cov_err_km = NaN;
            end
            rows(end+1,:) = {trial_id,K_true,noise_sd,'kmeans_sil',K_km,ari_km,cent_err_km,cov_err_km,t_km}; %#ok<AGROW>

            % ----- VB-GMM (single sweep; reuse for ELBO & ICL) -----
            tic;
            opts_vb.prior = robust_niw_prior_from_data(X, 'nu0', N+2, 'beta0', 1.0);
            res_vb = vb_gmm_freeenergy(X, max(Kgrid_eval), opts_vb);
            t_vb = toc;

            % ELBO selection:
            [~,ix_elbo] = max(res_vb.F_per_K);
            K_elbo = size(res_vb.models_per_K{ix_elbo}.R,1);
            labels_elbo = hard_labels(res_vb.models_per_K{ix_elbo}.R);
            ari_elbo = adjusted_rand_index(labels_true, labels_elbo);
            [mu_elbo, Sig_elbo] = cent_cov_from_model(res_vb.models_per_K{ix_elbo});
            match_elbo = match_components(mu_true, mu_elbo);
            cent_err_elbo = rmse_centres(mu_true, mu_elbo, match_elbo);
            cov_err_elbo  = cov_fro_error(Sigma_true, Sig_elbo, match_elbo);
            rows(end+1,:) = {trial_id,K_true,noise_sd,'vb_elbo',K_elbo,ari_elbo,cent_err_elbo,cov_err_elbo,t_vb}; %#ok<AGROW>

            % ICL selection:
            ICL_vals = cellfun(@(s) s.ICL, res_vb.info_per_K);
            [~,ix_icl] = min(ICL_vals);
            K_icl = size(res_vb.models_per_K{ix_icl}.R,1);
            labels_icl = hard_labels(res_vb.models_per_K{ix_icl}.R);
            ari_icl = adjusted_rand_index(labels_true, labels_icl);
            [mu_icl, Sig_icl] = cent_cov_from_model(res_vb.models_per_K{ix_icl});
            match_icl = match_components(mu_true, mu_icl);
            cent_err_icl = rmse_centres(mu_true, mu_icl, match_icl);
            cov_err_icl  = cov_fro_error(Sigma_true, Sig_icl, match_icl);
            rows(end+1,:) = {trial_id,K_true,noise_sd,'vb_icl',K_icl,ari_icl,cent_err_icl,cov_err_icl,0}; %#ok<AGROW>

            % ----- VB-GMM (Predictive ELBO k-fold) -----
            tic;
            K_pred = kfold_predictive_selection(X, Kgrid_eval, opts_vb, kfold_pred);
            % Refit at chosen K on full data (for labels/ARI)
            opts_local = opts_vb; opts_local.Kgrid = K_pred;
            res_pred = vb_gmm_freeenergy(X, K_pred, opts_local);
            Mpred = res_pred.model;
            labels_pred = hard_labels(Mpred.R);
            ari_pred = adjusted_rand_index(labels_true, labels_pred);
            [mu_pred, Sig_pred] = cent_cov_from_model(Mpred);
            match_pred = match_components(mu_true, mu_pred);
            cent_err_pred = rmse_centres(mu_true, mu_pred, match_pred);
            cov_err_pred  = cov_fro_error(Sigma_true, Sig_pred, match_pred);
            t_pred = toc;
            rows(end+1,:) = {trial_id,K_true,noise_sd,'vb_pred',K_pred,ari_pred,cent_err_pred,cov_err_pred,t_pred};
            fprintf(" - For True K: %d, The folded-ELBO-predicted K is %d\n", K_true, K_pred)
        end
    end
end

%% --------------- Save CSV ---------------
T_out = cell2table(rows, 'VariableNames', row_headers);
csv_name = sprintf('kselection_results_%s.csv', datestr(now,'yyyymmdd_HHMMSS'));
writetable(T_out, csv_name);
fprintf('\nSaved results to: %s\n', csv_name);

%% --------------- Plots ------------------
methods = {'kmeans_sil','vb_elbo','vb_icl','vb_pred'};
method_names = containers.Map(methods, {'k-means (sil)','VB-ELBO','VB-ICL','VB-Pred'});
colors = lines(numel(methods));

% (1) K selection accuracy vs true K, per noise
figure('Name','K selection accuracy vs true K'); 
plot_idx = 1;
for ns = 1:numel(noise_levels)
    noise_sd = noise_levels(ns);
    subplot(2, ceil(numel(noise_levels)/2), plot_idx); plot_idx=plot_idx+1; hold on;
    for m=1:numel(methods)
        meth = methods{m};
        acc = zeros(1, numel(K_true_list));
        for ki=1:numel(K_true_list)
            Kt = K_true_list(ki);
            rows_sel = strcmp(T_out.method,meth) & T_out.noise_sd==noise_sd & T_out.K_true==Kt;
            Ks = T_out.K_selected(rows_sel);
            acc(ki) = mean(Ks == Kt);
        end
        plot(K_true_list, acc, '-o', 'DisplayName', method_names(meth), 'LineWidth',1.5, 'Color', colors(m,:));
    end
    xlabel('True K'); ylabel('P(K̂ = K)'); title(sprintf('Noise sd = %.2f', noise_sd)); ylim([0 1]); grid on; legend('Location','southoutside');
end

% (2) ARI vs noise (averaged over true K)
figure('Name','ARI vs noise (avg over true K)'); hold on;
for m=1:numel(methods)
    meth = methods{m};
    ari_mean = zeros(1, numel(noise_levels));
    for ns=1:numel(noise_levels)
        noise_sd = noise_levels(ns);
        rows_sel = strcmp(T_out.method,meth) & T_out.noise_sd==noise_sd;
        ari_mean(ns) = mean(T_out.ARI(rows_sel), 'omitnan');
    end
    plot(noise_levels, ari_mean, '-o', 'DisplayName', method_names(meth), 'LineWidth',1.5, 'Color', colors(m,:));
end
xlabel('Noise sd'); ylabel('Mean ARI'); grid on; legend('Location','southoutside');

% (3) ARI vs true K for each noise
figure('Name','ARI vs true K (by noise)'); 
plot_idx = 1;
for ns=1:numel(noise_levels)
    noise_sd = noise_levels(ns);
    subplot(2, ceil(numel(noise_levels)/2), plot_idx); plot_idx=plot_idx+1; hold on;
    for m=1:numel(methods)
        meth = methods{m};
        ari_mean = zeros(1, numel(K_true_list));
        for ki=1:numel(K_true_list)
            Kt = K_true_list(ki);
            rows_sel = strcmp(T_out.method,meth) & T_out.noise_sd==noise_sd & T_out.K_true==Kt;
            ari_mean(ki) = mean(T_out.ARI(rows_sel), 'omitnan');
        end
        plot(K_true_list, ari_mean, '-o', 'DisplayName', method_names(meth), 'LineWidth',1.5, 'Color', colors(m,:));
    end
    xlabel('True K'); ylabel('Mean ARI'); title(sprintf('Noise sd = %.2f', noise_sd)); grid on; legend('Location','southoutside');
end

disp('Done.');

%% ================= Helper functions (toolbox-free) =====================

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

function [K_best, labels_best, C_best] = kmeans_silhouette_select(X, Kgrid)
% Toolbox-free k-means + manual silhouette (Euclidean), pick K maximising mean silhouette.
best_score = -inf; K_best = Kgrid(1); labels_best = []; C_best = [];
for K = Kgrid
    [idx, C] = lloyd_kmeans(X, K, 10, 1000);
    s = manual_silhouette(X, idx, K);
    m = mean(s);
    if m > best_score
        best_score = m; K_best = K; labels_best = idx'; C_best = C;
    end
end
end

function [idx_best, C_best] = lloyd_kmeans(X, K, replicates, maxIter)
[N,T] = size(X);
best_inertia = inf; idx_best = []; C_best = [];
for rep=1:replicates
    C = kmeanspp_seed(X, K);
    idx = zeros(1,T);
    for it=1:maxIter
        % assign
        for t=1:T
            x = X(:,t); dmin = inf; imin = 1;
            for k=1:K
                d = sum((x - C(:,k)).^2);
                if d<dmin, dmin=d; imin=k; end
            end
            idx(t) = imin;
        end
        % update
        C_new = zeros(N,K); counts = zeros(1,K);
        for t=1:T
            C_new(:,idx(t)) = C_new(:,idx(t)) + X(:,t);
            counts(idx(t))  = counts(idx(t)) + 1;
        end
        for k=1:K
            if counts(k)>0, C_new(:,k) = C_new(:,k)/counts(k);
            else, C_new(:,k) = X(:,randi(T)); end
        end
        if max(vecnorm(C_new - C,2,1)) < 1e-6, C = C_new; break; end
        C = C_new;
    end
    inertia = 0; for t=1:T, inertia = inertia + sum((X(:,t) - C(:,idx(t))).^2); end
    if inertia < best_inertia, best_inertia = inertia; idx_best = idx; C_best = C; end
end
end

function C = kmeanspp_seed(X,K)
[N,T] = size(X);
C = zeros(N,K);
idx = randi(T); C(:,1) = X(:,idx);
D = sum((X - C(:,1)).^2,1);
for k=2:K
    p = D./sum(D); p = max(p,realmin); cdf = cumsum(p);
    r = rand; j = find(cdf>=r,1,'first');
    C(:,k) = X(:,j);
    Dj = sum((X - C(:,k)).^2,1);
    D = min(D, Dj);
end
end

function s = manual_silhouette(X, idx, K)
T = size(X,2); s = zeros(T,1);
members = cell(K,1);
for k=1:K, members{k} = find(idx==k); end
for i=1:T
    xi = X(:,i); ki = idx(i);
    Ii = members{ki};
    if numel(Ii)<=1
        a = 0;
    else
        acc = 0; c = 0;
        for jj = Ii(:)'
            if jj==i, continue; end
            acc = acc + sqrt(sum((xi - X(:,jj)).^2)); c = c + 1;
        end
        a = acc / max(c,1);
    end
    b = inf;
    for k=1:K
        if k==ki || isempty(members{k}), continue; end
        acc = 0; c = numel(members{k});
        for jj = members{k}(:)'
            acc = acc + sqrt(sum((xi - X(:,jj)).^2));
        end
        b = min(b, acc / c);
    end
    if isinf(b), b = 0; end
    s(i) = (b - a) / max(a, b + eps);
end
end

function [mu, Sigma] = cent_cov_from_model(M)
K = numel(M.m); N = numel(M.m{1});
mu = zeros(N,K); Sigma = cell(1,K);
for k=1:K
    mu(:,k) = M.m{k};
    S = inv(M.W{k}) / max(M.nu(k) - N - 1, 1e-6);
    Sigma{k} = (S+S')/2;
end
end

function labels = hard_labels(R)
[~,labels] = max(R,[],1);
end

function match = match_components(mu_true, mu_hat)
Khat  = size(mu_hat,2); Ktrue = size(mu_true,2);
D = zeros(Khat, Ktrue);
for i=1:Khat
    for j=1:Ktrue
        d = mu_hat(:,i) - mu_true(:,j);
        D(i,j) = sqrt(sum(d.^2));
    end
end
match = zeros(1,Khat); used = false(1,Ktrue);
for i=1:Khat
    [~,ord] = sort(D(i,:),'ascend');
    j = find(~used(ord),1,'first');
    if isempty(j), match(i)=ord(1); else, match(i)=ord(j); used(match(i))=true; end
end
end

function e = rmse_centres(mu_true, mu_hat, match)
K = size(mu_hat,2); e2 = 0; c = 0;
for i=1:K
    j = match(i);
    if j>=1 && j<=size(mu_true,2)
        d = mu_hat(:,i) - mu_true(:,j);
        e2 = e2 + sum(d.^2); c = c + numel(d);
    end
end
e = sqrt(e2 / max(c,1));
end

function e = cov_fro_error(S_true, S_hat, match)
K = min(numel(S_hat), numel(S_true)); acc = 0;
for i=1:K
    j = match(i);
    if j>=1 && j<=numel(S_true)
        A = S_hat{i}; B = S_true{j};
        acc = acc + norm(A - B, 'fro');
    end
end
e = acc / max(K,1);
end

function Sigma = estimate_covariances_from_labels(X, labels, K)
N = size(X,1); Sigma = cell(1,K);
for k=1:K
    idx = (labels==k);
    if nnz(idx)>=N+2
        Xk = X(:,idx);
        Ck = cov(Xk'); Ck = (Ck + Ck')/2;
        Sigma{k} = Ck;
    else
        Sigma{k} = eye(N);
    end
end
end

function mu = estimate_centroids_from_labels(X, labels, K)
N = size(X,1);
mu = zeros(N,K);
for k=1:K
    idx = (labels==k);
    if any(idx), mu(:,k) = mean(X(:,idx),2); else, mu(:,k) = 0; end
end
end

function K_best = kfold_predictive_selection(X, Kgrid, opts, kfold)
% Robust, stratified k-fold predictive selection using NIW posterior-predictive
% (multivariate Student-t) and parallelism across K.
%
% - Builds stratified folds via a quick kmeans prepartition, to balance folds.
% - Uses closed-form t-mixture predictive log-density instead of E[log N].
% - Fully guards against degenerate models (all/pruned components).
%
% Inputs:
%   X      : N x T
%   Kgrid  : vector of candidate K
%   opts   : options for vb_gmm_freeenergy (Kgrid overridden per Kcand)
%   kfold  : number of folds (e.g. 3)
%
% Output:
%   K_best : argmax predictive score over Kgrid

[N,T] = size(X);
if T < kfold, kfold = T; end

% ---------- stratified folds via coarse partition ----------
G = min( max(3, round(sqrt(T/50))), 20 );  % small number of coarse groups
[~, Cg] = lloyd_kmeans(X, G, 3, 50);      % toolbox-free
groups = zeros(1,T);
for t=1:T
    x = X(:,t);
    dmin=inf; g=1;
    for j=1:G
        d = sum((x - Cg(:,j)).^2);
        if d<dmin, dmin=d; g=j; end
    end
    groups(t)=g;
end

folds = cell(1,kfold); for f=1:kfold, folds{f} = []; end
for g=1:G
    idxg = find(groups==g);
    idxg = idxg(randperm(numel(idxg)));
    for q=1:numel(idxg)
        f = 1 + mod(q-1, kfold);
        folds{f}(end+1) = idxg(q);
    end
end

% ---------- main scoring ----------
pred_scores = -inf(1,numel(Kgrid));

parfor ki = 1:numel(Kgrid)
    Kcand = Kgrid(ki);
    opts_local = opts; opts_local.Kgrid = Kcand;

    score_sum = 0; valid_folds = 0;
    for f=1:kfold
        test_idx  = folds{f};
        if isempty(test_idx), continue; end
        train_idx = setdiff(1:T, test_idx);

        % guard
        if numel(train_idx) < max(10, Kcand+2)
            continue;
        end

        Xtr = X(:,train_idx); Xte = X(:,test_idx);

        % fit VB on train
        res_tr = vb_gmm_freeenergy(Xtr, Kcand, opts_local);
        M = res_tr.model;

        % if degenerate, skip fold
        if ~isfield(M,'m') || isempty(M.m) || numel(M.m)<1
            continue;
        end

        % use NIW posterior-predictive (multivariate t) mixture
        logpred = log_pred_student_t_mixture(Xte, M.m, M.beta, M.W, M.nu, M.alpha, opts.diag_cov);

        % accumulate
        score_sum = score_sum + sum(logpred);
        valid_folds = valid_folds + 1;
    end

    if valid_folds > 0
        pred_scores(ki) = score_sum / valid_folds;
    else
        pred_scores(ki) = -inf;
    end
end

% tie-breaker: smallest K among maxima
mx = max(pred_scores);
cands = find(abs(pred_scores - mx) < 1e-10);
K_best = min(Kgrid(cands));
end

function lp = log_pred_student_t_mixture(X, m, beta, W, nu, alpha, diag_cov)
% Posterior-predictive of NIW is multivariate Student-t:
%   x | comp k  ~  St( m_k,
%                      Σpred_k = ((beta_k+1)/(beta_k*(nu_k - N + 1))) * inv(W_k),
%                      df = nu_k - N + 1 )
% We compute mixture log p(x) with weights E[pi] = alpha/sum(alpha).

[N,T] = size(X);
K = numel(m);

% weights (normalised)
tau = 0.9;
aw = alpha(:).^tau; aw = aw / sum(aw);
logcomp = -inf(K,T);

for k=1:K
    df = max(nu(k) - N + 1, 1e-6);
    if diag_cov
        Wk = diag(max(diag(W{k}), 1e-12));
    else
        Wk = (W{k} + W{k}')/2;
    end
    % predictive covariance
    S = ((beta(k)+1)/(beta(k)*df)) * inv(Wk);
    S = (S+S')/2;

    % log multivariate Student-t
    U = chol(S + 1e-12*eye(N),'upper');
    logdetS = 2*sum(log(diag(U)));
    Q = U \ (X - m{k});
    Q = sum(Q.^2,1);
    logZ = gammaln(0.5*(df+N)) - gammaln(0.5*df) - 0.5*N*log(df*pi) - 0.5*logdetS;
    logcomp(k,:) = log(aw(k)+eps) + (logZ - 0.5*(df+N)*log(1 + Q/df));
end

% log-sum-exp across components
mrow = max(logcomp,[],1);
lp = mrow + log(sum(exp(logcomp - mrow),1));
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

function ari = adjusted_rand_index(labels_true, labels_pred)
% Toolbox-free ARI implementation.
Lt = labels_true(:); Lp = labels_pred(:);
[~,~,Lt] = unique(Lt,'stable');  % relabel 1..nt
[~,~,Lp] = unique(Lp,'stable');  % relabel 1..np
nt = max(Lt); np = max(Lp);
n = numel(Lt);

% Contingency table
M = zeros(nt,np);
for i=1:n
    M(Lt(i), Lp(i)) = M(Lt(i), Lp(i)) + 1;
end
a = sum(M,2); b = sum(M,1);

% Combinatorial terms (n choose 2 style)
sumC2 = sum(sum(M.*(M-1)))/2;
sumA2 = sum(a.*(a-1))/2;
sumB2 = sum(b.*(b-1))/2;
denom = max(n*(n-1)/2, 1);

expected = (sumA2 * sumB2) / denom;
maxidx  = 0.5 * (sumA2 + sumB2);
ari = (sumC2 - expected) / max(maxidx - expected, eps);
end

function prior = robust_niw_prior_from_data(X, varargin)
% Construct a robust NIW prior from data using Ledoit–Wolf shrinkage towards diag.
% Optional name-value:
%   'nu0'   (default N+2)
%   'beta0' (default 1.0)
%   'lambda' in [0,1] to override shrinkage (default: LW analytic)
[N,~] = size(X);
p.nu0   = N + 2;
p.beta0 = 1.0;
p = parse_opts(p, varargin{:});

% empirical cov (fallback safe)
S = cov(X'); 
if ~isfinite(sum(S(:))) || any(isnan(S(:)))
    Xc = X - mean(X,2);
    S = (Xc*Xc')/max(size(X,2)-1,1);
end
S = (S+S')/2;

% Ledoit–Wolf shrinkage to diagonal
diagS = diag(diag(S));
lambda = ledoit_wolf_lambda(X, S, diagS);
lambda = min(max(lambda,0),1);
if any(strcmpi(varargin(1:2:end),'lambda'))
    lambda = p.lambda;
end
Ssh = (1 - lambda)*S + lambda*diagS;

% Prior parameters
prior = struct();
prior.m0   = mean(X,2);
prior.beta0= p.beta0;
prior.nu0  = p.nu0;
prior.W0   = inv(Ssh) * (prior.nu0 - N - 1);  % so E[Σ]=Ssh
prior.W0   = (prior.W0 + prior.W0')/2;

    function val = parse_opts(defs, varargin)
        val = defs;
        for i=1:2:numel(varargin)
            val.(varargin{i}) = varargin{i+1};
        end
    end
end

function lambda = ledoit_wolf_lambda(X, S, T)
% Approximate Ledoit–Wolf shrinkage intensity towards target T (diagonal).
% X: N x T, zero-mean not required (we subtract mean internally).
[N,Tn] = size(X);
Xc = X - mean(X,2);
mu = 0;
phi = 0;
for t=1:Tn
    xt = Xc(:,t);
    Yt = (xt*xt') - S;
    phi = phi + sum(sum(Yt.^2));
end
phi = phi / Tn;

gamma = norm(S - T, 'fro')^2;
lambda = min(max(phi / max(gamma, eps), 0), 1);
end
