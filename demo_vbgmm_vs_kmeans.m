%% demo_vbgmm_vs_kmeans.m
% Toolbox-free simulation comparing VB-GMM (free energy) vs manual k-means
% + manual silhouette scoring. Requires vb_gmm_freeenergy.m in the path.
clear; clc; rng(1);

%% ------------------- Simulation configuration --------------------------
N        = 6;
K_true   = 4;
Nk_min   = 250;
Nk_max   = 400;
iso_noise_std = 0.05;
outlier_frac   = 0.03;
box_width      = 10.0;

Kmax     = 10;
opts = struct('restarts',5,'tol',1e-6,'max_iter',1000,'verbose',1,'diag_cov',false);
Kgrid_km = 2:10;

%% ------------------- Generate synthetic mixture ------------------------
[w_true, mu_true, Sigma_true, labels_true, X] = ...
    generate_gaussian_mixture(N, K_true, Nk_min, Nk_max, box_width, iso_noise_std, outlier_frac);

fprintf('Generated data: N=%d, K_true=%d, T=%d (%.1f%% outliers)\n', ...
    N, K_true, size(X,2), 100*outlier_frac);

%% ------------------- Run VB-GMM (free energy) --------------------------
res = vb_gmm_freeenergy(X, Kmax, opts);

K_vb   = res.K_star;
lab_vb = res.labels;
[mu_vb, Sigma_vb, w_vb] = posterior_point_estimates(res.model);

%% ------------------- Match components and compute errors ---------------
match = match_components(mu_true, mu_vb); % inferred->true (length K_vb)

cent_err = rmse_centres(mu_true, mu_vb, match);
cov_err  = cov_fro_error(Sigma_true, Sigma_vb, match);
w_vb_aligned = align_weights_to_true(w_vb, match, K_true);   % <-- FIX
w_err    = sum(abs(w_true - w_vb_aligned));

ari_vb   = adjusted_rand_index(labels_true, lab_vb);

%% ------------------- k-means + silhouette (toolbox-free) ---------------
[sil_bestK, km_labels_best, km_centroids_map] = kmeans_with_manual_silhouette(X, Kgrid_km);

K_km   = sil_bestK;
lab_km = km_labels_best;

if K_km == K_true
    mu_km = km_centroids_map{K_km};
    Sigma_km = estimate_covariances_from_labels(X, lab_km, K_km);
    match_km = match_components(mu_true, mu_km);
    cent_err_km = rmse_centres(mu_true, mu_km, match_km);
    cov_err_km  = cov_fro_error(Sigma_true, Sigma_km, match_km);
else
    cent_err_km = NaN; cov_err_km = NaN;
end
ari_km = adjusted_rand_index(labels_true, lab_km);

%% ------------------- Report results ------------------------------------
fprintf('\n=== Model order selection ===\n');
fprintf('True K        : %d\n', K_true);
fprintf('VB-GMM K*     : %d  (by max free energy)\n', K_vb);
fprintf('k-means K_sil : %d  (by max mean silhouette)\n', K_km);

fprintf('\n=== Parameter recovery (when matched) ===\n');
fprintf('Centres RMSE (VB) : %.4f\n', cent_err);
fprintf('Covariances ΔFrob (VB): %.4f\n', cov_err);
fprintf('Weights L1 (VB)   : %.4f\n', w_err);
if ~isnan(cent_err_km)
    fprintf('Centres RMSE (k-means @K_true): %.4f\n', cent_err_km);
    fprintf('Covariances ΔFrob (k-means @K_true): %.4f\n', cov_err_km);
else
    fprintf('k-means centre/cov errors: n/a (K_sil ≠ K_true)\n');
end

fprintf('\n=== Clustering agreement (Adjusted Rand Index) ===\n');
fprintf('ARI (VB)      : %.4f\n', ari_vb);
fprintf('ARI (k-means) : %.4f\n', ari_km);

%% ------------------- (Optional) quick plots for 2D/3D ------------------
if N==2 || N==3
    quick_scatter_plot(X, labels_true, 'True labels');
    quick_scatter_plot(X, lab_vb,     sprintf('VB-GMM labels (K*=%d)', K_vb));
    quick_scatter_plot(X, lab_km,     sprintf('k-means labels (K_sil=%d)', K_km));
end

%% ======================= Helper functions ==============================
function [w_true, mu_true, Sigma_true, labels_true, X] = generate_gaussian_mixture(N, K, Nk_min, Nk_max, box_width, iso_noise_std, outlier_frac)
w = rand(K,1); w = w/sum(w);
Nk = randi([Nk_min Nk_max], K, 1);
T  = sum(Nk);
mu_true = (rand(N,K)-0.5)*2*box_width;
Sigma_true = cell(1,K);
for k=1:K
    A = randn(N); [Q,~] = qr(A,0);
    s = linspace(0.2, 1.0, N)'.^2;
    Sigma_true{k} = Q*diag(s)*Q';
end
X = zeros(N, T); labels_true = zeros(1,T);
idx = 1;
for k=1:K
    C = chol(Sigma_true{k} + 1e-12*eye(N),'lower');
    Z = randn(N, Nk(k));
    Xk = C*Z + mu_true(:,k);
    X(:,idx:idx+Nk(k)-1) = Xk;
    labels_true(idx:idx+Nk(k)-1) = k;
    idx = idx+Nk(k);
end
X = X + iso_noise_std*randn(size(X));
Nout = round(outlier_frac*T);
if Nout>0
    O = (rand(N,Nout)-0.5)*2*box_width;
    X(:,1:Nout) = O;
    labels_true(1:Nout) = 0;
end
valid = labels_true>0;
w_true = zeros(K,1);
for k=1:K, w_true(k) = sum(labels_true(valid)==k); end
w_true = w_true / sum(valid);
end

function [mu, Sigma, w] = posterior_point_estimates(model)
K = numel(model.m); N = numel(model.m{1});
mu = zeros(N,K); Sigma = cell(1,K);
for k=1:K
    mu(:,k) = model.m{k};
    Wk = model.W{k};
    Sig = inv(Wk) / max(model.nu(k) - N - 1, 1e-6);
    Sig = (Sig+Sig')/2; Sigma{k} = Sig;
end
w = model.Epi(:); w = w/sum(w);
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

function w_aligned = align_weights_to_true(w_vb, match, K_true)
% Map VB weights (length K_vb) onto K_true-by-1 by summing any VB weights
% assigned to the same true component (handles K_vb ≠ K_true, many-to-one).
w_aligned = zeros(K_true,1);
K_vb = numel(w_vb);
for i=1:K_vb
    j = match(i);
    if j>=1 && j<=K_true
        w_aligned(j) = w_aligned(j) + w_vb(i);
    end
end
% Ensure it sums to 1 (floating-point safety)
s = sum(w_aligned);
if s>0, w_aligned = w_aligned / s; end
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

function ari = adjusted_rand_index(labels_true, labels_pred)
Lt = labels_true(:); Lp = labels_pred(:);
[~,~,Lt] = unique(Lt,'stable'); [~,~,Lp] = unique(Lp,'stable');
nt = max(Lt); np = max(Lp); n = numel(Lt);
M = zeros(nt,np);
for i=1:n, M(Lt(i), Lp(i)) = M(Lt(i), Lp(i)) + 1; end
a = sum(M,2); b = sum(M,1);
sumC2 = sum(sum(M.*(M-1)))/2;
sumA2 = sum(a.*(a-1))/2; sumB2 = sum(b.*(b-1))/2;
expected = sumA2*sumB2 / max(n*(n-1)/2,1);
maxidx  = 0.5*(sumA2 + sumB2);
ari = (sumC2 - expected) / max(maxidx - expected, eps);
end

function [bestK, labels_best, centroids_map] = kmeans_with_manual_silhouette(X, Kgrid)
T = size(X,2);
centroids_map = cell(max(Kgrid),1);
mean_sil = -inf(size(Kgrid));
labels_store = cell(size(Kgrid));
for ii=1:numel(Kgrid)
    K = Kgrid(ii);
    [idx, C] = lloyd_kmeans(X, K, 10, 1000);
    s = manual_silhouette(X, idx, K);
    mean_sil(ii) = mean(s);
    labels_store{ii} = idx';
    centroids_map{K} = C;
end
[~,ix] = max(mean_sil);
bestK = Kgrid(ix);
labels_best = labels_store{ix};
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
            x = X(:,t);
            dmin = inf; imin = 1;
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
            counts(idx(t)) = counts(idx(t)) + 1;
        end
        for k=1:K
            if counts(k)>0, C_new(:,k) = C_new(:,k) / counts(k);
            else, C_new(:,k) = X(:,randi(T)); % reseed empty cluster
            end
        end
        if max(vecnorm(C_new - C,2,1)) < 1e-6, C = C_new; break; end
        C = C_new;
    end
    inertia = 0;
    for t=1:T, inertia = inertia + sum((X(:,t) - C(:,idx(t))).^2); end
    if inertia < best_inertia
        best_inertia = inertia; idx_best = idx; C_best = C;
    end
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
T = size(X,2);
s = zeros(T,1);
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
            acc = acc + sqrt(sum((xi - X(:,jj)).^2));
            c = c + 1;
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

function quick_scatter_plot(X, labels, ttl)
N = size(X,1);
cols = distinguishable_colors(max(labels)+1);
if N==2
    figure; hold on;
    u = unique(labels);
    for i=1:numel(u)
        li = u(i); idx = labels==li;
        scatter(X(1,idx), X(2,idx), 10, cols(1+li,:), 'filled');
    end
    axis equal; grid on; title(ttl);
elseif N==3
    figure; hold on;
    u = unique(labels);
    for i=1:numel(u)
        li = u(i); idx = labels==li;
        scatter3(X(1,idx), X(2,idx), X(3,idx), 10, cols(1+li,:), 'filled');
    end
    axis vis3d; grid on; title(ttl);
end
end

function C = distinguishable_colors(m)
if m<=1, C = [0 0 0]; return; end
phi = (1+sqrt(5))/2;
h = mod((0:m-1)'/phi,1);
s = 0.65*ones(m,1); v = 0.9*ones(m,1);
C = hsv2rgb([h s v]);
end
