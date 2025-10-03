%% benchmark_ms_k_selection.m
% Benchmark K selection: k-means (silhouette) vs VB-GMM (ELBO, ICL, predictive ELBO)
% Toolbox-free. Requires vb_gmm_freeenergy.m in path.
% This version loads microstate maps from MetaMaps (metaMaps_2023.mat)
% and synthesises "ecologically valid" EEG:
%  - sticky microstate sequence (random K in [3..7], random subset of 7 maps)
%  - within each activation, weight varies smoothly in [-1,1]
%  - additive Gaussian sensor noise
% Two clustering inputs are tested: RAW (all timepoints) and GFP15 (top 15% |w|).
%
% Outputs:
%   CSV with per-trial results + a MAT summary (opts, seeds, etc.)

clear; clc; rng(1);
if isempty(gcp('nocreate')), parpool('threads'); end

%% ---------------- Config ----------------
meta_file        = 'Metamaps MS Template/MetaMaps_2023_06.mat';
K_range_run      = [3 7];                % K per run chosen uniformly in this range
noise_levels     = [0.01 0.02 0.05 0.10 0.15];
replicates       = 20;                   % datasets per noise level
T_target_range   = [4000 8000];          % target total samples per run (pre-noise)
gfp_top_frac     = 0.15;                 % top 15% |weights|
smooth_sigma     = 15;                   % smoothing (samples) for weights
seg_len_range    = [80 250];             % sticky segments (samples)
outlier_frac     = 0.02;                 % set >0 if you want uniform artefact blocks
Kgrid_eval       = 1:12;
Kgrid_kmeans     = 2:10;

% VB options (aligned with your current work)
opts_vb = struct( ...
    'Kgrid',        Kgrid_eval, ...
    'restarts',     5, ...
    'tol',          1e-6, ...
    'max_iter',     1000, ...
    'verbose',      0, ...
    'diag_cov',     false, ...
    'whiten',       false, ...
    'anneal_beta',  [0.3 0.6 0.85 1.0], ...
    'learn_alpha_vec', true, ...
    'alpha0_min',   1e-3, ...
    'alpha0_maxit', 30, ...
    'tau_prune',    0.005, ...
    'posthoc_merge', false, ...
    'merge_thresh', 1.0, ...
    'use_background','t', ...
    'bg_df',        4, ...
    'bg_weight',    0.02, ...
    'split_once',   true, ...
    'merge_once',   true ...
);

% Predictive selection
kfold_pred = 3;

% Methods to run
input_modes = {'RAW','GFP15'};

%% --------------- Load MetaMaps ----------------
[Maps, Channels] = load_metamaps(meta_file);
[Nch, Nmaps] = size(Maps);
fprintf('Loaded %d-channel, %d-map microstate set from %s\n', Nch, Nmaps, meta_file);

%% --------------- Results store ---------------
rows = {};
row_headers = {'trial_id','input_mode','K_true','noise_sd','method','K_selected','ARI','cent_rmse','elapsed_sec'};

trial_id = 0;
master_seed = rng;   % reproducibility handle

%% --------------- Main sweep -------------------
for noise_sd = noise_levels
    fprintf('=== Noise sd = %.3f ===\n', noise_sd);
    for r = 1:replicates
        trial_id = trial_id + 1;

        % ----- choose K and subset of maps per run -----
        K_true = randi(K_range_run);
        pick = randperm(Nmaps, K_true);
        Maps_sel = Maps(:, pick);

        % ----- synthesise EEG -----
        seed_this = double(1e6*rand);    % recordable per-trial seed
        rng(seed_this, 'twister');
        T_target = randi(T_target_range);
        [X_clean, labels_true, w, seg_ids] = synth_ms_eeg(Maps_sel, T_target, seg_len_range, smooth_sigma);
        % add sensor noise
        X = X_clean + noise_sd * randn(size(X_clean));

        % polarity alignment so ±map don't double K
        X = polarity_align_by_pca(X);
        % truth centres for metrics (polarity-aligned to the same axis)
        mu_true = polarity_align_by_pca(Maps_sel);

        % compute GFP peaks as top gfp_top_frac of |w|
        idx_gfp = select_gfp_peaks_from_weight(w, gfp_top_frac);

        % optional outliers
        if outlier_frac > 0
            Nout = round(outlier_frac * size(X,2));
            X(:, 1:Nout) = (rand(Nch, Nout) - 0.5) * 2 * max(abs(X(:)));
            labels_true(1:Nout) = 0;
        end

        fprintf("EEG Simulated")

        % ----- run both input modes -----
        for mode_i = 1:numel(input_modes)
            mode = input_modes{mode_i};
            switch mode
                case 'RAW'
                    X_in = X; labels_in = labels_true;
                case 'GFP15'
                    X_in = X(:, idx_gfp);
                    labels_in = labels_true(idx_gfp);
                otherwise
                    error('Unknown input mode: %s', mode);
            end

            % quick guard: need at least a handful of samples
            if size(X_in,2) < max(30, 5*K_true)
                warning('Too few samples in %s (T=%d). Skipping trial %d.', mode, size(X_in,2), trial_id);
                continue;
            end

            % ---- k-means (silhouette) ----
            t0 = tic;
            [K_km, labels_km, ~] = kmeans_silhouette_select(X_in, Kgrid_kmeans);
            t_km = toc(t0);
            ari_km = adjusted_rand_index(labels_in, labels_km);
            cent_err_km = centroid_rmse_if_Kmatch(mu_true, X_in, labels_km, K_km);
            rows(end+1,:) = {trial_id,mode,K_true,noise_sd,'kmeans_sil',K_km,ari_km,cent_err_km,t_km};

            % ---- VB-GMM (ELBO/ICL) ----
            t0 = tic;
            opts_here = opts_vb;
            opts_here.prior = robust_niw_prior_from_data(X_in, 'nu0', Nch+2, 'beta0', 1.0);
            res_vb = vb_gmm_freeenergy(X_in, max(Kgrid_eval), opts_here);
            t_vb = toc(t0);

            % ELBO
            [~,ix_elbo] = max(res_vb.F_per_K);
            K_elbo = size(res_vb.models_per_K{ix_elbo}.R,1);
            labels_elbo = hard_labels(res_vb.models_per_K{ix_elbo}.R);
            ari_elbo = adjusted_rand_index(labels_in, labels_elbo);
            cent_err_elbo = centroid_rmse_vs_truth(mu_true, res_vb.models_per_K{ix_elbo});
            rows(end+1,:) = {trial_id,mode,K_true,noise_sd,'vb_elbo',K_elbo,ari_elbo,cent_err_elbo,t_vb};

            % ICL
            ICL_vals = cellfun(@(s) s.ICL, res_vb.info_per_K);
            [~,ix_icl] = min(ICL_vals);
            K_icl = size(res_vb.models_per_K{ix_icl}.R,1);
            labels_icl = hard_labels(res_vb.models_per_K{ix_icl}.R);
            ari_icl = adjusted_rand_index(labels_in, labels_icl);
            cent_err_icl = centroid_rmse_vs_truth(mu_true, res_vb.models_per_K{ix_icl});
            rows(end+1,:) = {trial_id,mode,K_true,noise_sd,'vb_icl',K_icl,ari_icl,cent_err_icl,0}; 

            % ---- VB predictive (k-fold) ----
            t0 = tic;
            K_pred = kfold_predictive_selection(X_in, Kgrid_eval, opts_here, kfold_pred);
            opts_local = opts_here; opts_local.Kgrid = K_pred;
            res_pred = vb_gmm_freeenergy(X_in, K_pred, opts_local);
            labels_pred = hard_labels(res_pred.model.R);
            ari_pred = adjusted_rand_index(labels_in, labels_pred);
            cent_err_pred = centroid_rmse_vs_truth(mu_true, res_pred.model);
            t_pred = toc(t0);
            rows(end+1,:) = {trial_id,mode,K_true,noise_sd,'vb_pred',K_pred,ari_pred,cent_err_pred,t_pred};
        end

        % reset RNG to master stream
        rng(master_seed);
    end
end

%% --------------- Save CSV/MAT ---------------
T_out = cell2table(rows, 'VariableNames', row_headers);
csv_name = sprintf('kselection_ms_results_%s.csv', datestr(now,'yyyymmdd_HHMMSS'));
writetable(T_out, csv_name);
fprintf('\nSaved results to: %s\n', csv_name);

meta = struct('meta_file',meta_file,'input_modes',{input_modes}, ...
              'K_range_run',K_range_run,'noise_levels',noise_levels, ...
              'replicates',replicates,'T_target_range',T_target_range, ...
              'gfp_top_frac',gfp_top_frac,'smooth_sigma',smooth_sigma, ...
              'seg_len_range',seg_len_range,'opts_vb',opts_vb);
mat_name = sprintf('kselection_ms_summary_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
save(mat_name, 'T_out','meta');
fprintf('Saved summary MAT to: %s\n', mat_name);

%% --------------- (Optional) quick plots ---------------
methods = {'kmeans_sil','vb_elbo','vb_icl','vb_pred'};
method_names = containers.Map(methods, {'k-means (sil)','VB-ELBO','VB-ICL','VB-Pred'});
colors = lines(numel(methods));
for mode_i = 1:numel(input_modes)
    mode = input_modes{mode_i};
    rows_mode = strcmp(T_out.input_mode, mode);

    figure('Name',sprintf('%s: K selection accuracy vs K_true',mode));
    hold on;
    K_true_list = K_range_run(1):K_range_run(2);
    for m=1:numel(methods)
        meth = methods{m};
        acc = zeros(1, numel(K_true_list));
        for ki=1:numel(K_true_list)
            Kt = K_true_list(ki);
            rows_sel = rows_mode & strcmp(T_out.method,meth) & T_out.K_true==Kt;
            Ks = T_out.K_selected(rows_sel);
            if isempty(Ks), acc(ki)=NaN; else, acc(ki) = mean(Ks == Kt); end
        end
        plot(K_true_list, acc, '-o', 'DisplayName', method_names(meth), 'LineWidth',1.5, 'Color', colors(m,:));
    end
    xlabel('True K'); ylabel('P(\hat K = K)'); ylim([0 1]); grid on; legend('Location','southoutside');
end

disp('Done.');

%% ====================== Helper functions =======================

function [Maps, Channels] = load_metamaps(fname)
%LOAD_METAMAPS  Load MetaMaps microstate templates robustly.
% Returns:
%   Maps     : Nch x Nms double, each column a unit-L2 map
%   Channels : [] or a struct/cell with channel names if present

S = load(fname);
vars = fieldnames(S);

% ---- pick the dataset variable ----
preferred = {'data','Data','Maps','maps','Templates','templates','MetaMaps','metaMaps'};
vname = '';
for k=1:numel(preferred)
    if ismember(preferred{k}, vars), vname = preferred{k}; break; end
end
if isempty(vname)
    % fallback: first non-Channel-like variable
    for k=1:numel(vars)
        if ~contains(lower(vars{k}), 'channel')
            vname = vars{k}; break;
        end
    end
end
if isempty(vname)
    error('Could not find a maps/data variable in %s. Found only: %s', fname, strjoin(vars, ', '));
end
D = S.(vname);

% ---- try to interpret D ----
Maps = [];
if isnumeric(D)
    % Case C: numeric matrix Nch x Nms
    Maps = double(D);
elseif isstruct(D)
    sz = size(D);
    fns = fieldnames(D);

    % Try to find a numeric field
    cand = '';
    scalar_field = false;
    vec_field = false;
    vec_len = [];

    % look at a few elements to decide
    probe_idx = sub2ind(sz, min(1,sz(1)), min(1,sz(2)));
    for i=1:numel(fns)
        try
            v = D(probe_idx).(fns{i});
        catch
            continue
        end
        if isnumeric(v) && isscalar(v)
            cand = fns{i}; scalar_field = true; break;
        elseif isnumeric(v) && isvector(v)
            cand = fns{i}; vec_field = true; vec_len = numel(v); break;
        end
    end
    if isempty(cand)
        error('No numeric field found in struct "%s" of %s. Fields: %s', vname, fname, strjoin(fns, ', '));
    end

    if numel(sz)==2 && all(sz>1) && scalar_field
        % Case A: struct Nch x Nms with scalar entries per element
        Nch = sz(1); Nms = sz(2);
        tmp = arrayfun(@(s) double(s.(cand)), D);
        Maps = reshape(tmp, Nch, Nms);
    elseif isvector(D) && vec_field
        % Case B: struct 1 x Nms (or Nms x 1) with vector field of length Nch
        Nms = numel(D);
        Nch = vec_len;
        Maps = zeros(Nch, Nms);
        for j=1:Nms
            v = D(j).(cand);
            if numel(v) ~= Nch
                error('Field "%s" length varies across maps (expected %d, got %d at j=%d).', cand, Nch, numel(v), j);
            end
            Maps(:,j) = double(v(:));
        end
    else
        % Last attempt: look for a field that itself is a numeric 2D array
        pulled = false;
        for i=1:numel(fns)
            V = D(1).(fns{i});
            if isnumeric(V) && ndims(V)==2
                Maps = double(V);
                pulled = true;
                break;
            end
        end
        if ~pulled
            error('Unrecognised struct layout for "%s" in %s. Size=%s; fields=%s', ...
                  vname, fname, mat2str(sz), strjoin(fns, ', '));
        end
    end
elseif iscell(D)
    % cell array: try to stack column-wise
    try
        Nms = numel(D);
        Nch = numel(D{1});
        Maps = zeros(Nch, Nms);
        for j=1:Nms, Maps(:,j) = double(D{j}(:)); end
    catch
        error('Cell array "%s" has unexpected contents in %s.', vname, fname);
    end
else
    error('Unsupported type for "%s": %s', vname, class(D));
end

% ---- channels (best-effort) ----
Channels = [];
candCh = {'Channels','channels','Chanlocs','chanlocs','Channel','EEG','eeg'};
for k=1:numel(candCh)
    if isfield(S, candCh{k})
        Channels = S.(candCh{k});
        break;
    end
end
% If Channels is EEGLAB-like, try to expose labels
if isstruct(Channels)
    fnsC = fieldnames(Channels);
    if ismember('labels', fnsC)
        try
            chnames = arrayfun(@(c) string(c.labels), Channels, 'UniformOutput', false);
            Channels = chnames(:);
        catch
            % leave as-is
        end
    elseif ismember('Name', fnsC)
        try
            chnames = arrayfun(@(c) string(c.Name), Channels, 'UniformOutput', false);
            Channels = chnames(:);
        catch
        end
    end
end

% ---- normalise columns to unit L2 ----
Maps = bsxfun(@rdivide, Maps, max(vecnorm(Maps,2,1), eps));

end


function [X, labels, w, seg_ids] = synth_ms_eeg(Maps, T_target, seg_len_range, smooth_sigma)
% Sticky microstate sequence with smooth weights in [-1,1]
% Maps: Nch x K_true
[Nch, K_true] = size(Maps);
labels = zeros(1,0);
w = zeros(1,0);
seg_ids = zeros(1,0);
t = 0; seg_id = 0;
state = randi(K_true);  % initial state
while t < T_target
    seg_len = randi(seg_len_range);
    seg_id = seg_id + 1;
    seg_labels = state * ones(1, seg_len);
    % smooth random weight in [-1,1] for this activation
    y = randn(1, seg_len);
    y = smooth_conv(y, smooth_sigma);
    % rescale to [-1,1]
    if max(abs(y)) < eps, y = zeros(size(y)); else, y = y / max(abs(y)); end
    % mild drift to ensure range span across longer segments
    y = 0.9*y + 0.1*linspace(-1,1,seg_len);
    y = max(min(y,1),-1);

    labels = [labels, seg_labels];            %#ok<AGROW>
    w      = [w, y];                          %#ok<AGROW>
    seg_ids= [seg_ids, seg_id*ones(1,seg_len)]; %#ok<AGROW>
    t = numel(labels);

    % decide next state (avoid immediate repeats with small prob to switch)
    if rand < 0.3
        nxt = randi(K_true-1);
        if nxt >= state, nxt = nxt + 1; end
        state = nxt;
    end
end
labels = labels(1:T_target);
w      = w(1:T_target);
seg_ids= seg_ids(1:T_target);

% generate EEG
X = zeros(Nch, T_target);
for t=1:T_target
    k = labels(t);
    X(:,t) = Maps(:,k) * w(t);
end
end

function y = smooth_conv(x, sigma)
% Gaussian-ish smoothing via FIR box cascade (no Toolboxes).
L = max(3, 2*ceil(3*sigma)+1);
h = ones(1, L) / L;
y = conv(x, h, 'same');
% cascade twice for better low-pass
y = conv(y, h, 'same');
end

function idx = select_gfp_peaks_from_weight(w, top_frac)
% "GFP peaks" proxy: take timepoints with largest |w|
T = numel(w);
k = max(1, round(top_frac * T));
[~, ord] = sort(abs(w), 'descend');
idx = sort(ord(1:k));
end

function X2 = polarity_align_by_pca(X)
% Flip column signs so the first PC has positive projection.
% Avoids ±map duplications in clustering.
[U,~,~] = svd((X - mean(X,2)), 'econ');
v = U(:,1);
s = sign(v' * X);
s(s==0) = 1;
X2 = X .* s;
end

function e = centroid_rmse_if_Kmatch(mu_true, X, labels, K_sel)
if K_sel ~= size(mu_true,2)
    e = NaN; return;
end
mu_hat = estimate_centroids_from_labels(X, labels, K_sel);
% align polarity similarly to data
mu_hat = polarity_align_by_pca(mu_hat);
e = rmse_centres(mu_true, mu_hat, match_components(mu_true, mu_hat));
end

function e = centroid_rmse_vs_truth(mu_true, M)
[mu_hat, ~] = cent_cov_from_model(M);
mu_hat = polarity_align_by_pca(mu_hat);
e = rmse_centres(mu_true, mu_hat, match_components(mu_true, mu_hat));
end

% ==================== your existing toolbox-free bits ====================

function [K_best, labels_best, C_best] = kmeans_silhouette_select(X, Kgrid)
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
        for t=1:T
            x = X(:,t); dmin = inf; imin = 1;
            for k=1:K
                d = sum((x - C(:,k)).^2);
                if d<dmin, dmin=d; imin=k; end
            end
            idx(t) = imin;
        end
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

function mu = estimate_centroids_from_labels(X, labels, K)
N = size(X,1);
mu = zeros(N,K);
for k=1:K
    idx = (labels==k);
    if any(idx), mu(:,k) = mean(X(:,idx),2); else, mu(:,k) = 0; end
end
end

function prior = robust_niw_prior_from_data(X, varargin)
[N,~] = size(X);
p.nu0   = N + 2;
p.beta0 = 1.0;
p = parse_opts(p, varargin{:});

S = cov(X'); 
if ~isfinite(sum(S(:))) || any(isnan(S(:)))
    Xc = X - mean(X,2);
    S = (Xc*Xc')/max(size(X,2)-1,1);
end
S = (S+S')/2;

diagS = diag(diag(S));
lambda = ledoit_wolf_lambda(X, S, diagS);
lambda = min(max(lambda,0),1);
if any(strcmpi(varargin(1:2:end),'lambda'))
    lambda = p.lambda;
end
Ssh = (1 - lambda)*S + lambda*diagS;

prior = struct();
prior.m0   = mean(X,2);
prior.beta0= p.beta0;
prior.nu0  = p.nu0;
prior.W0   = inv(Ssh) * (prior.nu0 - N - 1);
prior.W0   = (prior.W0 + prior.W0')/2;

    function val = parse_opts(defs, varargin)
        val = defs;
        for i=1:2:numel(varargin)
            val.(varargin{i}) = varargin{i+1};
        end
    end
end

function lambda = ledoit_wolf_lambda(X, S, T)
[N,Tn] = size(X);
Xc = X - mean(X,2);
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

function K_best = kfold_predictive_selection(X, Kgrid, opts, kfold)
% Stratified, NIW t-predictive scorer with mild weight tempering (tau=0.9)
[N,T] = size(X);
if T < kfold, kfold = T; end

G = min( max(3, round(sqrt(T/50))), 20 );
[~, Cg] = lloyd_kmeans(X, G, 3, 50);
groups = zeros(1,T);
for t=1:T
    x = X(:,t); dmin=inf; g=1;
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

pred_scores = -inf(1,numel(Kgrid));
parfor ki = 1:numel(Kgrid)
    Kcand = Kgrid(ki);
    opts_local = opts; opts_local.Kgrid = Kcand;
    score_sum = 0; valid_folds = 0;
    for f=1:kfold
        test_idx  = folds{f};
        if isempty(test_idx), continue; end
        train_idx = setdiff(1:T, test_idx);
        if numel(train_idx) < max(10, Kcand+2), continue; end
        Xtr = X(:,train_idx); Xte = X(:,test_idx);
        res_tr = vb_gmm_freeenergy(Xtr, Kcand, opts_local);
        M = res_tr.model;
        if ~isfield(M,'m') || isempty(M.m) || numel(M.m)<1, continue; end
        lp = log_pred_student_t_mixture(Xte, M.m, M.beta, M.W, M.nu, M.alpha, opts.diag_cov);
        score_sum = score_sum + sum(lp); valid_folds = valid_folds + 1;
    end
    if valid_folds > 0, pred_scores(ki) = score_sum / valid_folds; else, pred_scores(ki) = -inf; end
end
mx = max(pred_scores);
cands = find(abs(pred_scores - mx) < 1e-10);
K_best = min(Kgrid(cands));
end

function lp = log_pred_student_t_mixture(X, m, beta, W, nu, alpha, diag_cov)
[N,T] = size(X);
K = numel(m);
tau = 0.9;   % mild tempering for stability
aw = alpha(:).^tau; aw = aw / sum(aw);
logcomp = -inf(K,T);
for k=1:K
    df = max(nu(k) - N + 1, 1e-6);
    if diag_cov
        Wk = diag(max(diag(W{k}), 1e-12));
    else
        Wk = (W{k} + W{k}')/2;
    end
    S = ((beta(k)+1)/(beta(k)*df)) * inv(Wk);
    S = (S+S')/2;
    U = chol(S + 1e-12*eye(N),'upper');
    logdetS = 2*sum(log(diag(U)));
    Q = U \ (X - m{k});
    Q = sum(Q.^2,1);
    logZ = gammaln(0.5*(df+N)) - gammaln(0.5*df) - 0.5*N*log(df*pi) - 0.5*logdetS;
    logcomp(k,:) = log(aw(k)+eps) + (logZ - 0.5*(df+N)*log(1 + Q/df));
end
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
Lt = labels_true(:); Lp = labels_pred(:);
[~,~,Lt] = unique(Lt,'stable');
[~,~,Lp] = unique(Lp,'stable');
nt = max(Lt); np = max(Lp); n = numel(Lt);
M = zeros(nt,np);
for i=1:n, M(Lt(i), Lp(i)) = M(Lt(i), Lp(i)) + 1; end
a = sum(M,2); b = sum(M,1);
sumC2 = sum(sum(M.*(M-1)))/2;
sumA2 = sum(a.*(a-1))/2;
sumB2 = sum(b.*(b-1))/2;
denom = max(n*(n-1)/2, 1);
expected = (sumA2 * sumB2) / denom;
maxidx  = 0.5 * (sumA2 + sumB2);
ari = (sumC2 - expected) / max(maxidx - expected, eps);
end
