function Results = fit_microstate_spm_kmeans(Sim, K_candidates, criterion)
% FIT_MICROSTATE_SPM_KMEANS: K-means as limit case of GMM using SPM
%
% Uses SPM's mixture model with isotropic covariance and tiny variance to approximate hard K-means.

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'silhouette'; end

    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('SPM K-Means Microstate Fitting\n');
    fprintf('(GMM with Isotropic Covariance, sigma^2 -> 0)\n');
    fprintf('========================================\n');
    fprintf('Computing ALL criteria (will apply %s)\n', criterion);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);
    else
        fprintf('Real data mode (no ground truth)\n');
    end

    if ~exist('spm_mix', 'file')
        error('SPM spm_mix not found in MATLAB path');
    end

    util = microstate_utilities();

    % Preprocessing
    fprintf('1. Preprocessing...\n');
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original, preprocessing_info] = util.preprocess_maps(Sim);
    fprintf('   Extracted %d GFP peak maps (%d channels)\n', n_maps, C_dims);

    features = maps_norm;
    fprintf('2. Using normalized maps as features: %d dims\n', size(features,2));

    if ~isfield(Sim, 'n_restarts'), R = 20; else, R = Sim.n_restarts; end

    % Fit models across K
    fprintf('3. Fitting SPM GMM models (isotropic, sigma^2=1e-6)...\n');
    nK = numel(K_candidates);
    gev_vals = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    within_ss = zeros(nK, 1);
    free_energy_vals = zeros(nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    exp_var_all = zeros(nK, 1);

    variance_small = 1e-4; % slightly larger to avoid degenerate covariances

    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('   K=%d... ', K);
        try
            result = spm_mix(features, K, 0);
            for k = 1:K
                D = size(features, 2);
                result.state(k).C = variance_small * eye(D);
            end

            labels = assign_samples_hard(features, result);
            centers = recover_centers_from_labels(maps_norm, labels, K);

            % Polarity-aware warm start to sharpen centers and recover empty clusters
            [labels, centers] = polarity_kmeans_warmstart(maps_norm, K, 10, max(10, R));

            centers_all{iK} = centers;
            labels_all{iK} = labels;

            sim = abs(maps_original * centers');
            [max_sim, ~] = max(sim, [], 2);
            gfp_squared = sum(maps_original.^2, 2);
            gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);

            silhouette_vals(iK) = silhouette_microstatelab(maps_norm, labels, centers);

            wss = 0;
            for k = 1:K
                cluster_maps = maps_norm(labels == k, :);
                if ~isempty(cluster_maps)
                    dists = 1 - abs(cluster_maps * centers(k, :)');
                    wss = wss + sum(dists.^2);
                end
            end
            within_ss(iK) = wss;

            if isfield(result, 'fm') && ~isempty(result.fm)
                free_energy_vals(iK) = result.fm;
            else
                free_energy_vals(iK) = -Inf;
            end

            exp_var_per_cluster = zeros(K, 1);
            for k = 1:K
                cluster_idx = find(labels == k);
                if ~isempty(cluster_idx)
                    cluster_maps = maps_original(cluster_idx, :);
                    center_orig = centers(k, :);
                    sim_k = abs(cluster_maps * center_orig');
                    exp_var_per_cluster(k) = mean(sim_k);
                end
            end
            exp_var_all(iK) = mean(exp_var_per_cluster);
            fprintf('OK\n');
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            gev_vals(iK) = 0;
            silhouette_vals(iK) = -1;
            within_ss(iK) = Inf;
            exp_var_all(iK) = 0;
            free_energy_vals(iK) = -Inf;
            centers_all{iK} = [];
            labels_all{iK} = [];
        end
    end

    % Select K based on criterion
    switch lower(criterion)
        case 'silhouette'
            [K_est, best_score] = select_K_from_silhouette_koenig(silhouette_vals, K_candidates);
        case 'gev'
            [best_score, idx] = max(gev_vals);
            K_est = K_candidates(idx);
        case 'elbow'
            [K_est, best_score] = select_K_from_elbow(within_ss, K_candidates);
        case 'elbow_sil_combined'
            [K_est, best_score] = select_K_combined_elbow_silhouette(free_energy_vals, K_candidates, silhouette_vals);
        case 'free_energy'
            [best_score, idx] = max(free_energy_vals);
            K_est = K_candidates(idx);
        otherwise
            [K_est, best_score] = select_K_from_silhouette_koenig(silhouette_vals, K_candidates);
    end
    idx = find(K_candidates == K_est, 1);
    if isempty(idx), idx = 1; end

    centers = centers_all{idx};
    labels = labels_all{idx};

    fprintf('3b. Selected K=%d via %s (score=%.4f)\n', K_est, criterion, best_score);

    % Recovery
    fprintf('4. Computing map recovery...\n');
    if isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true)
        true_maps_norm = util.normalize_maps(Sim.maps_true);
        recovery_metrics = microstate_partial_alignment(true_maps_norm, centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
        fprintf('Map Recovery: Matched=%d, F1=%.4f, Sens=%.4f, Prec=%.4f\n\n', ...
            recovery_metrics.n_matched, recovery_metrics.f1_score, ...
            recovery_metrics.sensitivity, recovery_metrics.precision);
    else
        recovery_metrics = struct( ...
            'K_true', NaN, ...
            'K_estimated', K_est, ...
            'n_matched', 0, ...
            'mean_recovery_matched', NaN, ...
            'mean_recovery_padded', NaN, ...
            'sensitivity', NaN, ...
            'precision', NaN, ...
            'f1_score', NaN, ...
            'match_similarities', []);
        true_maps_norm = [];
        fprintf('No ground truth available - skipping recovery metrics\n\n');
    end

    runtime = toc(t_start);

    if isfield(Sim, 'K_true'), K_true_val = Sim.K_true; else, K_true_val = NaN; end
    if isfield(Sim, 'SNR_dB'), SNR_dB_val = Sim.SNR_dB; else, SNR_dB_val = NaN; end
    if isfield(Sim, 'duration_s'), duration_s_val = Sim.duration_s; else, duration_s_val = NaN; end

    Results = struct( ...
        'method', 'spm_kmeans', ...
        'criterion', criterion, ...
        'K_true', K_true_val, ...
        'K_estimated', K_est, ...
        'K_candidates', K_candidates, ...
        'SNR_dB', SNR_dB_val, ...
        'duration_s', duration_s_val, ...
        'n_maps', n_maps, ...
        'centers', centers, ...
        'maps_true', true_maps_norm, ...
        'labels', labels, ...
        'gev_vals', gev_vals, ...
        'silhouette_vals', silhouette_vals, ...
        'free_energy_vals', free_energy_vals, ...
        'within_ss', within_ss, ...
        'explained_variance', exp_var_all, ...
        'best_criterion_value', best_score, ...
        'maps_nc', maps_norm, ...
        'idx_peaks', idx_peaks, ...
        'gfp_vec', gfp_vec, ...
        'preprocessing_info', preprocessing_info, ...
        'recovery_metrics', recovery_metrics, ...
        'mean_recovery', recovery_metrics.mean_recovery_matched, ...
        'recovery_corr', recovery_metrics.match_similarities, ...
        'avg_recovery_per_state', recovery_metrics.mean_recovery_padded, ...
        'valid_fit', true, ...
        'runtime', runtime);
end

% ======================== K SELECTION HELPERS ========================

function [K_est, score] = select_K_from_silhouette_koenig(sil_vals, K_candidates)
    % KOENIG METHOD: Simple maximum without 2nd derivatives
    % Find K with highest silhouette score
    
    [best_score, idx] = max(sil_vals);
    K_est = K_candidates(idx);
    score = best_score;
    
    fprintf('  Silhouette (Koenig): selected K=%d (score=%.4f)\n', K_est, score);
end

function [K_est, score] = select_K_from_elbow(wss_vals, K_candidates)
    n = length(wss_vals);
    if n < 4
        [~, idx] = min(wss_vals);
        K_est = K_candidates(idx);
        score = wss_vals(idx);
        return;
    end
    
    wss_norm = (wss_vals - min(wss_vals)) / (max(wss_vals) - min(wss_vals) + eps);
    
    curvature = zeros(n, 1);
    for i = 2:(n-1)
        dy1 = wss_norm(i) - wss_norm(i-1);
        dy2 = wss_norm(i+1) - wss_norm(i);
        curvature(i) = abs(dy2 - dy1);
    end
    
    [~, idx] = max(curvature(2:(n-1)));
    idx = idx + 1;
    
    K_est = K_candidates(idx);
    score = curvature(idx);
    
    fprintf('  Elbow: K=%d (WSS=%.1f, curvature=%.4f)\n', K_est, wss_vals(idx), score);
end

function [K_est, score] = select_K_combined_elbow_silhouette(fe_vals, K_candidates, sil_vals)
    % Combined heuristic: elbow on free-energy trend + silhouette tie-break
    fe_valid = fe_vals;
    K_valid = K_candidates;
    sil_valid = sil_vals;
    
    fe_norm = (fe_valid - min(fe_valid)) / (max(fe_valid) - min(fe_valid) + eps);
    k_norm = (K_valid - min(K_valid)) / (max(K_valid) - min(K_valid) + eps);
    
    elbow_scores = zeros(size(fe_norm));
    for i = 2:length(fe_norm)-1
        p1 = [k_norm(1), fe_norm(1)];
        p2 = [k_norm(end), fe_norm(end)];
        p = [k_norm(i), fe_norm(i)];
        elbow_scores(i) = distance_from_line(p1, p2, p);
    end
    
    elbow_scores(1) = 0;
    elbow_scores(end) = 0;
    
    [max_elbow, idx_elbow] = max(elbow_scores);
    cand_idx = find(elbow_scores >= 0.9 * max_elbow);
    
    sil_subset = sil_valid(cand_idx);
    [best_sil, best_sub_idx] = max(sil_subset);
    
    best_idx = cand_idx(best_sub_idx);
    K_est = K_valid(best_idx);
    score = (max_elbow + best_sil) / 2;
    
    fprintf('  Combined elbow+silhouette: K=%d (elbow=%.4f, sil=%.4f)\n', K_est, max_elbow, best_sil);
end

function d = distance_from_line(p1, p2, p)
    if all(p1 == p2)
        d = 0;
        return;
    end
    num = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + p2(1)*p1(2) - p2(2)*p1(1));
    den = sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2);
    d = num / (den + eps);
end

function [labels_best, centers_best] = polarity_kmeans_warmstart(maps_norm, K, n_iter, n_restarts)
    % Simple polarity-aware K-means with random map seeds
    if nargin < 3, n_iter = 10; end
    if nargin < 4, n_restarts = 10; end
    N = size(maps_norm, 1);
    best_inertia = Inf;
    labels_best = [];
    centers_best = [];
    for r = 1:n_restarts
        seed_idx = randperm(N, K);
        centers = maps_norm(seed_idx, :);
        for it = 1:n_iter
            % Assignment with polarity (absolute cosine similarity)
            sims = abs(maps_norm * centers');
            [~, labels] = max(sims, [], 2);
            % Update centers
            for k = 1:K
                idx = find(labels == k);
                if isempty(idx)
                    centers(k, :) = maps_norm(randi(N), :);
                    continue;
                end
                cluster = maps_norm(idx, :);
                ref = cluster(1, :);
                aligned = cluster;
                flips = (cluster * ref') < 0;
                aligned(flips, :) = -aligned(flips, :);
                c = mean(aligned, 1);
                c = c - mean(c);
                cn = norm(c);
                if cn > 0, c = c / cn; end
                centers(k, :) = c;
            end
        end
        inertia = sum(1 - abs(maps_norm * centers(labels, :)'), 'all');
        if inertia < best_inertia
            best_inertia = inertia;
            labels_best = labels;
            centers_best = centers;
        end
    end
    if isempty(labels_best)
        labels_best = ones(N,1);
        centers_best = maps_norm(1:K, :);
    end
end

% ======================== GMM HELPERS ========================

function labels = assign_samples_hard(X, vbmix)
% Assign samples to mixture components using hard assignments (argmax)
    
    [N, D] = size(X);
    K = vbmix.m;
    log_prob = zeros(N, K);
    
    for k = 1:K
        m = vbmix.state(k).m(:)';
        C = vbmix.state(k).C;
        
        if length(m) ~= D
            m = m(1:min(D, length(m)));
            if size(C, 1) ~= D
                C = C(1:min(D, size(C, 1)), 1:min(D, size(C, 2)));
            end
        end
        
        if isfield(vbmix.state(k), 'prior')
            prior = vbmix.state(k).prior;
        elseif isfield(vbmix, 'priors') && length(vbmix.priors) >= k
            prior = vbmix.priors(k);
        else
            prior = 1 / K;
        end
        
        log_prob(:, k) = log(prior + eps) + log_mvnpdf(X, m, C);
    end
    
    % Hard assignment: argmax (K-means-like)
    [~, labels] = max(log_prob, [], 2);
end

function log_p = log_mvnpdf(X, mu, Sigma)
% Log multivariate normal PDF
    
    [N, D] = size(X);
    X_centered = X - repmat(mu, N, 1);
    
    [U, S, ~] = svd(Sigma);
    s = diag(S);
    s(s < 1e-10) = 1e-10;
    Sigma_reg = U * diag(s) * U';
    
    try
        L = chol(Sigma_reg, 'lower');
        z = L \ X_centered';
        maha = sum(z.^2, 1)';
    catch
        Sigma_inv = pinv(Sigma_reg);
        maha = sum((X_centered * Sigma_inv) .* X_centered, 2);
    end
    
    log_det = sum(log(s));
    log_p = -0.5 * (D*log(2*pi) + log_det + maha);
end

function centers = recover_centers_from_labels(maps, labels, K)
% Recover cluster centers with polarity alignment (KOENIG METHOD)
    
    [N, D] = size(maps);
    centers = zeros(K, D);
    
    for k = 1:K
        idx = find(labels == k);
        
        if isempty(idx)
            centers(k, :) = maps(randi(N), :);
            continue;
        end
        
        ref = maps(idx(1), :);
        
        aligned = zeros(length(idx), D);
        for i = 1:length(idx)
            m = maps(idx(i), :);
            if (m * ref') < 0
                aligned(i, :) = -m;
            else
                aligned(i, :) = m;
            end
        end
        
        centers(k, :) = mean(aligned, 1);
        centers(k, :) = centers(k, :) - mean(centers(k, :));
        centers(k, :) = centers(k, :) / (norm(centers(k, :)) + eps);
    end
end

% ======================== IMPORTED SILHOUETTE METHOD ========================

function sil = silhouette_microstatelab(X, labels, centers)
    % SILHOUETTE_MICROSTATELAB: Silhouette score using Koenig's method
    % Computes polarity-insensitive (absolute) cosine-based silhouette coefficient
    %
    % INPUTS:
    %   X       - Data matrix (N_samples x N_channels)
    %   labels  - Cluster assignments (N_samples x 1)
    %   centers - Cluster centers (K_clusters x N_channels)
    %
    % OUTPUT:
    %   sil     - Average silhouette coefficient
    
    K = size(centers, 1);
    N = size(X, 1);
    
    if K < 2 || N < 2
        sil = 0;
        return;
    end
    
    sil_vals = zeros(N, 1);
    
    for i = 1:N
        k = labels(i);
        
        % Intra-cluster: mean distance to other points in same cluster
        same_cluster = find(labels == k);
        if numel(same_cluster) > 1
            % Distance = 1 - |cosine_similarity| (polarity-insensitive)
            similarities = abs(X(same_cluster, :) * X(i, :)');
            a = mean(1 - similarities(similarities < 1));  % Exclude self-similarity (=1)
            if isnan(a) || a < 0, a = 0; end
        else
            a = 0;
        end
        
        % Inter-cluster: mean distance to nearest other cluster
        b = Inf;
        for kk = 1:K
            if kk == k, continue; end
            other_cluster = find(labels == kk);
            if ~isempty(other_cluster)
                similarities = abs(X(other_cluster, :) * X(i, :)');
                b_kk = mean(1 - similarities);
                b = min(b, b_kk);
            end
        end
        
        if isinf(b) || isnan(b)
            b = a;
        end
        
        if a == 0 && b == 0
            sil_vals(i) = 0;
        else
            sil_vals(i) = (b - a) / (max(a, b) + eps);
        end
    end
    
    sil = mean(sil_vals(~isnan(sil_vals) & ~isinf(sil_vals)));
    if isnan(sil) || isinf(sil)
        sil = 0;
    end
end
