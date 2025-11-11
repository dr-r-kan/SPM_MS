function Results = fit_microstate_spm_vb(Sim, K_candidates, criterion)
% FIT_MICROSTATE_SPM_VB: VB microstate fitting using SPM
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Vector of K values to test
%   criterion    - 'silhouette', 'free_energy', or 'elbow_sil_combined'
%
% OUTPUTS:
%   Results      - Complete results with recovery metrics

    if nargin < 2
        K_candidates = 2:10;
    end
    if nargin < 3
        criterion = 'silhouette';
    end
    
    % GEV criterion is only valid for k-means methods, not for VB/GMM
    if strcmp(criterion, 'gev')
        error('GEV criterion is not supported for spm_vb method. Use ''silhouette'', ''free_energy'', ''elbow'', or ''elbow_sil_combined'' instead.');
    end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('Microstate VB Fitting (SPM)\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);
    else
        fprintf('Real data mode (no ground truth)\n');
    end

    if ~exist('spm_mix', 'file')
        error('SPM spm_mix not found in MATLAB path');
    end

    % Load shared utilities
    util = microstate_utilities_SHARED();

    % Preprocessing (using shared utility)
    fprintf('1. Preprocessing...\n');
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = util.preprocess_maps(Sim);
    fprintf('   Extracted %d GFP peak maps (%d channels)\n', n_maps, C_dims);
    
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original] = util.preprocess_maps(Sim);

    % Dimensionality reduction
    fprintf('2. Dimensionality reduction...\n');
    [coeff, score, latent] = pca(maps_norm);
    var_explained = cumsum(latent) / sum(latent);
    n_dims = find(var_explained >= 0.95, 1, 'first');
    n_dims = min(n_dims, 20);
    n_dims = max(n_dims, 5);
    features = score(:, 1:n_dims);
    fprintf('   Using %d PCA dimensions (%.1f%% variance)\n', n_dims, var_explained(n_dims)*100);

    % Fit VB GMM
    fprintf('3. Fitting VB GMM models using SPM...\n');
    nK = numel(K_candidates);
    free_energy = zeros(nK, 1);
    silhouette_score = zeros(nK, 1);
    within_ss = zeros(nK, 1);
    gev_vals = zeros(nK, 1);
    vbmix = cell(nK, 1);

    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('   K=%d... ', K);
        
        try
            result = spm_mix(features, K, 0);
            
            if isnan(result.fm) || isinf(result.fm) || result.fm == 0
                fprintf('Invalid\n');
                free_energy(iK) = -Inf;
                silhouette_score(iK) = -1;
                within_ss(iK) = Inf;
                gev_vals(iK) = 0;
            else
                free_energy(iK) = result.fm;
                vbmix{iK} = result;
                
                labels = assign_samples(features, result);
                
                % ✅ Recover centers to compute GEV
                temp_centers = recover_centers_from_labels(maps_norm, labels, K);
                sim = abs(maps_original * temp_centers');
                [max_sim, ~] = max(sim, [], 2);
                gfp_squared = sum(maps_original.^2, 2);
                gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);

                % Compute silhouette using Euclidean distance in PCA space
                sil = compute_silhouette_euclidean(features, labels);
                silhouette_score(iK) = sil;
                
                % Compute within-cluster sum of squares for elbow method
                wss = compute_within_ss(features, labels, K);
                within_ss(iK) = wss;
                
                fprintf('F=%.1f, Sil=%.3f, WSS=%.1f\n', result.fm, sil, wss);
            end
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            free_energy(iK) = -Inf;
            silhouette_score(iK) = -1;
            within_ss(iK) = Inf;
            gev_vals(iK) = 0;
            within_ss(iK) = Inf;
        end
    end

    % Model selection
    valid_idx = ~isinf(free_energy) & free_energy ~= 0;
    
    if ~any(valid_idx)
        warning('All fits failed!');
        Results = create_empty_results(Sim, K_candidates, n_maps, C_dims, criterion);
        return;
    end
    
    fe_valid = free_energy(valid_idx);
    K_valid = K_candidates(valid_idx);
    sil_valid = silhouette_score(valid_idx);
    wss_valid = within_ss(valid_idx);
    
    fprintf('\nModel selection using: %s\n', criterion);
    
    % Select based on criterion (KOENIG METHOD - no 2nd derivatives)
    switch criterion
        case 'silhouette'
            % KOENIG METHOD: Simple maximum silhouette
            [best_score, idx] = max(sil_valid);
            K_estimated = K_valid(idx);
            best_idx = find(valid_idx);
            best_idx = best_idx(idx);
            fprintf('  Silhouette (Koenig): selected K=%d (score=%.4f)\n', K_estimated, best_score);
            
        case 'free_energy'
            % Maximum free energy
            [best_score, idx] = max(fe_valid);
            K_estimated = K_valid(idx);
            best_idx = find(valid_idx);
            best_idx = best_idx(idx);
            fprintf('  Free Energy: selected K=%d (FE=%.1f)\n', K_estimated, best_score);
            
        case 'elbow'
            % Elbow method on within-cluster SS
            [K_estimated, best_score] = select_K_from_elbow_spm(wss_valid, K_valid);
            best_idx = find(valid_idx);
            best_idx_local = find(K_valid == K_estimated, 1);
            best_idx = best_idx(best_idx_local);
            
        case 'elbow_sil_combined'
            % Original combined approach (kept for backwards compatibility)
            [K_estimated, best_score] = select_K_combined_elbow_silhouette(fe_valid, K_valid, sil_valid);
            best_idx = find(valid_idx);
            best_idx_local = find(K_valid == K_estimated, 1);
            best_idx = best_idx(best_idx_local);
            
        otherwise
            error('Unknown criterion: %s', criterion);
    end
    
    K_estimated = K_candidates(best_idx);
    best_mix = vbmix{best_idx};
    
    fprintf('Best K: %d\n', K_estimated);
    if isfield(Sim, 'K_true') && ~isnan(Sim.K_true)
        fprintf('True K: %d\n\n', Sim.K_true);
    else
        fprintf('\n');
    end

    % Recover centers
    fprintf('4. Recovering microstate topographies...\n');
    labels = assign_samples(features, best_mix);
    centers = recover_centers_from_labels(maps_norm, labels, K_estimated);
    fprintf('   Recovered %d microstate centers\n', K_estimated);

    % Map recovery (only if ground truth is available)
    fprintf('5. Computing map recovery...\n');
    if isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true)
        true_maps_norm = util.normalize_maps(Sim.maps_true);
        recovery_metrics = microstate_partial_alignment(true_maps_norm, centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
        
        fprintf('Map Recovery Analysis:\n');
        fprintf('  K true: %d, K estimated: %d\n', recovery_metrics.K_true, recovery_metrics.K_estimated);
        fprintf('  Matched: %d, F1: %.4f\n', recovery_metrics.n_matched, recovery_metrics.f1_score);
        fprintf('  Sensitivity: %.4f, Precision: %.4f\n\n', ...
            recovery_metrics.sensitivity, recovery_metrics.precision);
    else
        % No ground truth - create empty recovery metrics
        recovery_metrics = struct(...
            'K_true', NaN, ...
            'K_estimated', K_estimated, ...
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

    % Return results
    if isfield(Sim, 'K_true')
        K_true_val = Sim.K_true;
    else
        K_true_val = NaN;
    end
    if isfield(Sim, 'SNR_dB')
        SNR_dB_val = Sim.SNR_dB;
    else
        SNR_dB_val = NaN;
    end
    if isfield(Sim, 'duration_s')
        duration_s_val = Sim.duration_s;
    else
        duration_s_val = NaN;
    end
    
    Results = struct( ...
    'method', 'spm_vb', ...
    'criterion', criterion, ...
    'K_true', K_true_val, ...
    'K_estimated', K_estimated, ...
    'K_candidates', K_candidates, ...
    'SNR_dB', SNR_dB_val, ...
    'duration_s', duration_s_val, ...
    'n_maps', n_maps, ...
    'centers', centers, ...
    'maps_true', true_maps_norm, ...
    'labels', labels, ...
    'free_energy', free_energy, ...
    'silhouette_vals', silhouette_score, ...  
    'free_energy_vals', free_energy, ...      
    'within_ss', within_ss, ...            
    'gev_vals', gev_vals, ...             
    'best_criterion_value', best_score, ...
    'maps_nc', maps_norm, ...
    'idx_peaks', idx_peaks, ...
    'gfp_vec', gfp_vec, ...
    'recovery_metrics', recovery_metrics, ...
    'mean_recovery', recovery_metrics.mean_recovery_matched, ...
    'recovery_corr', recovery_metrics.match_similarities, ...
    'avg_recovery_per_state', recovery_metrics.mean_recovery_padded, ...
    'valid_fit', true, ...
    'runtime', runtime);
end

% ======================== K SELECTION HELPERS ========================

function [K_est, score] = select_K_from_elbow_spm(wss_vals, K_candidates)
    % Elbow detection for within-cluster sum of squares
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

function [K_est, score] = select_K_combined_elbow_silhouette(fe_valid, K_valid, sil_valid)
    % Original combined elbow + silhouette heuristic (backwards compatibility)
    
    fe_norm = (fe_valid - min(fe_valid)) / (max(fe_valid) - min(fe_valid) + eps);
    k_norm = (K_valid - min(K_valid)) / (max(K_valid) - min(K_valid) + eps);
    
    elbow_scores = zeros(size(fe_norm));
    for i = 2:length(fe_norm)-1
        p1 = [k_norm(1), fe_norm(1)];
        p2 = [k_norm(end), fe_norm(end)];
        p = [k_norm(i), fe_norm(i)];
        
        elbow_scores(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + ...
            p2(1)*p1(2) - p2(2)*p1(1)) / ...
            sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    
    [~, elbow_idx] = max(elbow_scores);
    K_elbow = K_valid(elbow_idx);
    
    % Combined score
    combined_score = zeros(length(K_valid), 1);
    for i = 1:length(K_valid)
        elbow_penalty = exp(-abs(K_valid(i) - K_elbow));
        sil_bonus = (sil_valid(i) + 1) / 2;
        combined_score(i) = 0.6 * elbow_penalty + 0.4 * sil_bonus;
    end
    
    [score, best_idx] = max(combined_score);
    K_est = K_valid(best_idx);
    
    fprintf('  Combined (Elbow+Silhouette): selected K=%d (score=%.4f, elbow K=%d)\n', K_est, score, K_elbow);
end

% ======================== HELPERS (LOCAL ONLY) ========================

function labels = assign_samples(X, vbmix)
% Assign samples to mixture components
    
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

function sil = compute_silhouette_euclidean(X, labels)
% Compute silhouette score using Euclidean distance (appropriate for PCA space)
    
    K = max(labels);
    N = size(X, 1);
    
    if K < 2
        sil = 0;
        return;
    end
    
    sil_vals = zeros(N, 1);
    
    for i = 1:N
        k = labels(i);
        
        same_cluster = find(labels == k);
        if numel(same_cluster) > 1
            dists = sqrt(sum((X(same_cluster, :) - X(i, :)).^2, 2));
            a = mean(dists(dists > 0));
        else
            a = 0;
        end
        
        b = Inf;
        for kk = 1:K
            if kk == k, continue; end
            other_cluster = find(labels == kk);
            if ~isempty(other_cluster)
                dists = sqrt(sum((X(other_cluster, :) - X(i, :)).^2, 2));
                b = min(b, mean(dists));
            end
        end
        
        if a == 0 && b == Inf
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

function wss = compute_within_ss(X, labels, K)
% Compute within-cluster sum of squares
    
    wss = 0;
    for k = 1:K
        cluster_idx = find(labels == k);
        if ~isempty(cluster_idx)
            cluster_data = X(cluster_idx, :);
            center = mean(cluster_data, 1);
            wss = wss + sum(sum((cluster_data - center).^2));
        end
    end
end

function centers = recover_centers_from_labels(maps, labels, K)
% Recover cluster centers with polarity alignment
    
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

function Results = create_empty_results(Sim, K_candidates, n_maps, C_dims, criterion)
% Create empty results structure for failed fits
    
    Results = struct(...
        'method', 'spm_vb', ...
        'criterion', criterion, ...
        'K_true', Sim.K_true, ...
        'K_estimated', NaN, ...
        'K_candidates', K_candidates, ...
        'SNR_dB', Sim.SNR_dB, ...
        'duration_s', Sim.duration_s, ...
        'n_maps', n_maps, ...
        'centers', nan(1, C_dims), ...
        'maps_true', nan(Sim.K_true, C_dims), ...
        'labels', ones(n_maps, 1), ...
        'free_energy', -Inf(numel(K_candidates), 1), ...
        'silhouette_vals', -ones(numel(K_candidates), 1), ...
        'within_ss', Inf(numel(K_candidates), 1), ...
        'recovery_metrics', struct('n_matched', 0, 'mean_recovery_matched', 0, ...
            'mean_recovery_padded', 0, 'f1_score', 0, 'sensitivity', 0, 'precision', 0), ...
        'mean_recovery', NaN, ...
        'recovery_corr', [], ...
        'avg_recovery_per_state', NaN, ...
        'valid_fit', false, ...
        'runtime', 0);
end