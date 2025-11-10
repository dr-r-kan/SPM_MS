function Results = fit_microstate_dp_mixture(Sim, K_candidates, criterion)
% FIT_MICROSTATE_DP_MIXTURE: Dirichlet Process Mixture
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Max K values to consider (default 2:10)
%   criterion    - 'free_energy' or 'silhouette' (default 'free_energy')
%
% OUTPUTS:
%   Results      - Structure with fitted model and recovery metrics

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'free_energy'; end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('Dirichlet Process Mixture\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);

    % Load shared utilities
    util = microstate_utilities_SHARED();

    % Preprocessing (using shared utility)
    fprintf('1. Preprocessing...\n');
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original] = util.preprocess_maps(Sim);
    
    fprintf('2. Fitting DP mixture models...\n');
    
    nK = numel(K_candidates);
    scores = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    free_energy_vals = zeros(nK, 1);
    gev_vals = zeros(nK, 1);  % ← NOW COMPUTE THIS
    K_inferred = zeros(nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    
    for iK = 1:nK
        K_max = K_candidates(iK);
        fprintf('  K_max=%d... ', K_max);
        
        try
            [centers, labels, K_inf, fe, sil] = dp_mixture_polarity(maps_norm, K_max);
            
            centers_all{iK} = centers;
            labels_all{iK} = labels;
            K_inferred(iK) = K_inf;
            free_energy_vals(iK) = fe;
            silhouette_vals(iK) = sil;
            
            % ✅ COMPUTE GEV (KOENIG METHOD)
            sim = abs(maps_original * centers');
            [max_sim, ~] = max(sim, [], 2);
            gfp_squared = sum(maps_original.^2, 2);
            gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);
            
            if strcmp(criterion, 'silhouette')
                scores(iK) = sil;
                fprintf('K_inf=%d, Sil=%.3f, FE=%.1f, GEV=%.3f\n', K_inf, sil, fe, gev_vals(iK));
            else
                scores(iK) = fe;
                fprintf('K_inf=%d, FE=%.1f, Sil=%.3f, GEV=%.3f\n', K_inf, fe, sil, gev_vals(iK));
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            scores(iK) = -Inf;
            free_energy_vals(iK) = -Inf;
            silhouette_vals(iK) = -1;
            K_inferred(iK) = NaN;
        end
    end
    
    [best_score, best_idx] = max(scores);
    K_estimated = K_inferred(best_idx);
    centers = centers_all{best_idx};
    labels = labels_all{best_idx};
    
    fprintf('Best K=%d (score=%.3f, true K=%s)\n', K_estimated, best_score, ...
        iif(isfield(Sim, 'K_true') && ~isnan(Sim.K_true), num2str(Sim.K_true), 'N/A'));
    
    % Recovery (only if ground truth is available)
    if isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true)
        true_maps_norm = util.normalize_maps(Sim.maps_true);
        recovery_metrics = microstate_partial_alignment(true_maps_norm, centers, ...
            'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
        
        fprintf('\n3. Map Recovery Analysis:\n');
        fprintf('  Matched: %d, Sensitivity: %.4f, Precision: %.4f, F1: %.4f\n', ...
            recovery_metrics.n_matched, recovery_metrics.sensitivity, ...
            recovery_metrics.precision, recovery_metrics.f1_score);
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
        fprintf('\nNo ground truth available - skipping recovery metrics\n');
    end
    
    runtime = toc(t_start);
    
    Results = struct( ...
        'method', 'dp_mixture', ...
        'criterion', criterion, ...
        'K_true', Sim.K_true, ...
        'K_estimated', K_estimated, ...
        'K_candidates', K_candidates, ...
        'K_inferred', K_inferred, ...
        'SNR_dB', Sim.SNR_dB, ...
        'duration_s', Sim.duration_s, ...
        'n_maps', n_maps, ...
        'centers', centers, ...
        'maps_true', true_maps_norm, ...
        'labels', labels, ...
        'free_energy_vals', free_energy_vals, ...
        'silhouette_vals', silhouette_vals, ...
        'gev_vals', zeros(nK, 1), ...            
        'within_ss', zeros(nK, 1), ...           
        'scores', scores, ...
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

% ======================== LOCAL HELPERS ========================

function [centers, labels, K_inferred, free_energy, silhouette_score] = dp_mixture_polarity(X, K_max)
% Dirichlet Process mixture model with polarity invariance
    
    [N, D] = size(X);
    max_iter = 100;
    tol = 1e-6;
    
    alpha_dp = 1.0;
    
    idx = randperm(N, min(K_max, N));
    C = X(idx, :);
    C = C ./ (sqrt(sum(C.^2, 2)) + eps);
    
    v = ones(K_max, 1) * 0.5;
    
    for iter = 1:max_iter
        C_old = C;
        
        sim = abs(X * C');
        pi = stick_breaking_weights(v);
        
        log_rho = log(sim + eps) + log(pi' + eps);
        log_rho = log_rho - max(log_rho, [], 2);
        rho = exp(log_rho);
        rho = rho ./ (sum(rho, 2) + eps);
        
        active = false(K_max, 1);
        for k = 1:K_max
            Nk = sum(rho(:, k));
            if Nk > 0.5
                active(k) = true;
                Xk = X .* sign(X * C(k, :)');
                C(k, :) = sum(Xk .* rho(:, k), 1) / Nk;
                C(k, :) = C(k, :) / (norm(C(k, :)) + eps);
            end
        end
        
        for k = 1:K_max
            v(k) = 1 + sum(rho(:, k));
            for kk = (k+1):K_max
                v(k) = v(k) + sum(rho(:, kk));
            end
            v(k) = v(k) / (alpha_dp + N);
        end
        
        if max(abs(abs(diag(C * C_old')) - 1)) < tol
            break;
        end
    end
    
    K_inferred = sum(active);
    centers = C(active, :);
    
    sim = abs(X * centers');
    [~, labels] = max(sim, [], 2);
    
    free_energy = compute_free_energy_dp(X, centers, rho(:, active), v(active), alpha_dp);
    silhouette_score = polarity_silhouette_koenig(X, labels, centers);
end

function pi = stick_breaking_weights(v)
% Stick breaking weights for DP
    
    K = numel(v);
    pi = zeros(K, 1);
    remaining = 1;
    for k = 1:K
        pi(k) = v(k) * remaining;
        remaining = remaining * (1 - v(k));
    end
    pi = pi / sum(pi);
end

function fe = compute_free_energy_dp(X, C, rho, v, alpha_dp)
% Compute free energy for DP mixture
    
    [N, ~] = size(X);
    K = size(C, 1);
    
    log_like = 0;
    for k = 1:K
        sim = abs(X * C(k, :)');
        log_like = log_like + sum(rho(:, k) .* log(sim + eps));
    end
    
    pi = stick_breaking_weights(v);
    kl_pi = sum(pi .* log(pi + eps)) + alpha_dp * log(K);
    
    entropy = -sum(rho(:) .* log(rho(:) + eps));
    
    fe = log_like - kl_pi + entropy;
end

function sil = polarity_silhouette_koenig(X, labels, centers)
    % KOENIG METHOD: Simple cosine-based silhouette
    
    K = size(centers, 1);
    N = size(X, 1);
    
    if K < 2 || N < 2
        sil = 0;
        return;
    end
    
    sil_vals = zeros(N, 1);
    
    for i = 1:N
        k = labels(i);
        
        same_cluster = find(labels == k);
        if numel(same_cluster) > 1
            similarities = abs(X(same_cluster, :) * X(i, :)');
            a = mean(1 - similarities(similarities < 1));
            if isnan(a) || a < 0, a = 0; end
        else
            a = 0;
        end
        
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

function out = iif(condition, true_val, false_val)
    % Inline if function
    if condition
        out = true_val;
    else
        out = false_val;
    end
end