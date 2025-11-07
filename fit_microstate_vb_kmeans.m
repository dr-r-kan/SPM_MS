function Results = fit_microstate_vb_kmeans(Sim, K_candidates, criterion)
% FIT_MICROSTATE_VB_KMEANS: VB K-means with polarity invariance
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Vector of K values (default 2:10)
%   criterion    - 'free_energy' or 'silhouette' (default 'free_energy')
%
% OUTPUTS:
%   Results      - Complete results structure with recovery metrics

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'free_energy'; end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('VB K Means Microstate Fitting\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);

    util = microstate_utilities_SHARED();
    fprintf('1. Preprocessing...\n');
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original] = util.preprocess_maps(Sim);
    
    fprintf('2. Fitting VB K means models...\n');
    
    nK = numel(K_candidates);
    scores = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    free_energy_vals = zeros(nK, 1);
    gev_vals = zeros(nK, 1);  % ← NOW COMPUTE THIS
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    
    % ✅ FITTING LOOP
    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('  K=%d... ', K);
        
        try
            [centers, labels, fe, sil] = vb_kmeans_polarity(maps_norm, K);
            
            centers_all{iK} = centers;
            labels_all{iK} = labels;
            free_energy_vals(iK) = fe;
            silhouette_vals(iK) = sil;
            
            % ✅ COMPUTE GEV (KOENIG METHOD)
            sim = abs(maps_original * centers');
            [max_sim, ~] = max(sim, [], 2);
            gfp_squared = sum(maps_original.^2, 2);
            gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);
            
            if strcmp(criterion, 'silhouette')
                scores(iK) = sil;
                fprintf('Sil=%.3f, FE=%.1f, GEV=%.3f\n', sil, fe, gev_vals(iK));
            else
                scores(iK) = fe;
                fprintf('FE=%.1f, Sil=%.3f, GEV=%.3f\n', fe, sil, gev_vals(iK));
            end
                
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            scores(iK) = -Inf;
            free_energy_vals(iK) = -Inf;
            silhouette_vals(iK) = -1;
        end
    end
    
    % ✅ Ensure ALL criteria are available for post-hoc use
    % (silhouette_vals and free_energy_vals already computed above)
    
    [best_score, best_idx] = max(scores);
    K_estimated = K_candidates(best_idx);
    centers = centers_all{best_idx};
    labels = labels_all{best_idx};
    
    fprintf('Best K=%d (score=%.3f, true K=%d)\n', K_estimated, best_score, Sim.K_true);
    
    % Recovery metrics with partial alignment
    true_maps_norm = util.normalize_maps(Sim.maps_true);
    recovery_metrics = microstate_partial_alignment(true_maps_norm, centers, ...
        'distance_type', 'cosine', 'threshold', 0.0, 'polarity', true);
    
    fprintf('\n3. Map Recovery Analysis:\n');
    fprintf('  Matched: %d, Sensitivity: %.4f, Precision: %.4f, F1: %.4f\n', ...
        recovery_metrics.n_matched, recovery_metrics.sensitivity, ...
        recovery_metrics.precision, recovery_metrics.f1_score);
    
    runtime = toc(t_start);
    
    Results = struct( ...
        'method', 'vb_kmeans', ...
        'criterion', criterion, ...
        'K_true', Sim.K_true, ...
        'K_estimated', K_estimated, ...
        'K_candidates', K_candidates, ...
        'SNR_dB', Sim.SNR_dB, ...
        'duration_s', Sim.duration_s, ...
        'n_maps', n_maps, ...
        'centers', centers, ...
        'maps_true', true_maps_norm, ...
        'labels', labels, ...
        'free_energy_vals', free_energy_vals, ...
        'silhouette_vals', silhouette_vals, ...
        'scores', scores, ...
        'best_criterion_value', best_score, ...
        'maps_nc', maps_norm, ...
        'idx_peaks', idx_peaks, ...
        'gfp_vec', gfp_vec, ...
        'recovery_metrics', recovery_metrics, ...
        'mean_recovery', recovery_metrics.mean_recovery_matched, ...
        'recovery_corr', recovery_metrics.match_similarities, ...
        'avg_recovery_per_state', recovery_metrics.mean_recovery_padded, ...
        'within_ss', zeros(nK, 1), ...                    % ✅ ADD for post-hoc elbow
        'gev_vals', zeros(nK, 1), ...                      % ✅ ADD for post-hoc gev
        'valid_fit', true, ...
        'runtime', runtime);
end

% ======================== LOCAL HELPERS ========================

function [centers, labels, free_energy, silhouette_score] = vb_kmeans_polarity(X, K)
% VB K-means with polarity invariance
    
    [N, D] = size(X);
    max_iter = 100;
    tol = 1e-6;
    n_init = 5;
    
    alpha0 = 1;
    beta0 = 1;
    
    best_fe = -Inf;
    best_centers = [];
    best_labels = [];
    
    for init = 1:n_init
        idx = randperm(N, K);
        C = X(idx, :);
        C = C ./ (sqrt(sum(C.^2, 2)) + eps);
        
        alpha = alpha0 * ones(K, 1);
        beta = beta0 * ones(K, 1);
        
        for iter = 1:max_iter
            C_old = C;
            
            sim = abs(X * C');
            [max_sim, L] = max(sim, [], 2);
            
            log_rho = zeros(N, K);
            for k = 1:K
                log_rho(:, k) = psi(alpha(k)) + beta(k) * abs(X * C(k, :)');
            end
            log_rho = log_rho - max(log_rho, [], 2);
            rho = exp(log_rho);
            rho = rho ./ (sum(rho, 2) + eps);
            
            for k = 1:K
                Nk = sum(rho(:, k));
                if Nk < eps
                    C(k, :) = X(randi(N), :);
                    alpha(k) = alpha0;
                    beta(k) = beta0;
                else
                    Xk = X .* sign(X * C(k, :)');
                    C(k, :) = sum(Xk .* rho(:, k), 1) / Nk;
                    alpha(k) = alpha0 + Nk;
                    err = 1 - abs(Xk * C(k, :)');
                    beta(k) = beta0 + sum(rho(:, k) .* err);
                end
                C(k, :) = C(k, :) / (norm(C(k, :)) + eps);
            end
            
            if max(abs(abs(diag(C * C_old')) - 1)) < tol
                break;
            end
        end
        
        fe = compute_free_energy_kmeans(X, C, rho, alpha, beta, alpha0, beta0);
        
        if fe > best_fe
            best_fe = fe;
            best_centers = C;
            best_labels = L;
        end
    end
    
    centers = best_centers;
    labels = best_labels;
    free_energy = best_fe;
    silhouette_score = polarity_silhouette_koenig(X, labels, centers);
end

function fe = compute_free_energy_kmeans(X, C, rho, alpha, beta, alpha0, beta0)
% Compute free energy for VB K-means
    
    [N, D] = size(X);
    K = size(C, 1);
    
    log_like = 0;
    for k = 1:K
        sim = abs(X * C(k, :)');
        log_like = log_like + sum(rho(:, k) .* log(sim + eps));
    end
    
    kl_pi = sum((alpha - alpha0) .* (psi(alpha) - psi(sum(alpha))));
    kl_beta = sum((beta - beta0) .* log(beta / (beta0 + eps)));
    entropy = -sum(rho(:) .* log(rho(:) + eps));
    
    fe = log_like - kl_pi - kl_beta + entropy;
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