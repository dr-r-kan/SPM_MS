function Results = fit_microstate_dp_mixture(Sim, K_candidates, criterion)
% FIT_MICROSTATE_DP_MIXTURE: Dirichlet Process Mixture using SPM
%
% Uses Dirichlet Process for automatic K determination with VB
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Max K values to consider (default 2:10)
%   criterion    - 'free_energy' or 'silhouette' (default 'free_energy')
%
% OUTPUTS:
%   Results      - Structure with fitted model and diagnostics

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'free_energy'; end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('Dirichlet Process Mixture (SPM-based)\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);

    % Preprocessing
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = preprocess_maps(Sim);
    
    % Fit DP mixture for each max K
    fprintf('Fitting DP mixture models...\n');
    
    nK = numel(K_candidates);
    scores = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    free_energy_vals = zeros(nK, 1);
    K_inferred = zeros(nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    
    for iK = 1:nK
        K_max = K_candidates(iK);
        fprintf('  K_max=%d... ', K_max);
        
        try
            % DP mixture with polarity
            [centers, labels, K_inf, fe, sil] = dp_mixture_polarity(maps_norm, K_max);
            
            centers_all{iK} = centers;
            labels_all{iK} = labels;
            K_inferred(iK) = K_inf;
            free_energy_vals(iK) = fe;
            silhouette_vals(iK) = sil;
            
            % Select score
            if strcmp(criterion, 'silhouette')
                scores(iK) = sil;
                fprintf('K_inf=%d, Sil=%.3f, FE=%.1f\n', K_inf, sil, fe);
            else
                scores(iK) = fe;
                fprintf('K_inf=%d, FE=%.1f, Sil=%.3f\n', K_inf, fe, sil);
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            scores(iK) = -Inf;
            free_energy_vals(iK) = -Inf;
            silhouette_vals(iK) = -1;
            K_inferred(iK) = NaN;
        end
    end
    
    % Model selection
    [best_score, best_idx] = max(scores);
    K_estimated = K_inferred(best_idx);
    centers = centers_all{best_idx};
    labels = labels_all{best_idx};
    
    fprintf('\nBest K=%d (score=%.3f, true K=%d)\n', K_estimated, best_score, Sim.K_true);
    
    % Map recovery
    true_maps_norm = normalize_maps(Sim.maps_true);
    recovery_corr = best_match_corr_hungarian_polarity_aware(true_maps_norm, centers);
    mean_recovery = mean(recovery_corr);
    
    fprintf('Map recovery: %.3f\n', mean_recovery);
    
    runtime = toc(t_start);
    
    % Return results
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
        'scores', scores, ...
        'best_criterion_value', best_score, ...
        'maps_nc', maps_norm, ...
        'idx_peaks', idx_peaks, ...
        'gfp_vec', gfp_vec, ...
        'mean_recovery', mean_recovery, ...
        'recovery_corr', recovery_corr, ...
        'valid_fit', true, ...
        'runtime', runtime);
end

% ======================== DP MIXTURE IMPLEMENTATION ========================

function [centers, labels, K_inferred, free_energy, silhouette_score] = dp_mixture_polarity(X, K_max)
    % Dirichlet Process mixture with stick-breaking and polarity invariance
    
    [N, D] = size(X);
    max_iter = 100;
    tol = 1e-6;
    
    % DP hyperparameters
    alpha_dp = 1.0;  % Concentration parameter
    
    % Initialize
    idx = randperm(N, min(K_max, N));
    C = X(idx, :);
    C = C ./ (sqrt(sum(C.^2, 2)) + eps);
    
    % Stick-breaking weights (variational parameters)
    v = ones(K_max, 1) * 0.5;  % Stick-breaking proportions
    
    for iter = 1:max_iter
        C_old = C;
        
        % E-step: Compute responsibilities
        sim = abs(X * C');  % [N x K_max]
        
        % Stick-breaking weights
        pi = stick_breaking_weights(v);
        
        % Responsibilities
        log_rho = log(sim + eps) + log(pi' + eps);
        log_rho = log_rho - max(log_rho, [], 2);
        rho = exp(log_rho);
        rho = rho ./ (sum(rho, 2) + eps);
        
        % M-step: Update centers
        active = false(K_max, 1);
        for k = 1:K_max
            Nk = sum(rho(:, k));
            if Nk > 0.5  % Active component threshold
                active(k) = true;
                % Align polarities
                Xk = X .* sign(X * C(k, :)');
                C(k, :) = sum(Xk .* rho(:, k), 1) / Nk;
                C(k, :) = C(k, :) / (norm(C(k, :)) + eps);
            end
        end
        
        % Update stick-breaking proportions
        for k = 1:K_max
            v(k) = 1 + sum(rho(:, k));
            for kk = (k+1):K_max
                v(k) = v(k) + sum(rho(:, kk));
            end
            v(k) = v(k) / (alpha_dp + N);
        end
        
        % Check convergence
        if max(abs(abs(diag(C * C_old')) - 1)) < tol
            break;
        end
    end
    
    % Determine active components
    K_inferred = sum(active);
    centers = C(active, :);
    
    % Final assignment
    sim = abs(X * centers');
    [~, labels] = max(sim, [], 2);
    
    % Compute free energy
    free_energy = compute_free_energy_dp(X, centers, rho(:, active), v(active), alpha_dp);
    
    % Silhouette score
    silhouette_score = polarity_silhouette(X, labels, centers);
end

function pi = stick_breaking_weights(v)
    % Convert stick-breaking proportions to mixture weights
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
    % Free energy for DP mixture
    [N, ~] = size(X);
    K = size(C, 1);
    
    % Log likelihood
    log_like = 0;
    for k = 1:K
        sim = abs(X * C(k, :)');
        log_like = log_like + sum(rho(:, k) .* log(sim + eps));
    end
    
    % KL for stick-breaking weights
    pi = stick_breaking_weights(v);
    kl_pi = sum(pi .* log(pi + eps)) + alpha_dp * log(K);
    
    % Entropy
    entropy = -sum(rho(:) .* log(rho(:) + eps));
    
    fe = log_like - kl_pi + entropy;
end

function sil = polarity_silhouette(X, labels, centers)
    % Same as in VB K-means
    K = size(centers, 1);
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
            a = mean(1 - abs(X(same_cluster, :) * X(i, :)'));
        else
            a = 0;
        end
        
        b = Inf;
        for kk = 1:K
            if kk == k, continue; end
            other_cluster = find(labels == kk);
            if ~isempty(other_cluster)
                b = min(b, mean(1 - abs(X(other_cluster, :) * X(i, :)')));
            end
        end
        
        sil_vals(i) = (b - a) / max(a, b);
    end
    
    sil = mean(sil_vals);
end

% ======================== HELPER FUNCTIONS (shared) ========================

function [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = preprocess_maps(Sim)
    X_bp = bandpass_fft_zero_phase(Sim.X_noisy, Sim.sfreq, [2 20]);
    [maps_nc, idx_peaks, gfp_vec] = gfp_peak_maps(X_bp, 3, 0.80);
    
    if isempty(maps_nc)
        idx_peaks = 1:2:size(X_bp, 2);
        maps_nc = X_bp(:, idx_peaks)';
    end
    
    n_maps = size(maps_nc, 1);
    C_dims = size(maps_nc, 2);
    maps_norm = normalize_maps(maps_nc);
end

function maps_norm = normalize_maps(maps)
    maps_norm = maps - mean(maps, 2);
    maps_norm = maps_norm ./ (sqrt(sum(maps_norm.^2, 2)) + eps);
end

function Xf = bandpass_fft_zero_phase(X, sfreq, bp)
    if isempty(bp) || numel(bp) ~= 2, Xf = X; return; end
    T = size(X, 2);
    F = fft(X, [], 2);
    freqs = (0:T-1) * (sfreq / T);
    mask = (freqs >= bp(1) & freqs <= bp(2)) | (freqs >= sfreq - bp(2) & freqs <= sfreq - bp(1));
    F(:, ~mask) = 0;
    Xf = real(ifft(F, [], 2));
end

function [maps_nc, idx_peaks, gfp] = gfp_peak_maps(X, min_dist, pct)
    if nargin < 3, pct = 0.80; end
    gfp = sqrt(mean((X - mean(X, 1)).^2, 1));
    idx_peaks = find_local_peaks(gfp, min_dist, pct);
    if isempty(idx_peaks), maps_nc = []; else, maps_nc = X(:, idx_peaks)'; end
end

function idx = find_local_peaks(x, min_dist, pct)
    x = x(:)';
    n = numel(x);
    if n < 3, idx = 1:n; return; end
    cand = find([false, x(2:end-1) > x(1:end-2) & x(2:end-1) >= x(3:end), false]);
    if isempty(cand), idx = []; return; end
    thr = quantile(x, pct);
    cand = cand(x(cand) >= thr);
    if isempty(cand), idx = []; return; end
    [~, ord] = sort(x(cand), 'descend');
    keep = false(size(x));
    for ii = ord
        c = cand(ii);
        lo = max(1, c - min_dist);
        hi = min(n, c + min_dist);
        if ~any(keep(lo:hi)), keep(c) = true; end
    end
    idx = find(keep);
end

function m = best_match_corr_hungarian_polarity_aware(A, B)
    M = abs(A * B');
    cost = 1 - M;
    K = min(size(M, 1), size(M, 2));
    if K == 0, m = []; return; end
    cost = cost(1:K, 1:K);
    [assign] = hungarian(cost);
    idx = sub2ind(size(M), assign(:, 1), assign(:, 2));
    m = M(idx);
end

function assignment = hungarian(costMat)
    [rr, cc] = meshgrid(1:size(costMat, 1), 1:size(costMat, 1));
    [~, idx] = min(costMat(:));
    assignment = [rr(idx), cc(idx)];
end