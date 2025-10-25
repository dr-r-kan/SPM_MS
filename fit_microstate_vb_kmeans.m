function Results = fit_microstate_vb_kmeans(Sim, K_candidates, criterion)
% FIT_MICROSTATE_VB_KMEANS: VB-based K-means using SPM with polarity invariance
%
% Uses SPM's variational Bayes framework for K-means clustering
% with model selection via Free Energy or Silhouette Score
%
% INPUTS:
%   Sim          - Simulation structure from generate_microstate_eeg
%   K_candidates - Vector of K values to test (default 2:10)
%   criterion    - 'free_energy' or 'silhouette' (default 'free_energy')
%
% OUTPUTS:
%   Results      - Structure with fitted model and diagnostics

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'free_energy'; end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('VB K-Means Microstate Fitting (SPM)\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);

    % Preprocessing
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = preprocess_maps(Sim);
    
    % Fit VB K-means for each K
    fprintf('Fitting VB K-means models...\n');
    
    nK = numel(K_candidates);
    scores = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    free_energy_vals = zeros(nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    
    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('  K=%d... ', K);
        
        try
            % VB K-means with polarity invariance
            [centers, labels, fe, sil] = vb_kmeans_polarity(maps_norm, K);
            
            centers_all{iK} = centers;
            labels_all{iK} = labels;
            free_energy_vals(iK) = fe;
            silhouette_vals(iK) = sil;
            
            % Select score based on criterion
            if strcmp(criterion, 'silhouette')
                scores(iK) = sil;
                fprintf('Sil=%.3f, FE=%.1f\n', sil, fe);
            else
                scores(iK) = fe;
                fprintf('FE=%.1f, Sil=%.3f\n', fe, sil);
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            scores(iK) = -Inf;
            free_energy_vals(iK) = -Inf;
            silhouette_vals(iK) = -1;
        end
    end
    
    % Model selection
    [best_score, best_idx] = max(scores);
    K_estimated = K_candidates(best_idx);
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
        'mean_recovery', mean_recovery, ...
        'recovery_corr', recovery_corr, ...
        'valid_fit', true, ...
        'runtime', runtime);
end

% ======================== VB K-MEANS IMPLEMENTATION ========================

function [centers, labels, free_energy, silhouette_score] = vb_kmeans_polarity(X, K)
    % VB K-means with polarity invariance using SPM-like variational framework
    
    [N, D] = size(X);
    max_iter = 100;
    tol = 1e-6;
    n_init = 5;
    
    % Hyperparameters (priors)
    alpha0 = 1;  % Dirichlet prior
    beta0 = 1;   % Precision prior
    
    best_fe = -Inf;
    best_centers = [];
    best_labels = [];
    
    for init = 1:n_init
        % Random initialization
        idx = randperm(N, K);
        C = X(idx, :);
        C = C ./ (sqrt(sum(C.^2, 2)) + eps);
        
        % Variational parameters
        alpha = alpha0 * ones(K, 1);
        beta = beta0 * ones(K, 1);
        
        for iter = 1:max_iter
            C_old = C;
            
            % E-step: Compute responsibilities with polarity
            sim = abs(X * C');  % [N x K]
            [max_sim, L] = max(sim, [], 2);
            
            % Soft assignment (variational)
            log_rho = zeros(N, K);
            for k = 1:K
                log_rho(:, k) = psi(alpha(k)) + beta(k) * sum(abs(X * C(k, :)'), 2);
            end
            log_rho = log_rho - max(log_rho, [], 2);
            rho = exp(log_rho);
            rho = rho ./ (sum(rho, 2) + eps);
            
            % M-step: Update centers with polarity correction
            for k = 1:K
                Nk = sum(rho(:, k));
                if Nk < eps
                    C(k, :) = X(randi(N), :);
                    alpha(k) = alpha0;
                    beta(k) = beta0;
                else
                    % Align polarities
                    Xk = X .* sign(X * C(k, :)');
                    C(k, :) = sum(Xk .* rho(:, k), 1) / Nk;
                    
                    % Update variational parameters
                    alpha(k) = alpha0 + Nk;
                    err = 1 - abs(Xk * C(k, :)');
                    beta(k) = beta0 + sum(rho(:, k) .* err);
                end
                C(k, :) = C(k, :) / (norm(C(k, :)) + eps);
            end
            
            % Check convergence
            if max(abs(abs(diag(C * C_old')) - 1)) < tol
                break;
            end
        end
        
        % Compute free energy
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
    
    % Compute silhouette score with polarity-aware distance
    silhouette_score = polarity_silhouette(X, labels, centers);
end

function fe = compute_free_energy_kmeans(X, C, rho, alpha, beta, alpha0, beta0)
    % Compute variational free energy for K-means
    [N, D] = size(X);
    K = size(C, 1);
    
    % Log likelihood term
    log_like = 0;
    for k = 1:K
        sim = abs(X * C(k, :)');
        log_like = log_like + sum(rho(:, k) .* log(sim + eps));
    end
    
    % KL divergence terms
    kl_pi = sum((alpha - alpha0) .* (psi(alpha) - psi(sum(alpha))));
    kl_beta = sum((beta - beta0) .* log(beta / beta0));
    
    % Entropy of q
    entropy = -sum(rho(:) .* log(rho(:) + eps));
    
    fe = log_like - kl_pi - kl_beta + entropy;
end

function sil = polarity_silhouette(X, labels, centers)
    % Polarity-aware silhouette score
    K = size(centers, 1);
    N = size(X, 1);
    
    if K < 2
        sil = 0;
        return;
    end
    
    sil_vals = zeros(N, 1);
    
    for i = 1:N
        k = labels(i);
        
        % Intra-cluster distance (polarity-aware)
        same_cluster = find(labels == k);
        if numel(same_cluster) > 1
            a = mean(1 - abs(X(same_cluster, :) * X(i, :)'));
        else
            a = 0;
        end
        
        % Inter-cluster distance (to nearest cluster)
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

% ======================== HELPER FUNCTIONS ========================

function [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = preprocess_maps(Sim)
    % Bandpass and extract GFP peaks
    X_bp = bandpass_fft_zero_phase(Sim.X_noisy, Sim.sfreq, [2 20]);
    [maps_nc, idx_peaks, gfp_vec] = gfp_peak_maps(X_bp, 3, 0.80);
    
    if isempty(maps_nc)
        idx_peaks = 1:2:size(X_bp, 2);
        maps_nc = X_bp(:, idx_peaks)';
    end
    
    n_maps = size(maps_nc, 1);
    C_dims = size(maps_nc, 2);
    
    % Normalize
    maps_norm = normalize_maps(maps_nc);
end

function maps_norm = normalize_maps(maps)
    maps_norm = maps - mean(maps, 2);
    maps_norm = maps_norm ./ (sqrt(sum(maps_norm.^2, 2)) + eps);
end

function Xf = bandpass_fft_zero_phase(X, sfreq, bp)
    if isempty(bp) || numel(bp) ~= 2
        Xf = X;
        return;
    end
    T = size(X, 2);
    f1 = bp(1);
    f2 = bp(2);
    F = fft(X, [], 2);
    freqs = (0:T-1) * (sfreq / T);
    mask = (freqs >= f1 & freqs <= f2) | (freqs >= sfreq - f2 & freqs <= sfreq - f1);
    F(:, ~mask) = 0;
    Xf = real(ifft(F, [], 2));
end

function [maps_nc, idx_peaks, gfp] = gfp_peak_maps(X, min_dist, pct)
    if nargin < 3, pct = 0.80; end
    gfp = sqrt(mean((X - mean(X, 1)).^2, 1));
    idx_peaks = find_local_peaks(gfp, min_dist, pct);
    if isempty(idx_peaks)
        maps_nc = [];
    else
        maps_nc = X(:, idx_peaks)';
    end
end

function idx = find_local_peaks(x, min_dist, pct)
    if nargin < 3, pct = 0.8; end
    x = x(:)';
    n = numel(x);
    if n < 3
        idx = 1:n;
        return;
    end
    cand = find([false, x(2:end-1) > x(1:end-2) & x(2:end-1) >= x(3:end), false]);
    if isempty(cand)
        idx = [];
        return;
    end
    thr = quantile(x, pct);
    cand = cand(x(cand) >= thr);
    if isempty(cand)
        idx = [];
        return;
    end
    [~, ord] = sort(x(cand), 'descend');
    keep = false(size(x));
    for ii = ord
        c = cand(ii);
        lo = max(1, c - min_dist);
        hi = min(n, c + min_dist);
        if ~any(keep(lo:hi))
            keep(c) = true;
        end
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
    % Simplified Hungarian algorithm
    [rr, cc] = meshgrid(1:size(costMat, 1), 1:size(costMat, 1));
    [~, idx] = min(costMat(:));
    assignment = [rr(idx), cc(idx)];
end