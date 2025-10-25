function Results = fit_microstate_kmeans_standard(Sim, K_candidates, criterion)
% FIT_MICROSTATE_KMEANS_STANDARD: Standard (non-Bayesian) K-means with polarity
%
% Classic modified K-means for microstate analysis (the field standard)
% Model selection via Silhouette, GEV, or Elbow method
%
% INPUTS:
%   Sim          - Simulation structure from generate_microstate_eeg
%   K_candidates - Vector of K values to test (default 2:10)
%   criterion    - 'silhouette', 'gev', or 'elbow' (default 'silhouette')
%
% OUTPUTS:
%   Results      - Structure with fitted model and diagnostics

    if nargin < 2, K_candidates = 2:10; end
    if nargin < 3, criterion = 'silhouette'; end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('Standard K-Means Microstate Fitting\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    fprintf('True K: %d, SNR: %+.1f dB\n', Sim.K_true, Sim.SNR_dB);

    % Preprocessing
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = preprocess_maps(Sim);
    
    % Fit standard K-means for each K
    fprintf('Fitting standard K-means models...\n');
    
    nK = numel(K_candidates);
    gev_vals = zeros(nK, 1);
    silhouette_vals = zeros(nK, 1);
    within_ss = zeros(nK, 1);
    centers_all = cell(nK, 1);
    labels_all = cell(nK, 1);
    
    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('  K=%d... ', K);
        
        try
            % Standard modified K-means with polarity invariance
            [centers, labels] = modified_kmeans_polarity(maps_norm, K);
            
            centers_all{iK} = centers;
            labels_all{iK} = labels;
            
            % Compute metrics
            % 1. Global Explained Variance
            sim = abs(maps_norm * centers');
            [max_sim, ~] = max(sim, [], 2);
            gev_vals(iK) = mean(max_sim.^2);
            
            % 2. Silhouette score
            silhouette_vals(iK) = polarity_silhouette(maps_norm, labels, centers);
            
            % 3. Within-cluster sum of squares (for elbow)
            wss = 0;
            for k = 1:K
                cluster_maps = maps_norm(labels == k, :);
                if ~isempty(cluster_maps)
                    % Polarity-aware distances
                    dists = 1 - abs(cluster_maps * centers(k, :)');
                    wss = wss + sum(dists.^2);
                end
            end
            within_ss(iK) = wss;
            
            fprintf('GEV=%.3f, Sil=%.3f, WSS=%.1f\n', ...
                gev_vals(iK), silhouette_vals(iK), within_ss(iK));
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            gev_vals(iK) = 0;
            silhouette_vals(iK) = -1;
            within_ss(iK) = Inf;
        end
    end
    
    % Model selection based on criterion
    fprintf('\nModel selection using %s...\n', criterion);
    
    switch criterion
        case 'silhouette'
            [best_score, best_idx] = max(silhouette_vals);
            
        case 'gev'
            [best_score, best_idx] = max(gev_vals);
            
        case 'elbow'
            % Elbow detection on WSS curve
            if length(within_ss) >= 3
                % Normalize
                wss_norm = (within_ss - min(within_ss)) / (max(within_ss) - min(within_ss) + eps);
                k_norm = (K_candidates - min(K_candidates)) / (max(K_candidates) - min(K_candidates) + eps);
                
                % Find elbow (maximum distance to line)
                elbow_scores = zeros(size(wss_norm));
                for i = 2:length(wss_norm)-1
                    p1 = [k_norm(1), wss_norm(1)];
                    p2 = [k_norm(end), wss_norm(end)];
                    p = [k_norm(i), wss_norm(i)];
                    
                    elbow_scores(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + ...
                        p2(1)*p1(2) - p2(2)*p1(1)) / ...
                        sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2);
                end
                
                [best_score, best_idx] = max(elbow_scores);
            else
                [~, best_idx] = min(within_ss);
                best_score = 0;
            end
            
        otherwise
            error('Unknown criterion: %s', criterion);
    end
    
    K_estimated = K_candidates(best_idx);
    centers = centers_all{best_idx};
    labels = labels_all{best_idx};
    
    fprintf('Best K=%d (score=%.3f, true K=%d)\n', K_estimated, best_score, Sim.K_true);
    
    % Map recovery
    true_maps_norm = normalize_maps(Sim.maps_true);
    recovery_corr = best_match_corr_hungarian_polarity_aware(true_maps_norm, centers);
    mean_recovery = mean(recovery_corr);
    
    % NEW: Average recovery per extracted state
    avg_recovery_per_state = mean(max(abs(centers * true_maps_norm'), [], 2));
    
    fprintf('Map recovery (best match): %.3f\n', mean_recovery);
    fprintf('Avg recovery per extracted state: %.3f\n', avg_recovery_per_state);
    
    runtime = toc(t_start);
    
    % Return results
    Results = struct( ...
        'method', 'kmeans_standard', ...
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
        'gev_vals', gev_vals, ...
        'silhouette_vals', silhouette_vals, ...
        'within_ss', within_ss, ...
        'best_criterion_value', best_score, ...
        'maps_nc', maps_norm, ...
        'idx_peaks', idx_peaks, ...
        'gfp_vec', gfp_vec, ...
        'mean_recovery', mean_recovery, ...
        'recovery_corr', recovery_corr, ...
        'avg_recovery_per_state', avg_recovery_per_state, ...
        'valid_fit', true, ...
        'runtime', runtime);
end

% ======================== STANDARD K-MEANS ========================

function [centers, labels] = modified_kmeans_polarity(X, K)
    % Standard modified K-means with polarity invariance
    % The gold standard method in microstate analysis
    
    [N, D] = size(X);
    max_iter = 100;
    tol = 1e-6;
    n_init = 10;  % Multiple random starts
    
    best_gev = -Inf;
    best_centers = [];
    best_labels = [];
    
    for init = 1:n_init
        % Random initialization
        idx = randperm(N, K);
        C = X(idx, :);
        C = C ./ (sqrt(sum(C.^2, 2)) + eps);
        
        for iter = 1:max_iter
            C_old = C;
            
            % Assignment step (polarity-invariant)
            sim = abs(X * C');  % [N x K]
            [~, L] = max(sim, [], 2);
            
            % Update step
            for k = 1:K
                mask = (L == k);
                if sum(mask) == 0
                    % Empty cluster: reinitialize
                    C(k, :) = X(randi(N), :);
                else
                    % Compute mean with polarity correction
                    Xk = X(mask, :);
                    % Flip polarities to align with current center
                    signs = sign(Xk * C(k, :)');
                    Xk = Xk .* signs;
                    % Average and normalize
                    C(k, :) = mean(Xk, 1);
                end
                C(k, :) = C(k, :) / (norm(C(k, :)) + eps);
            end
            
            % Check convergence
            if max(abs(abs(diag(C * C_old')) - 1)) < tol
                break;
            end
        end
        
        % Calculate GEV for this initialization
        sim = abs(X * C');
        [max_sim, L] = max(sim, [], 2);
        gev = mean(max_sim.^2);
        
        if gev > best_gev
            best_gev = gev;
            best_centers = C;
            best_labels = L;
        end
    end
    
    centers = best_centers;
    labels = best_labels;
end

function sil = polarity_silhouette(X, labels, centers)
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
    n = size(costMat, 1);
    if n == 0, assignment = []; return; end
    costMat = costMat - min(costMat, [], 2);
    costMat = costMat - min(costMat, [], 1);
    assignment = zeros(n, 2);
    assigned_rows = false(n, 1);
    assigned_cols = false(n, 1);
    for i = 1:n
        [row_min, row_idx] = min(costMat, [], 2);
        [~, best_row] = min(row_min);
        if ~assigned_rows(best_row) && ~assigned_cols(row_idx(best_row))
            assignment(i, :) = [best_row, row_idx(best_row)];
            assigned_rows(best_row) = true;
            assigned_cols(row_idx(best_row)) = true;
            costMat(best_row, :) = Inf;
            costMat(:, row_idx(best_row)) = Inf;
        end
    end
    assignment = assignment(assignment(:, 1) > 0, :);
end