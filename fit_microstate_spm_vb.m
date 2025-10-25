function Results = fit_microstate_spm_vb(Sim, K_candidates, criterion)
% FIT_MICROSTATE_SPM_VB: Proper VB microstate fitting using SPM
%
% Uses SPM's spm_mix with polarity-aware preprocessing and 
% automatic K selection via free energy maximization
%
% INPUTS:
%   Sim          - Simulation structure
%   K_candidates - Vector of K values to test (default 2:10)
%   criterion    - 'elbow_sil_combined' or 'elbow_only' (default 'elbow_sil_combined')

    if nargin < 2
        K_candidates = 2:10;
    end
    
    if nargin < 3
        criterion = 'elbow_sil_combined';
    end
    
    t_start = tic;

    fprintf('\n========================================\n');
    fprintf('Microstate VB Fitting (SPM spm_mix)\n');
    fprintf('========================================\n');
    fprintf('Criterion: %s\n', criterion);
    fprintf('True K: %d, SNR: %+.1f dB, Duration: %.1f s\n', ...
        Sim.K_true, Sim.SNR_dB, Sim.duration_s);

    if ~exist('spm_mix', 'file')
        error('SPM spm_mix not found. Add SPM to path.');
    end

    % --- Preprocessing ---
    fprintf('1. Preprocessing...\n');
    X_bp = bandpass_fft_zero_phase(Sim.X_noisy, Sim.sfreq, [2 20]);
    [maps_nc, idx_peaks, gfp_vec] = gfp_peak_maps(X_bp, 3, 0.80);
    
    if isempty(maps_nc)
        idx_peaks = 1:2:size(X_bp, 2);
        maps_nc = X_bp(:, idx_peaks)';
    end
    
    n_maps = size(maps_nc, 1);
    C_dims = size(maps_nc, 2);
    fprintf('   Extracted %d GFP-peak maps (%d channels)\n', n_maps, C_dims);

    % --- Normalize maps ---
    fprintf('2. Normalizing maps...\n');
    maps_norm = maps_nc - mean(maps_nc, 2);
    maps_norm = maps_norm ./ (sqrt(sum(maps_norm.^2, 2)) + eps);
    
    % --- Use REDUCED SPACE instead of outer products ---
    fprintf('3. Dimensionality reduction...\n');
    
    % PCA to extract dominant spatial patterns
    [coeff, score, latent] = pca(maps_norm);
    
    % Keep dimensions explaining 95% variance
    var_explained = cumsum(latent) / sum(latent);
    n_dims = find(var_explained >= 0.95, 1, 'first');
    n_dims = min(n_dims, 20);  % Cap at 20 for stability
    n_dims = max(n_dims, 5);   % At least 5 dimensions
    
    features = score(:, 1:n_dims);
    
    fprintf('   Using %d PCA dimensions (%.1f%% variance)\n', ...
        n_dims, var_explained(n_dims)*100);
    
    % Store PCA model for reconstruction
    pca_model = struct('coeff', coeff(:, 1:n_dims), ...
                       'score', score(:, 1:n_dims), ...
                       'n_dims', n_dims);

    % --- Fit VB GMM using SPM for each K ---
    fprintf('4. Fitting VB-GMM models using SPM...\n');
    
    nK = numel(K_candidates);
    free_energy = zeros(nK, 1);
    silhouette_score = zeros(nK, 1);
    vbmix = cell(nK, 1);
    
    for iK = 1:nK
        K = K_candidates(iK);
        fprintf('   K=%d... ', K);
        
        try
            % Call SPM's spm_mix
            result = spm_mix(features, K, 0);
            
            if isnan(result.fm) || isinf(result.fm) || result.fm == 0
                fprintf('Invalid\n');
                free_energy(iK) = -Inf;
                silhouette_score(iK) = -1;
            else
                free_energy(iK) = result.fm;
                vbmix{iK} = result;
                
                % Compute silhouette score
                labels = assign_samples(features, result);
                sil = compute_silhouette_euclidean(features, labels);
                silhouette_score(iK) = sil;
                
                fprintf('F=%.1f, Sil=%.3f\n', result.fm, sil);
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            free_energy(iK) = -Inf;
            silhouette_score(iK) = -1;
        end
    end

    % --- Model Selection: Use ELBOW +/- SILHOUETTE ---
    valid_idx = ~isinf(free_energy) & free_energy ~= 0;
    
    if ~any(valid_idx)
        warning('All fits failed!');
        Results = create_empty_results(Sim, K_candidates, n_maps, C_dims, criterion);
        return;
    end
    
    % Detect elbow in free energy curve
    fe_valid = free_energy(valid_idx);
    K_valid = K_candidates(valid_idx);
    sil_valid = silhouette_score(valid_idx);
    
    % Compute elbow
    if length(fe_valid) >= 3
        % Normalize free energy to [0, 1]
        fe_norm = (fe_valid - min(fe_valid)) / (max(fe_valid) - min(fe_valid) + eps);
        k_norm = (K_valid - min(K_valid)) / (max(K_valid) - min(K_valid) + eps);
        
        % Find elbow using maximum curvature
        elbow_scores = zeros(size(fe_norm));
        for i = 2:length(fe_norm)-1
            % Distance to line connecting first and last points
            p1 = [k_norm(1), fe_norm(1)];
            p2 = [k_norm(end), fe_norm(end)];
            p = [k_norm(i), fe_norm(i)];
            
            % Perpendicular distance
            elbow_scores(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + ...
                p2(1)*p1(2) - p2(2)*p1(1)) / ...
                sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2);
        end
        
        % Find maximum distance (elbow)
        [~, elbow_idx] = max(elbow_scores);
        K_elbow = K_valid(elbow_idx);
    else
        K_elbow = K_valid(1);
        elbow_idx = 1;
    end
    
    % Select based on criterion
    if strcmp(criterion, 'elbow_only')
        % Use ELBOW ONLY
        best_idx_valid = elbow_idx;
        best_score = free_energy(find(valid_idx, elbow_idx, 'first'));
        fprintf('\n   Using ELBOW ONLY for model selection\n');
        
    else
        % Use COMBINED elbow + silhouette (default)
        combined_score = zeros(length(K_valid), 1);
        for i = 1:length(K_valid)
            % Penalty for distance from elbow
            elbow_penalty = exp(-abs(K_valid(i) - K_elbow));
            
            % Silhouette bonus (normalized to [0, 1])
            sil_bonus = (sil_valid(i) + 1) / 2;  % Map [-1, 1] to [0, 1]
            
            combined_score(i) = 0.6 * elbow_penalty + 0.4 * sil_bonus;
        end
        
        [best_score, best_idx_valid] = max(combined_score);
        fprintf('\n   Using COMBINED (elbow + silhouette) for model selection\n');
    end
    
    best_idx = find(valid_idx);
    best_idx = best_idx(best_idx_valid);
    
    K_estimated = K_candidates(best_idx);
    best_mix = vbmix{best_idx};
    
    fprintf('\n========================================\n');
    fprintf('MODEL SELECTION RESULTS\n');
    fprintf('========================================\n');
    fprintf('Elbow K: %d\n', K_elbow);
    fprintf('Best K: %d (score=%.3f)\n', K_estimated, best_score);
    fprintf('True K: %d\n', Sim.K_true);
    fprintf('\nModel Selection Landscape:\n');
    fprintf('   K      Free Energy  Silhouette\n');
    fprintf('   %s\n', repmat('-', 1, 45));
    
    for iK = 1:nK
        if valid_idx(iK)
            marker = '';
            if iK == best_idx
                marker = ' <-- SELECTED';
            elseif K_candidates(iK) == K_elbow
                marker = ' <-- ELBOW';
            end
            
            fprintf('  %2d      %10.1f    %8.3f%s\n', ...
                K_candidates(iK), free_energy(iK), ...
                silhouette_score(iK), marker);
        end
    end
    fprintf('\n');

    % --- Recover microstate centers from feature space ---
    fprintf('5. Recovering microstate topographies...\n');
    
    % Get cluster assignments
    labels = assign_samples(features, best_mix);
    
    % Recover centers in original map space with polarity-aware averaging
    centers = recover_centers_from_labels(maps_norm, labels, K_estimated);
    
    fprintf('   Recovered %d microstate centers\n', K_estimated);

    % --- Map Recovery ---
    fprintf('6. Computing map recovery...\n');
    
    true_maps_norm = Sim.maps_true - mean(Sim.maps_true, 2);
    true_maps_norm = true_maps_norm ./ (sqrt(sum(true_maps_norm.^2, 2)) + eps);
    
    % Best match recovery (Hungarian)
    recovery_corr = best_match_corr_hungarian_polarity_aware(true_maps_norm, centers);
    mean_recovery = mean(recovery_corr);
    
    % Average recovery per extracted state (regardless of K correct)
    % For each extracted center, find best match with any true state
    avg_recovery_per_state = mean(max(abs(centers * true_maps_norm'), [], 2));
    
    fprintf('   Mean recovery (best match): %.3f\n', mean_recovery);
    fprintf('   Avg recovery per extracted state: %.3f\n', avg_recovery_per_state);
    fprintf('   Per-state: %s\n\n', sprintf('%.3f ', recovery_corr));
    
    runtime = toc(t_start);

    % --- Return Results ---
    Results = struct( ...
        'method', 'spm_vb', ...
        'criterion', criterion, ...
        'K_true', Sim.K_true, ...
        'K_estimated', K_estimated, ...
        'K_elbow', K_elbow, ...
        'K_candidates', K_candidates, ...
        'SNR_dB', Sim.SNR_dB, ...
        'duration_s', Sim.duration_s, ...
        'n_maps', n_maps, ...
        'centers', centers, ...
        'maps_true', true_maps_norm, ...
        'labels', labels, ...
        'free_energy', free_energy, ...
        'silhouette_score', silhouette_score, ...
        'best_criterion_value', best_score, ...
        'maps_nc', maps_norm, ...
        'idx_peaks', idx_peaks, ...
        'gfp_vec', gfp_vec, ...
        'mean_recovery', mean_recovery, ...
        'recovery_corr', recovery_corr, ...
        'avg_recovery_per_state', avg_recovery_per_state, ...
        'valid_fit', true, ...
        'runtime', runtime, ...
        'vbmix', best_mix, ...
        'pca_model', pca_model);
end

% ======================== HELPER FUNCTIONS ========================

function labels = assign_samples(X, vbmix)
    % Assign samples to clusters using fitted VB-GMM
    [N, D] = size(X);
    K = vbmix.m;
    log_prob = zeros(N, K);
    
    for k = 1:K
        m = vbmix.state(k).m(:)';
        C = vbmix.state(k).C;
        
        % Ensure dimensions match
        if length(m) ~= D
            m = m(1:min(D, length(m)));
            if size(C, 1) ~= D
                C = C(1:min(D, size(C, 1)), 1:min(D, size(C, 2)));
            end
        end
        
        % Get prior (handle both .prior and .priors fields)
        if isfield(vbmix.state(k), 'prior')
            prior = vbmix.state(k).prior;
        elseif isfield(vbmix, 'priors') && length(vbmix.priors) >= k
            prior = vbmix.priors(k);
        else
            prior = 1 / K;  % Uniform if not available
        end
        
        % Log probability
        log_prob(:, k) = log(prior + eps) + log_mvnpdf(X, m, C);
    end
    
    % Maximum a posteriori assignment
    [~, labels] = max(log_prob, [], 2);
end

function log_p = log_mvnpdf(X, mu, Sigma)
    % Log of multivariate normal PDF (robust version)
    [N, D] = size(X);
    
    % Center data
    X_centered = X - repmat(mu, N, 1);
    
    % Regularize covariance if needed
    [U, S, ~] = svd(Sigma);
    s = diag(S);
    s(s < 1e-10) = 1e-10;  % Regularize small eigenvalues
    Sigma_reg = U * diag(s) * U';
    
    % Mahalanobis distance
    try
        L = chol(Sigma_reg, 'lower');
        z = L \ X_centered';
        maha = sum(z.^2, 1)';
    catch
        % Fallback if Cholesky fails
        Sigma_inv = pinv(Sigma_reg);
        maha = sum((X_centered * Sigma_inv) .* X_centered, 2);
    end
    
    % Log probability
    log_det = sum(log(s));
    log_p = -0.5 * (D*log(2*pi) + log_det + maha);
end

function sil = compute_silhouette_euclidean(X, labels)
    % Compute mean silhouette score using Euclidean distance
    K = max(labels);
    N = size(X, 1);
    
    if K < 2
        sil = 0;
        return;
    end
    
    sil_vals = zeros(N, 1);
    
    for i = 1:N
        k = labels(i);
        
        % Intra-cluster distance
        same_cluster = find(labels == k);
        if numel(same_cluster) > 1
            dists = sqrt(sum((X(same_cluster, :) - X(i, :)).^2, 2));
            a = mean(dists(dists > 0));  % Exclude self
        else
            a = 0;
        end
        
        % Inter-cluster distance (to nearest cluster)
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
            sil_vals(i) = (b - a) / max(a, b);
        end
    end
    
    sil = mean(sil_vals);
end

function centers = recover_centers_from_labels(maps, labels, K)
    % Recover microstate centers from labels with polarity-aware averaging
    [N, D] = size(maps);
    centers = zeros(K, D);
    
    for k = 1:K
        idx = find(labels == k);
        
        if isempty(idx)
            % Empty cluster - use random map
            centers(k, :) = maps(randi(N), :);
            continue;
        end
        
        % Initialize with first map in cluster
        ref = maps(idx(1), :);
        
        % Align all maps to reference (flip polarity if needed)
        aligned = zeros(length(idx), D);
        for i = 1:length(idx)
            m = maps(idx(i), :);
            if (m * ref') < 0
                aligned(i, :) = -m;  % Flip polarity
            else
                aligned(i, :) = m;
            end
        end
        
        % Average aligned maps
        centers(k, :) = mean(aligned, 1);
        
        % Normalize
        centers(k, :) = centers(k, :) - mean(centers(k, :));
        centers(k, :) = centers(k, :) / (norm(centers(k, :)) + eps);
    end
end

function Results = create_empty_results(Sim, K_candidates, n_maps, C_dims, criterion)
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
        'mean_recovery', NaN, ...
        'recovery_corr', NaN(Sim.K_true, 1), ...
        'avg_recovery_per_state', NaN, ...
        'valid_fit', false, ...
        'runtime', 0);
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
    % Proper Hungarian algorithm implementation
    n = size(costMat, 1);
    if n == 0, assignment = []; return; end
    
    % Step 1: Subtract row minima
    costMat = costMat - min(costMat, [], 2);
    
    % Step 2: Subtract column minima
    costMat = costMat - min(costMat, [], 1);
    
    % Step 3: Greedy assignment
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