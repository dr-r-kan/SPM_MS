function utils = microstate_utilities_SHARED()
% MICROSTATE_UTILITIES_SHARED: Consolidated shared utilities
%
% Returns struct of function handles to avoid duplication

    utils.preprocess_maps = @preprocess_maps_internal;
    utils.normalize_maps = @normalize_maps_internal;
    utils.bandpass_filter = @bandpass_fft_zero_phase_internal;
    utils.extract_gfp_peaks = @gfp_peak_maps_internal;
    utils.padded_vector = @pad_vector_internal;
    utils.format_method_name = @format_method_name_internal;
    utils.format_criterion_name = @format_criterion_name_internal;
    utils.progress_bar = @progbar_internal;
    utils.duration_string = @dur_str_internal;
    utils.on_off_string = @onoff_internal;
    utils.fibonacci_sphere = @fibonacci_sphere_internal;
end

% ======================== SIGNAL PROCESSING ========================

function [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original] = preprocess_maps_internal(Sim)
% Unified preprocessing: bandpass + GFP peak extraction + normalization
% NOW RETURNS maps_original for proper GEV calculation
    
    X_bp = bandpass_fft_zero_phase_internal(Sim.X_noisy, Sim.sfreq, [2 20]);
    [maps_nc, idx_peaks, gfp_vec] = gfp_peak_maps_internal(X_bp, 3);
    
    if isempty(maps_nc)
        warning('GFP peak extraction found no peaks; using uniform subsampling');
        idx_peaks = 1:2:size(X_bp, 2);
        maps_nc = X_bp(:, idx_peaks)';
    end
    
    n_maps = size(maps_nc, 1);
    C_dims = size(maps_nc, 2);
    maps_original = maps_nc;  % ← STORE BEFORE NORMALIZATION
    maps_norm = normalize_maps_internal(maps_nc);
end

function maps_norm = normalize_maps_internal(maps)
% Normalize maps to zero mean and unit norm
    
    if isempty(maps) || size(maps, 1) == 0
        maps_norm = maps;
        return;
    end
    
    maps_norm = maps - mean(maps, 2);
    norms = sqrt(sum(maps_norm.^2, 2));
    norms(norms < eps) = 1;
    maps_norm = maps_norm ./ norms;
end

function Xf = bandpass_fft_zero_phase_internal(X, sfreq, bp)
% Zero-phase bandpass filtering using FFT
    
    if isempty(bp) || numel(bp) ~= 2
        Xf = X;
        return;
    end
    
    T = size(X, 2);
    F = fft(X, [], 2);
    freqs = (0:T-1) * (sfreq / T);
    mask = (freqs >= bp(1) & freqs <= bp(2)) | (freqs >= sfreq - bp(2) & freqs <= sfreq - bp(1));
    F(:, ~mask) = 0;
    Xf = real(ifft(F, [], 2));
end

function [maps_nc, idx_peaks, gfp] = gfp_peak_maps_internal(X, min_dist)
% Extract maps at Global Field Power (GFP) peaks
    
    gfp = sqrt(mean((X - mean(X, 1)).^2, 1));
    
    for pct = [0.50, 0.60, 0.70, 0.80, 0.90]
        idx_peaks = find_local_peaks_internal(gfp, min_dist, pct);
        if ~isempty(idx_peaks) && length(idx_peaks) >= 3
            break;
        end
    end
    
    if isempty(idx_peaks)
        maps_nc = [];
    else
        maps_nc = X(:, idx_peaks)';
    end
end

function idx = find_local_peaks_internal(x, min_dist, pct)
% ✅ OPTIMIZED: Remove redundant loop
    
    x = x(:)';
    n = numel(x);
    if n < 3
        idx = 1:n;
        return;
    end
    
    % Single peak detection
    cand = find([false, x(2:end-1) > x(1:end-2) & x(2:end-1) >= x(3:end), false]);
    if isempty(cand)
        idx = [];
        return;
    end
    
    % Threshold (single computation)
    thr = quantile(x, pct);
    cand = cand(x(cand) >= thr);
    if isempty(cand)
        idx = [];
        return;
    end
    
    % Greedy assignment
    [~, ord] = sort(x(cand), 'descend');
    keep = false(1, n);
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

% ======================== VECTOR UTILITIES ========================

function v_pad = pad_vector_internal(v, n)
% Pad vector with NaN to length n
    
    v_pad = nan(1, n);
    if ~isempty(v) && ~all(isnan(v))
        v_pad(1:min(length(v), n)) = v(1:min(length(v), n));
    end
end

% ======================== PROGRESS & STRING UTILITIES ========================

function pb = progbar_internal(total, label)
% Simple progress bar
    
    if nargin < 2
        label = '';
    end
    
    c = struct('n', 0, 'N', total, 't0', tic, 'label', label);
    pb.update = @() local_update();
    pb.update_by = @(n) local_update_by(n);
    pb.done = @() fprintf('\n');
    
    function local_update()
        c.n = c.n + 1;
        if mod(c.n, max(1, floor(c.N/50))) == 0 || c.n == c.N
            dt = toc(c.t0);
            eta = dt * (c.N - c.n) / max(c.n, 1);
            fprintf('\r[%s] %d/%d (%.1f%%) ETA %s', c.label, c.n, c.N, 100*c.n/c.N, dur_str_internal(eta));
        end
    end
    
    function local_update_by(n)
        c.n = c.n + n;
        if mod(c.n, max(1, floor(c.N/50))) == 0 || c.n == c.N
            dt = toc(c.t0);
            eta = dt * (c.N - c.n) / max(c.n, 1);
            fprintf('\r[%s] %d/%d (%.1f%%) ETA %s', c.label, c.n, c.N, 100*c.n/c.N, dur_str_internal(eta));
        end
    end
end

function s = dur_str_internal(t)
% Format duration as string
    
    if t < 60
        s = sprintf('%.0fs', t);
    else
        m = floor(t / 60);
        ssec = mod(t, 60);
        s = sprintf('%dm%.0fs', m, ssec);
    end
end

function s = onoff_internal(bool_val)
% Convert boolean to ON/OFF string
    
    if bool_val
        s = 'ON';
    else
        s = 'OFF';
    end
end

% ======================== GEOMETRY ========================

function pos = fibonacci_sphere_internal(C)
% Generate uniformly distributed points on sphere using Fibonacci sequence
    
    ga = (sqrt(5) - 1) / 2;
    i = (0:C-1)' + 0.5;
    phi = 2 * pi * mod(i * ga, 1);
    z = 1 - 2 * i / C;
    r = sqrt(max(0, 1 - z.^2));
    pos = [r.*cos(phi), r.*sin(phi), z];
end

% ======================== DISPLAY FORMATTING ========================

function display_name = format_method_name_internal(method_code)
% FORMAT_METHOD_NAME_INTERNAL: Convert method code to display name
% Examples: 'kmeans_koenig' -> 'K-means', 'spm_vb' -> 'VB GMM'
    
    if ischar(method_code) || isstring(method_code)
        method_code = char(method_code);
    end
    
    switch method_code
        case 'kmeans_koenig'
            display_name = 'K-means';
        case 'spm_vb'
            display_name = 'VB GMM';
        case 'vb_kmeans'
            display_name = 'VB K-means';
        case 'dp_mixture'
            display_name = 'DP Mixture';
        otherwise
            display_name = method_code;  % Fallback to original
    end
end

function display_name = format_criterion_name_internal(criterion_code)
% FORMAT_CRITERION_NAME_INTERNAL: Convert criterion code to display name
% Examples: 'elbow_sil_combined' -> 'Elbow+Silhouette'
    
    if ischar(criterion_code) || isstring(criterion_code)
        criterion_code = char(criterion_code);
    end
    
    switch criterion_code
        case 'silhouette'
            display_name = 'Silhouette';
        case 'free_energy'
            display_name = 'Free Energy';
        case 'elbow'
            display_name = 'Elbow';
        case 'elbow_sil_combined'
            display_name = 'Elbow+Silhouette';
        case 'gev'
            display_name = 'GEV';
        otherwise
            display_name = criterion_code;  % Fallback to original
    end
end