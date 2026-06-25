function backfit = backfit_microstate_timecourse(Sim, Results, varargin)
%BACKFIT_MICROSTATE_TIMECOURSE Backfit full-record microstate weights.
%
% Hard backfitting uses polarity-invariant correlation. For GFP-peak based
% fits, the Koenig-style path assigns states at full-record GFP peaks and
% interpolates labels between those peaks.
% For SPM-VB fits, a proportional Gaussian soft backfit is also computed in
% topography space. The Gaussian widths are calibrated from each fitted
% cluster's within-cluster spread over GFP-peak maps, then applied to the
% full record to yield normalized per-timepoint cluster weights.

    p = inputParser;
    addRequired(p, 'Sim', @isstruct);
    addRequired(p, 'Results', @isstruct);
    parse(p, Sim, Results, varargin{:});

    backfit = struct('ok', false, 'message', '');
    if ~isfield(Sim, 'X_noisy') || isempty(Sim.X_noisy)
        backfit.message = 'Sim.X_noisy is missing.';
        return;
    end
    if ~isfield(Results, 'centers') || isempty(Results.centers)
        backfit.message = 'Results.centers is missing.';
        return;
    end

    util = microstate_utilities();
    try
        [X_fit, backfit_cfg] = preprocess_full_record_for_backfit_local(Sim, Results, util);
    catch ME
        backfit.message = ME.message;
        return;
    end
    maps_norm = util.normalize_maps(double(X_fit'));
    centers_norm = util.normalize_maps(double(Results.centers));

    if size(maps_norm, 2) ~= size(centers_norm, 2)
        backfit.message = sprintf('Channel mismatch after preprocessing/remap: data=%d, centers=%d.', ...
            size(maps_norm, 2), size(centers_norm, 2));
        return;
    end

    [n_samples, K_est] = size(abs(maps_norm * centers_norm'));
    [hard_assignments, hard_confidence, hard_debug] = koenig_hard_backfit_local( ...
        X_fit, maps_norm, centers_norm, Results, n_samples, util, backfit_cfg);
    hard_weights = zeros(n_samples, K_est);
    hard_weights(sub2ind([n_samples, K_est], (1:n_samples)', hard_assignments)) = 1;

    backfit.ok = true;
    backfit.message = 'ok';
    backfit.n_samples = n_samples;
    backfit.hard = struct( ...
        'weights', hard_weights, ...
        'assignments', hard_assignments, ...
        'confidence', hard_confidence, ...
        'mode', hard_debug.mode, ...
        'peak_sample_index', hard_debug.peak_sample_index, ...
        'peak_assignments', hard_debug.peak_assignments, ...
        'peak_confidence', hard_debug.peak_confidence, ...
        'instantaneous_assignments', hard_debug.instantaneous_assignments, ...
        'instantaneous_confidence', hard_debug.instantaneous_confidence);

    backfit.mixture = struct( ...
        'available', false, ...
        'message', 'Gaussian soft backfit requires saved peak-map labels and peak maps.', ...
        'weights', zeros(n_samples, K_est), ...
        'assignments', nan(n_samples, 1), ...
        'confidence', nan(n_samples, 1), ...
        'sigmas', nan(1, K_est));

    [supports_mixture, peak_maps, peak_labels] = supports_mixture_backfit(Results);
    if ~supports_mixture
        return;
    end

    try
        [resp, sigmas] = gaussian_soft_weights_topography(maps_norm, centers_norm, peak_maps, peak_labels);
        [mix_confidence, mix_assignments] = max(resp, [], 2);
        backfit.mixture = struct( ...
            'available', true, ...
            'message', 'ok', ...
            'weights', resp, ...
            'assignments', mix_assignments, ...
            'confidence', mix_confidence, ...
            'sigmas', sigmas);
    catch ME
        backfit.mixture.available = false;
        backfit.mixture.message = ME.message;
    end
end

function [tf, peak_maps, peak_labels] = supports_mixture_backfit(Results)
    tf = false;
    peak_maps = [];
    peak_labels = [];
    if ~isstruct(Results) || ~isfield(Results, 'centers') || isempty(Results.centers)
        return;
    end
    if ~isfield(Results, 'method') || ~strcmpi(char(string(Results.method)), 'spm_vb')
        return;
    end
    if ~isfield(Results, 'maps_nc') || isempty(Results.maps_nc)
        return;
    end

    peak_maps = double(Results.maps_nc);
    if isfield(Results, 'backfit_peak_labels') && ~isempty(Results.backfit_peak_labels)
        peak_labels = double(Results.backfit_peak_labels(:));
    elseif isfield(Results, 'labels') && ~isempty(Results.labels)
        peak_labels = double(Results.labels(:));
    else
        peak_labels = [];
    end

    if isempty(peak_labels)
        peak_maps = [];
        return;
    end
    if numel(peak_labels) ~= size(peak_maps, 1)
        peak_labels = peak_labels(1:min(numel(peak_labels), size(peak_maps, 1)));
        peak_maps = peak_maps(1:numel(peak_labels), :);
    end
    if isempty(peak_maps) || isempty(peak_labels)
        peak_maps = [];
        peak_labels = [];
        return;
    end
    tf = true;
end

function [X_fit, cfg] = preprocess_full_record_for_backfit_local(Sim, Results, util)
    X_fit = remap_full_record_if_needed_local(Sim, Results, util);
    cfg = resolve_backfit_preprocessing_cfg_local(Sim, Results);
    if isfield(cfg, 'apply_average_reference') && cfg.apply_average_reference
        X_fit = X_fit - mean(X_fit, 1);
    end
    if isfield(cfg, 'filter_band') && ~isempty(cfg.filter_band)
        X_fit = util.bandpass_filter(X_fit, Sim.sfreq, cfg.filter_band);
    end
end

function cfg = resolve_backfit_preprocessing_cfg_local(Sim, Results)
    cfg = struct();
    if isfield(Sim, 'preprocessing') && isstruct(Sim.preprocessing)
        cfg = Sim.preprocessing;
    end
    if isfield(Results, 'preprocessing') && isstruct(Results.preprocessing)
        cfg = merge_structs_local(cfg, Results.preprocessing);
    end
    if isfield(Results, 'preprocessing_info') && isstruct(Results.preprocessing_info)
        info = Results.preprocessing_info;
        fields = {'apply_average_reference', 'filter_band', 'spatial_filter', ...
            'gfp_peak_min_distance', 'gfp_peak_threshold_schedule'};
        for i = 1:numel(fields)
            f = fields{i};
            if isfield(info, f) && ~isempty(info.(f))
                cfg.(f) = info.(f);
            end
        end
    end
    if ~isfield(cfg, 'gfp_peak_min_distance') || isempty(cfg.gfp_peak_min_distance)
        cfg.gfp_peak_min_distance = 3;
    end
end

function out = merge_structs_local(base, extra)
    out = base;
    if ~isstruct(extra)
        return;
    end
    fields = fieldnames(extra);
    for i = 1:numel(fields)
        out.(fields{i}) = extra.(fields{i});
    end
end

function X_out = remap_full_record_if_needed_local(Sim, Results, util)
    X_out = double(Sim.X_noisy);
    target_n_channels = size(Results.centers, 2);
    if size(X_out, 1) == target_n_channels
        return;
    end
    if ~isfield(Results, 'backfit_support') || ~isstruct(Results.backfit_support)
        error('Channel mismatch: full record has %d channels, centers have %d, and no backfit remap metadata is available.', ...
            size(X_out, 1), target_n_channels);
    end
    support = Results.backfit_support;
    if ~isfield(support, 'channel_remap_spec') || isempty(support.channel_remap_spec)
        error('Channel mismatch: backfit remap metadata is missing channel_remap_spec.');
    end

    source_labels = infer_source_labels_local(Sim, util, size(X_out, 1));
    source_chanlocs = [];
    if isfield(Sim, 'chanlocs')
        source_chanlocs = Sim.chanlocs;
    end
    source_pos = [];
    if isfield(Sim, 'pos')
        source_pos = double(Sim.pos);
    end

    spec = support.channel_remap_spec;
    source_idx = double(spec.source_data_index(:));
    if isempty(source_idx)
        error('Channel remap spec has an empty source_data_index.');
    end
    if any(source_idx < 1) || any(source_idx > size(X_out, 1))
        error('Channel remap spec source_data_index is out of range for the supplied full record.');
    end

    source_subset = X_out(source_idx, :);
    if isfield(support, 'scalp_channel_labels') && ~isempty(support.scalp_channel_labels)
        expected_labels = cellstr(string(support.scalp_channel_labels(:)));
        actual_labels = source_labels(source_idx);
        if numel(actual_labels) == numel(expected_labels) && any(~strcmpi(cellstr(string(actual_labels(:))), expected_labels))
            warning('Backfit remap source labels do not exactly match the saved scalp channel labels. Proceeding with saved indices.');
        end
    elseif ~isempty(source_chanlocs)
        scalp_mask = infer_scalp_mask_local(source_labels, source_chanlocs, source_pos);
        if nnz(scalp_mask) == numel(source_idx)
            source_subset = X_out(find(scalp_mask), :);
        end
    end

    X_out = remap_full_record_to_target_local(source_subset, spec);
    if size(X_out, 1) ~= target_n_channels
        error('Backfit remap produced %d channels, expected %d.', size(X_out, 1), target_n_channels);
    end
end

function labels = infer_source_labels_local(Sim, util, n_channels)
    labels = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        labels = cellstr(string(Sim.channel_labels(:)));
    elseif isfield(Sim, 'chanlocs') && ~isempty(Sim.chanlocs)
        labels = util.channel_labels_from_chanlocs(Sim.chanlocs, n_channels);
    end
    if isempty(labels)
        labels = arrayfun(@(i) sprintf('Ch%d', i), 1:n_channels, 'UniformOutput', false)';
    end
end

function mask = infer_scalp_mask_local(labels, chanlocs, pos)
    n = numel(labels);
    mask = true(n, 1);
    bad_patterns = {'ECG','EKG','EOG','HEOG','VEOG','EMG','EXG','GSR','RESP','TRIG','TRIGGER','STI','STATUS','PHOTO','AUX','MISC','REF'};
    for i = 1:n
        lab = upper(strtrim(char(string(labels{i}))));
        for b = 1:numel(bad_patterns)
            if contains(lab, bad_patterns{b})
                mask(i) = false;
                break;
            end
        end
    end
    if ~isempty(pos) && size(pos, 1) == n
        has_pos = all(isfinite(pos), 2) & sqrt(sum(pos.^2, 2)) > 0;
        if nnz(has_pos & mask) >= 16
            mask = mask & has_pos;
        end
    elseif ~isempty(chanlocs) && numel(chanlocs) >= n
        has_xy = false(n, 1);
        for i = 1:n
            has_xy(i) = isfield(chanlocs, 'X') && ~isempty(chanlocs(i).X) && isfinite(double(chanlocs(i).X));
        end
        if nnz(has_xy & mask) >= 16
            mask = mask & has_xy;
        end
    end
end

function X_target = remap_full_record_to_target_local(source_subset, spec)
    n_target = double(spec.n_target_channels);
    n_samples = size(source_subset, 2);
    X_target = nan(n_target, n_samples);
    direct = double(spec.direct_local_index(:));
    observed = isfinite(direct);
    if any(observed)
        X_target(observed, :) = source_subset(direct(observed), :);
    end
    missing = find(~observed);
    for j = 1:numel(missing)
        t = missing(j);
        local_idx = double(spec.interpolation_source_local_index{t});
        weights = double(spec.interpolation_weights{t});
        if isempty(local_idx) || isempty(weights)
            error('Missing interpolation weights for target channel %d.', t);
        end
        X_target(t, :) = weights(:)' * source_subset(local_idx, :);
    end
end

function [weights, sigmas] = gaussian_soft_weights_topography(full_maps, centers_norm, peak_maps, peak_labels)
%GAUSSIAN_SOFT_WEIGHTS_TOPOGRAPHY Soft assignment from cluster-wise spread.

    peak_maps = double(peak_maps);
    peak_labels = double(peak_labels(:));
    K = size(centers_norm, 1);
    similarities = abs(full_maps * centers_norm');
    distances = 1 - similarities;
    sigmas = cluster_distance_sigmas(peak_maps, peak_labels, centers_norm, K);

    % Empirically, a half-width scale preserves hard top-1 behaviour while
    % improving overlap-weight recovery on the simulated mixtures.
    sigma_eff = max(0.5 * sigmas, eps);
    weights = exp(-0.5 * (distances .^ 2) ./ (sigma_eff .^ 2 + eps));
    weights = weights ./ max(sum(weights, 2), eps);
end

function sigmas = cluster_distance_sigmas(peak_maps, peak_labels, centers_norm, K)
    sigmas = zeros(1, K);
    global_d = 1 - abs(peak_maps * centers_norm');
    sigma_floor = sqrt(mean(global_d(:) .^ 2, 'omitnan') + eps);
    for k = 1:K
        idx = peak_labels == k;
        if any(idx)
            dk = 1 - abs(peak_maps(idx, :) * centers_norm(k, :)');
            sigmas(k) = sqrt(mean(dk .^ 2, 'omitnan') + eps);
        else
            sigmas(k) = sigma_floor;
        end
    end
    sigmas(~isfinite(sigmas) | sigmas < eps) = sigma_floor;
end

function [assignments, confidence, debug] = koenig_hard_backfit_local(X_fit, full_maps, centers_norm, Results, n_samples, util, cfg)
    instantaneous_similarity = abs(full_maps * centers_norm');
    [instantaneous_confidence, instantaneous_assignments] = max(instantaneous_similarity, [], 2);

    debug = struct( ...
        'mode', 'all_samples_direct', ...
        'peak_sample_index', nan(0, 1), ...
        'peak_assignments', nan(0, 1), ...
        'peak_confidence', nan(0, 1), ...
        'instantaneous_assignments', instantaneous_assignments, ...
        'instantaneous_confidence', instantaneous_confidence);
    assignments = instantaneous_assignments;
    confidence = instantaneous_confidence;

    [has_peak_fit, peak_maps, peak_sample_index] = full_record_peak_backfit_support_local(X_fit, cfg, util);
    if ~has_peak_fit
        [has_peak_fit, peak_maps, peak_sample_index] = supports_peak_backfit_local(Results, n_samples);
    end
    if has_peak_fit
        peak_maps = util.normalize_maps(double(peak_maps));
        if size(peak_maps, 2) == size(centers_norm, 2)
            peak_similarity = abs(peak_maps * centers_norm');
            [peak_confidence, peak_assignments] = max(peak_similarity, [], 2);
            [assignments, confidence] = interpolate_peak_assignments_local( ...
                peak_sample_index, peak_assignments, peak_confidence, n_samples);

            debug.mode = 'gfp_peaks_interpolated';
            debug.peak_sample_index = peak_sample_index(:);
            debug.peak_assignments = peak_assignments(:);
            debug.peak_confidence = peak_confidence(:);
        end
    end
end

function [tf, peak_maps, peak_sample_index] = full_record_peak_backfit_support_local(X_fit, cfg, util)
    tf = false;
    peak_maps = [];
    peak_sample_index = [];
    if isempty(X_fit)
        return;
    end

    min_dist = 3;
    if isfield(cfg, 'gfp_peak_min_distance') && ~isempty(cfg.gfp_peak_min_distance)
        min_dist = max(1, round(double(cfg.gfp_peak_min_distance)));
    end
    gfp = sqrt(mean((double(X_fit) - mean(double(X_fit), 1)).^2, 1));
    peak_sample_index = local_peak_finder_backfit_local(gfp, min_dist);
    if isempty(peak_sample_index)
        return;
    end

    peak_maps = double(X_fit(:, peak_sample_index))';
    valid = all(isfinite(peak_maps), 2);
    peak_maps = peak_maps(valid, :);
    peak_sample_index = peak_sample_index(valid);
    if isempty(peak_sample_index)
        return;
    end

    peak_maps = util.normalize_maps(peak_maps);
    tf = true;
end

function idx = local_peak_finder_backfit_local(x, min_dist)
    x = double(x(:)');
    idx = [];
    if numel(x) < 3
        return;
    end

    cand = find(x(2:end-1) >= x(1:end-2) & x(2:end-1) > x(3:end)) + 1;
    if isempty(cand)
        [~, best] = max(x);
        idx = best;
        return;
    end

    [~, order] = sort(x(cand), 'descend');
    cand = cand(order);
    taken = false(size(x));
    keep = false(size(cand));
    for i = 1:numel(cand)
        c = cand(i);
        left = max(1, c - min_dist);
        right = min(numel(x), c + min_dist);
        if any(taken(left:right))
            continue;
        end
        keep(i) = true;
        taken(left:right) = true;
    end
    idx = sort(cand(keep), 'ascend');
end

function [tf, peak_maps, peak_sample_index] = supports_peak_backfit_local(Results, n_samples)
    tf = false;
    peak_maps = [];
    peak_sample_index = [];
    if ~isstruct(Results) || ~isfield(Results, 'maps_nc') || isempty(Results.maps_nc)
        return;
    end

    peak_maps = double(Results.maps_nc);
    if isfield(Results, 'idx_peaks') && ~isempty(Results.idx_peaks)
        peak_sample_index = double(Results.idx_peaks(:));
    elseif isfield(Results, 'backfit_support') && isstruct(Results.backfit_support) && ...
            isfield(Results.backfit_support, 'peak_sample') && ~isempty(Results.backfit_support.peak_sample)
        peak_sample_index = double(Results.backfit_support.peak_sample(:));
    else
        peak_sample_index = [];
    end

    if isempty(peak_sample_index)
        peak_maps = [];
        return;
    end

    n_keep = min(numel(peak_sample_index), size(peak_maps, 1));
    peak_sample_index = peak_sample_index(1:n_keep);
    peak_maps = peak_maps(1:n_keep, :);
    valid = isfinite(peak_sample_index) & peak_sample_index >= 1 & peak_sample_index <= n_samples;
    peak_sample_index = round(peak_sample_index(valid));
    peak_maps = peak_maps(valid, :);
    if isempty(peak_sample_index) || isempty(peak_maps)
        peak_maps = [];
        peak_sample_index = [];
        return;
    end

    [peak_sample_index, sort_idx] = sort(peak_sample_index(:), 'ascend');
    peak_maps = peak_maps(sort_idx, :);
    [peak_sample_index, unique_idx] = unique(peak_sample_index, 'stable');
    peak_maps = peak_maps(unique_idx, :);
    tf = ~isempty(peak_sample_index) && ~isempty(peak_maps);
end

function [assignments, confidence] = interpolate_peak_assignments_local(peak_idx, peak_assignments, peak_confidence, n_samples)
    assignments = nan(n_samples, 1);
    confidence = nan(n_samples, 1);
    if isempty(peak_idx)
        return;
    end

    peak_idx = peak_idx(:);
    peak_assignments = peak_assignments(:);
    peak_confidence = peak_confidence(:);

    if numel(peak_idx) == 1
        assignments(:) = peak_assignments(1);
        confidence(:) = peak_confidence(1);
        return;
    end

    assignments(1:peak_idx(1)) = peak_assignments(1);
    confidence(1:peak_idx(1)) = peak_confidence(1);
    for i = 1:(numel(peak_idx) - 1)
        left = peak_idx(i);
        right = peak_idx(i + 1);
        midpoint = floor((left + right) / 2);
        assignments(left:min(midpoint, n_samples)) = peak_assignments(i);
        confidence(left:min(midpoint, n_samples)) = peak_confidence(i);
        assignments((midpoint + 1):min(right, n_samples)) = peak_assignments(i + 1);
        confidence((midpoint + 1):min(right, n_samples)) = peak_confidence(i + 1);
    end
    assignments(peak_idx(end):n_samples) = peak_assignments(end);
    confidence(peak_idx(end):n_samples) = peak_confidence(end);

    if any(~isfinite(assignments))
        missing = find(~isfinite(assignments));
        assignments(missing) = peak_assignments(end);
        confidence(missing) = peak_confidence(end);
    end
end
