function backfit = backfit_microstate_timecourse(Sim, Results, varargin)
%BACKFIT_MICROSTATE_TIMECOURSE Backfit full-record microstate weights.
%
% Hard backfitting is always available via polarity-invariant correlation.
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
        X_fit = preprocess_full_record_for_backfit_local(Sim, Results, util);
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

    hard_similarity = abs(maps_norm * centers_norm');
    [n_samples, K_est] = size(hard_similarity);
    [hard_confidence, hard_assignments] = max(hard_similarity, [], 2);
    hard_weights = zeros(n_samples, K_est);
    hard_weights(sub2ind([n_samples, K_est], (1:n_samples)', hard_assignments)) = 1;

    backfit.ok = true;
    backfit.message = 'ok';
    backfit.n_samples = n_samples;
    backfit.hard = struct( ...
        'weights', hard_weights, ...
        'assignments', hard_assignments, ...
        'confidence', hard_confidence);

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

function X_fit = preprocess_full_record_for_backfit_local(Sim, Results, util)
    X_fit = remap_full_record_if_needed_local(Sim, Results, util);
    cfg = struct();
    if isfield(Sim, 'preprocessing') && isstruct(Sim.preprocessing)
        cfg = Sim.preprocessing;
    end
    if isfield(cfg, 'apply_average_reference') && cfg.apply_average_reference
        X_fit = X_fit - mean(X_fit, 1);
    end
    if isfield(cfg, 'filter_band') && ~isempty(cfg.filter_band)
        X_fit = util.bandpass_filter(X_fit, Sim.sfreq, cfg.filter_band);
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
