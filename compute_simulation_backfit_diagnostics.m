function diagnostics = compute_simulation_backfit_diagnostics(Sim, Results, template_file, varargin)
% COMPUTE_SIMULATION_BACKFIT_DIAGNOSTICS Compare backfit coverage to truth.
%
% The true and estimated states are both projected into template-label
% space, then the continuous noisy record is backfit against the selected
% estimated maps using polarity-invariant correlation.

    p = inputParser;
    addRequired(p, 'Sim', @isstruct);
    addRequired(p, 'Results', @isstruct);
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_K', 7, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'strong_threshold', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    parse(p, Sim, Results, template_file, varargin{:});

    diagnostics = struct('ok', false, 'message', '');
    if ~isfield(Sim, 'z_true') || isempty(Sim.z_true)
        diagnostics.message = 'Sim.z_true is missing.';
        return;
    end
    if ~isfield(Results, 'centers') || isempty(Results.centers)
        diagnostics.message = 'Results.centers is missing.';
        return;
    end

    util = microstate_utilities();
    template_file = char(template_file);
    estimated_channel_labels = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        estimated_channel_labels = cellstr(Sim.channel_labels);
    end

    if isfield(Sim, 'true_template_alignment') && ~isempty(Sim.true_template_alignment)
        true_alignment = Sim.true_template_alignment;
    else
        true_alignment = align_microstates_to_template(Sim.maps_true, template_file, ...
            'estimated_channel_labels', estimated_channel_labels, ...
            'template_K', p.Results.template_K, ...
            'strong_threshold', p.Results.strong_threshold);
    end
    if isfield(Results, 'template_alignment') && ~isempty(Results.template_alignment)
        estimated_alignment = Results.template_alignment;
    else
        estimated_alignment = align_microstates_to_template(Results.centers, template_file, ...
            'estimated_channel_labels', estimated_channel_labels, ...
            'template_K', p.Results.template_K, ...
            'strong_threshold', p.Results.strong_threshold);
    end

    X_fit = preprocess_full_record_for_backfit(Sim, util);
    maps_norm = util.normalize_maps(X_fit');
    centers_norm = util.normalize_maps(double(Results.centers));
    sims = abs(maps_norm * centers_norm');
    [~, est_idx] = max(sims, [], 2);
    true_idx = Sim.z_true(:);

    true_state_labels = alignment_labels_or_fallback(true_alignment, size(Sim.maps_true, 1), 'T');
    est_state_labels = alignment_labels_or_fallback(estimated_alignment, size(Results.centers, 1), 'E');
    template_order = template_order_union(true_alignment, estimated_alignment, true_state_labels, est_state_labels);

    true_sample_labels = true_state_labels(true_idx);
    est_sample_labels = est_state_labels(est_idx);

    coverage_true = label_coverage(true_sample_labels, template_order);
    coverage_est = label_coverage(est_sample_labels, template_order);
    active = coverage_true > 0 | coverage_est > 0;
    if ~any(active)
        active(:) = true;
    end
    diff_vec = coverage_est(active) - coverage_true(active);
    if numel(diff_vec) >= 2 && numel(unique(coverage_true(active))) > 1 && numel(unique(coverage_est(active))) > 1
        coverage_corr = corr(coverage_true(active), coverage_est(active), 'Type', 'Pearson', 'Rows', 'complete');
        coverage_spearman = corr(coverage_true(active), coverage_est(active), 'Type', 'Spearman', 'Rows', 'complete');
    else
        coverage_corr = NaN;
        coverage_spearman = NaN;
    end

    confusion_counts = label_confusion_counts(true_sample_labels, est_sample_labels, template_order);
    row_sums = sum(confusion_counts, 2);
    confusion_row_normalized = confusion_counts ./ max(row_sums, 1);

    diagnostics.ok = true;
    diagnostics.message = 'ok';
    diagnostics.n_samples = numel(true_idx);
    diagnostics.template_labels = template_order(:);
    diagnostics.true_state_template_labels = true_state_labels(:);
    diagnostics.estimated_state_template_labels = est_state_labels(:);
    diagnostics.coverage_true = coverage_true(:);
    diagnostics.coverage_estimated = coverage_est(:);
    diagnostics.coverage_difference = (coverage_est(:) - coverage_true(:));
    diagnostics.coverage_corr = coverage_corr;
    diagnostics.coverage_spearman = coverage_spearman;
    diagnostics.coverage_mae = mean(abs(diff_vec), 'omitnan');
    diagnostics.coverage_rmse = sqrt(mean(diff_vec .^ 2, 'omitnan'));
    diagnostics.coverage_l1 = sum(abs(diff_vec), 'omitnan');
    diagnostics.coverage_linf = max(abs(diff_vec));
    diagnostics.confusion_counts = confusion_counts;
    diagnostics.confusion_row_normalized = confusion_row_normalized;
    diagnostics.true_alignment = true_alignment;
    diagnostics.estimated_alignment = estimated_alignment;
end

function X_fit = preprocess_full_record_for_backfit(Sim, util)
    X_fit = double(Sim.X_noisy);
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

function labels = alignment_labels_or_fallback(alignment, n_states, prefix)
    labels = cell(n_states, 1);
    if isstruct(alignment) && isfield(alignment, 'labels') && numel(alignment.labels) >= n_states
        raw = cellstr(alignment.labels);
        for i = 1:n_states
            labels{i} = char(string(raw{i}));
        end
        return;
    end
    for i = 1:n_states
        labels{i} = sprintf('%s%d', prefix, i);
    end
end

function order = template_order_union(true_alignment, est_alignment, true_labels, est_labels)
    order = {};
    if isstruct(true_alignment) && isfield(true_alignment, 'template_labels') && ~isempty(true_alignment.template_labels)
        order = [order; cellstr(true_alignment.template_labels(:))]; %#ok<AGROW>
    elseif isstruct(est_alignment) && isfield(est_alignment, 'template_labels') && ~isempty(est_alignment.template_labels)
        order = [order; cellstr(est_alignment.template_labels(:))]; %#ok<AGROW>
    end
    order = [order; true_labels(:); est_labels(:)]; %#ok<AGROW>
    order = unique(order, 'stable');
end

function coverage = label_coverage(sample_labels, label_order)
    n = numel(sample_labels);
    coverage = zeros(numel(label_order), 1);
    for i = 1:numel(label_order)
        coverage(i) = sum(strcmp(sample_labels, label_order{i})) / max(n, 1);
    end
end

function counts = label_confusion_counts(true_sample_labels, est_sample_labels, label_order)
    n_labels = numel(label_order);
    counts = zeros(n_labels, n_labels);
    for i = 1:n_labels
        mask_i = strcmp(true_sample_labels, label_order{i});
        for j = 1:n_labels
            counts(i, j) = sum(mask_i & strcmp(est_sample_labels, label_order{j}));
        end
    end
end
