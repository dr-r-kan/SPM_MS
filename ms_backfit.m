function Results = ms_backfit(hierarchical_results_mat, varargin)
% HARDENED_BACKFIT_HIERARCHICAL_MICROSTATES
%
% Robust backfitting for fit_microstate_hierarchical_dataset outputs.
%
% This is deliberately a post-hoc backfitter rather than a refitter.  It
% consumes hierarchical_microstate_results.mat and writes hardened backfit
% metrics that can be passed into postprocess_hierarchical_microstates via
% the 'existing_backfit_metrics' option.
%
% Main hardening choices:
%   1) Default backfitting uses global templates, not file-level templates.
%      File-level templates are useful for diagnostics, but make occurrence
%      and dwell-time comparisons less comparable across files.
%   2) Every map is channel-demeaned and unit-normed before spatial
%      correlation.
%   3) Low-GFP, low-correlation, and ambiguous assignments are labelled 0
%      and excluded from state-specific coverage/dwell estimates.
%   4) Per-file QC flags are written for dominant-state collapse, low valid
%      fraction, duplicate templates, low channel match, and low median
%      assignment correlation.
%
% REQUIRED INPUT
%   hierarchical_results_mat   path to hierarchical_microstate_results.mat
%
% NAME-VALUE OPTIONS
%   'output_dir'              default: <hierarchical output>/hardened_backfit
%   'manifest_csv'            optional manifest override
%   'template_level'          'global' [default], 'group_condition',
%                             'participant_condition', or 'file'
%   'filter_band'             default [2 20]; [] disables filtering
%   'ignore_polarity'         default true
%   'use_scalp_channels'      default true
%   'reject_bad_channels'     default true
%   'bad_channel_sd_ratio'    default 20; flags extreme channel SDs
%   'min_gfp_quantile'        default 0.10; low GFP samples become label 0
%   'min_abs_corr'            default 0.50; weak matches become label 0
%   'min_corr_margin'         default 0.02; ambiguous winner-vs-runner-up
%                             matches become label 0
%   'smooth_window_ms'        default 0; majority smoothing after thresholding
%   'min_segment_ms'          default 20; short positive segments are merged
%                             into positive neighbours or set to label 0
%   'dominance_qc_threshold'  default 0.95
%   'low_valid_qc_threshold'  default 0.50
%   'duplicate_template_corr' default 0.95
%   'sfreq_fallback'          default 250
%   'save_segmentation'       default true
%   'save_scores'             default false; can be large
%   'verbose'                 default true
%
% OUTPUTS
%   backfit_metrics_long.csv        state-wise metrics, compatible with the
%                                   postprocess_hierarchical_microstates code
%   backfit_qc_file_summary.csv     per-file diagnostics and flags
%   template_similarity_by_file.csv template duplicate diagnostics
%   hardened_backfit_results.mat
%
% Example:
%   R = hardened_backfit_hierarchical_microstates('hierarchical_microstate_results.mat', ...
%       'template_level', 'global', 'min_gfp_quantile', 0.10, ...
%       'min_abs_corr', 0.50, 'min_corr_margin', 0.02);
%
% Then:
%   postprocess_hierarchical_microstates('hierarchical_microstate_results.mat', ...
%       'existing_backfit_metrics', R.metrics_file, ...
%       'force_recompute_backfit', false);

    p = inputParser;
    addRequired(p, 'hierarchical_results_mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'manifest_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_level', 'global', @(x) ischar(x) || isstring(x));
    addParameter(p, 'filter_band', [2 20], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    addParameter(p, 'ignore_polarity', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'use_scalp_channels', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'reject_bad_channels', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'bad_channel_sd_ratio', 20, @(x) isnumeric(x) && isscalar(x) && x > 1);
    addParameter(p, 'min_gfp_quantile', 0.10, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 1);
    addParameter(p, 'min_abs_corr', 0.50, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'min_corr_margin', 0.02, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'smooth_window_ms', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'min_segment_ms', 20, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'dominance_qc_threshold', 0.95, @(x) isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
    addParameter(p, 'low_valid_qc_threshold', 0.50, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'duplicate_template_corr', 0.95, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'sfreq_fallback', 250, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'save_segmentation', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'save_scores', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'verbose', true, @(x) islogical(x) && isscalar(x));
    parse(p, hierarchical_results_mat, varargin{:});
    cfg = p.Results;

    cfg.hierarchical_results_mat = char(cfg.hierarchical_results_mat);
    cfg.output_dir = char(cfg.output_dir);
    cfg.manifest_csv = char(cfg.manifest_csv);
    cfg.template_level = lower(char(cfg.template_level));

    allowed_levels = {'global','group_condition','participant_condition','file'};
    if ~ismember(cfg.template_level, allowed_levels)
        error('template_level must be one of: %s', strjoin(allowed_levels, ', '));
    end
    if ~isfile(cfg.hierarchical_results_mat)
        error('Hierarchical results file not found: %s', cfg.hierarchical_results_mat);
    end

    S = load(cfg.hierarchical_results_mat, 'HResults');
    if ~isfield(S, 'HResults')
        error('No HResults variable found in: %s', cfg.hierarchical_results_mat);
    end
    H = S.HResults;
    if isfield(H, 'selected_K')
        K = H.selected_K;
    elseif isfield(H, 'global') && isfield(H.global, 'centers')
        K = size(H.global.centers, 1);
    else
        error('Could not determine selected K from HResults.');
    end

    if isempty(cfg.output_dir)
        [root_dir, ~, ~] = fileparts(cfg.hierarchical_results_mat);
        cfg.output_dir = fullfile(root_dir, 'hardened_backfit');
    end
    if ~exist(cfg.output_dir, 'dir'), mkdir(cfg.output_dir); end
    seg_dir = fullfile(cfg.output_dir, 'segmentations');
    if cfg.save_segmentation && ~exist(seg_dir, 'dir'), mkdir(seg_dir); end

    if ~isempty(cfg.manifest_csv)
        manifest = read_manifest_robust(cfg.manifest_csv);
    elseif isfield(H, 'manifest') && istable(H.manifest)
        manifest = H.manifest;
    else
        error('No manifest found in HResults and manifest_csv was not supplied.');
    end
    manifest = standardise_manifest_table(manifest);
    n_files = height(manifest);

    common_labels = {};
    if isfield(H, 'common_channel_labels')
        common_labels = cellstr(string(H.common_channel_labels));
    end

    if cfg.verbose
        fprintf('\n========================================\n');
        fprintf('Hardened hierarchical microstate backfit\n');
        fprintf('========================================\n');
        fprintf('Input:          %s\n', cfg.hierarchical_results_mat);
        fprintf('Output:         %s\n', cfg.output_dir);
        fprintf('Files:          %d\n', n_files);
        fprintf('Selected K:     %d\n', K);
        fprintf('Template level: %s\n', cfg.template_level);
        fprintf('Min GFP q:      %.3f\n', cfg.min_gfp_quantile);
        fprintf('Min abs corr:   %.3f\n', cfg.min_abs_corr);
        fprintf('Min margin:     %.3f\n', cfg.min_corr_margin);
        fprintf('========================================\n\n');
    end

    metric_tables = cell(n_files, 1);
    qc_rows = cell(n_files, 1);
    sim_tables = cell(n_files, 1);

    for i = 1:n_files
        eeg_file = char(manifest.file_path{i});
        if cfg.verbose
            fprintf('[%d/%d] %s\n', i, n_files, eeg_file);
        end

        templates = get_templates_for_file(H, manifest, i, cfg.template_level);
        if isempty(templates)
            error('Could not obtain templates for manifest row %d.', i);
        end
        templates = templates(1:min(K, size(templates,1)), :);

        [X, sfreq, labels, chanlocs] = load_eeg_matrix_local(eeg_file, cfg.sfreq_fallback);
        if cfg.use_scalp_channels && ~isempty(chanlocs)
            scalp_mask = scalp_channel_mask_local(chanlocs, size(X, 1));
            if any(scalp_mask) && nnz(scalp_mask) < size(X, 1)
                X = X(scalp_mask, :);
                labels = labels(scalp_mask);
            end
        end

        n_template_channels_original = size(templates, 2);
        if ~isempty(common_labels)
            [idx_eeg, idx_template] = match_channels(labels, common_labels);
            if numel(idx_eeg) < max(8, round(0.5 * numel(common_labels)))
                error('Too few channels matched for %s: %d/%d.', eeg_file, numel(idx_eeg), numel(common_labels));
            end
            X = X(idx_eeg, :);
            labels = labels(idx_eeg);
            templates = templates(:, idx_template);
        else
            if size(X, 1) ~= size(templates, 2)
                error('Channel count mismatch for %s: EEG=%d, templates=%d.', eeg_file, size(X,1), size(templates,2));
            end
        end

        kept_channel_mask = true(size(X,1), 1);
        if cfg.reject_bad_channels
            kept_channel_mask = robust_good_channel_mask(X, cfg.bad_channel_sd_ratio);
            if nnz(kept_channel_mask) >= max(8, round(0.5 * size(X, 1))) && nnz(kept_channel_mask) < size(X, 1)
                X = X(kept_channel_mask, :);
                labels = labels(kept_channel_mask);
                templates = templates(:, kept_channel_mask);
            else
                kept_channel_mask(:) = true;
            end
        end

        Seg = robust_backfit_matrix(X, sfreq, templates, cfg);
        Metrics = metrics_table_from_segmentation(Seg, manifest, i, K);
        QC = qc_table_from_segmentation(Seg, manifest, i, cfg, numel(labels), n_template_channels_original, nnz(kept_channel_mask));
        SimT = template_similarity_table(Seg.templates, manifest, i, cfg);

        metric_tables{i} = Metrics;
        qc_rows{i} = QC;
        sim_tables{i} = SimT;

        if cfg.save_segmentation
            safe = sprintf('%04d_%s_%s_%s', i, safe_label(manifest.participant{i}), safe_label(manifest.group{i}), safe_label(manifest.condition{i}));
            labels_seg = Seg.labels; %#ok<NASGU>
            valid_seg = Seg.valid; %#ok<NASGU>
            gfp = Seg.gfp; %#ok<NASGU>
            best_corr = Seg.best_corr; %#ok<NASGU>
            second_corr = Seg.second_corr; %#ok<NASGU>
            corr_margin = Seg.corr_margin; %#ok<NASGU>
            sfreq_seg = sfreq; %#ok<NASGU>
            templates_used = Seg.templates; %#ok<NASGU>
            channel_labels_used = labels; %#ok<NASGU>
            if cfg.save_scores
                score = Seg.score; %#ok<NASGU>
                save(fullfile(seg_dir, [safe '_hardened_segmentation.mat']), 'labels_seg', 'valid_seg', 'gfp', 'best_corr', 'second_corr', 'corr_margin', 'sfreq_seg', 'templates_used', 'channel_labels_used', 'score', '-v7.3');
            else
                save(fullfile(seg_dir, [safe '_hardened_segmentation.mat']), 'labels_seg', 'valid_seg', 'gfp', 'best_corr', 'second_corr', 'corr_margin', 'sfreq_seg', 'templates_used', 'channel_labels_used', '-v7.3');
            end
        end
    end

    MetricsLong = vertcat(metric_tables{:});
    QCFile = vertcat(qc_rows{:});
    TemplateSimilarity = vertcat(sim_tables{:});

    metrics_file = fullfile(cfg.output_dir, 'backfit_metrics_long.csv');
    qc_file = fullfile(cfg.output_dir, 'backfit_qc_file_summary.csv');
    sim_file = fullfile(cfg.output_dir, 'template_similarity_by_file.csv');
    writetable(MetricsLong, metrics_file);
    writetable(QCFile, qc_file);
    writetable(TemplateSimilarity, sim_file);

    Results = struct();
    Results.metrics = MetricsLong;
    Results.qc = QCFile;
    Results.template_similarity = TemplateSimilarity;
    Results.metrics_file = metrics_file;
    Results.qc_file = qc_file;
    Results.template_similarity_file = sim_file;
    Results.output_dir = cfg.output_dir;
    Results.cfg = cfg;
    Results.K = K;

    save(fullfile(cfg.output_dir, 'hardened_backfit_results.mat'), 'Results', '-v7.3');
    write_options_json(cfg, fullfile(cfg.output_dir, 'hardened_backfit_options.json'));

    if cfg.verbose
        fprintf('\nDone. Outputs:\n');
        fprintf('  %s\n', metrics_file);
        fprintf('  %s\n', qc_file);
        fprintf('  %s\n', sim_file);
        fprintf('\nFlag summary:\n');
        if height(QCFile) > 0
            fprintf('  dominant-state collapse: %d/%d\n', nnz(QCFile.flag_dominant_state), height(QCFile));
            fprintf('  low valid fraction:      %d/%d\n', nnz(QCFile.flag_low_valid_fraction), height(QCFile));
            fprintf('  duplicate templates:     %d/%d\n', nnz(QCFile.flag_duplicate_templates), height(QCFile));
            fprintf('  <2 states used:          %d/%d\n', nnz(QCFile.flag_lt_two_states_used), height(QCFile));
        end
    end
end

% ======================================================================
% Core backfitting
% ======================================================================

function Seg = robust_backfit_matrix(X, sfreq, templates, cfg)
    X = double(X);
    if ndims(X) > 2
        X = reshape(X, size(X, 1), []);
    end
    X(~isfinite(X)) = 0;

    % Re-reference and de-mean.  The template correlation is topographic:
    % amplitude is handled by GFP, not by the normalised map itself.
    X = X - mean(X, 1);
    X = X - mean(X, 2);

    if ~isempty(cfg.filter_band)
        X = fft_bandpass_local(X, sfreq, cfg.filter_band);
        X = X - mean(X, 1);
        X = X - mean(X, 2);
    end

    templates = normalize_maps_local(templates);
    K = size(templates, 1);
    T = size(X, 2);

    gfp = std(X, 0, 1);
    X_topo = X - mean(X, 1);
    norms = sqrt(sum(X_topo.^2, 1));
    Xn = X_topo ./ (norms + eps);

    corr_raw = templates * Xn;
    if cfg.ignore_polarity
        score = abs(corr_raw);
    else
        score = corr_raw;
    end

    [sorted_score, sorted_idx] = sort(score, 1, 'descend');
    best_corr = sorted_score(1, :);
    labels = sorted_idx(1, :);
    if K >= 2
        second_corr = sorted_score(2, :);
    else
        second_corr = zeros(1, T);
    end
    corr_margin = best_corr - second_corr;

    gfp_thr = quantile_fallback(gfp(isfinite(gfp)), cfg.min_gfp_quantile);
    valid = isfinite(gfp) & isfinite(best_corr) & ...
        gfp > gfp_thr & ...
        best_corr >= cfg.min_abs_corr & ...
        corr_margin >= cfg.min_corr_margin & ...
        norms > eps;

    labels(~valid) = 0;

    if cfg.smooth_window_ms > 0
        w = max(1, round(cfg.smooth_window_ms / 1000 * sfreq));
        labels = smooth_labels_ignore_zero(labels, w);
        labels(~valid) = 0;
    end

    if cfg.min_segment_ms > 0
        min_len = max(1, round(cfg.min_segment_ms / 1000 * sfreq));
        labels = merge_short_positive_segments(labels, min_len, score);
        valid = labels > 0;
    end

    idx_pos = find(labels > 0);
    best_corr_smoothed = nan(1, T);
    if ~isempty(idx_pos)
        lin_idx = sub2ind(size(score), labels(idx_pos), idx_pos);
        best_corr_smoothed(idx_pos) = score(lin_idx);
    end

    % GEV over valid samples only.  Invalid samples are not forced into the
    % denominator of state occupancy, but the valid fraction is reported.
    if any(valid)
        gev_valid = sum((gfp(valid) .* best_corr_smoothed(valid)).^2) / (sum(gfp(valid).^2) + eps);
    else
        gev_valid = NaN;
    end

    Seg = struct();
    Seg.labels = labels;
    Seg.valid = valid;
    Seg.gfp = gfp;
    Seg.gfp_threshold = gfp_thr;
    Seg.score = score;
    Seg.best_corr = best_corr_smoothed;
    Seg.second_corr = second_corr;
    Seg.corr_margin = corr_margin;
    Seg.templates = templates;
    Seg.sfreq = sfreq;
    Seg.n_samples = T;
    Seg.n_valid_samples = nnz(valid);
    Seg.valid_fraction = nnz(valid) / max(1, T);
    Seg.duration_s = T / sfreq;
    Seg.gev_valid = gev_valid;
end

function T = metrics_table_from_segmentation(Seg, manifest, file_index, K)
    labels = Seg.labels(:)';
    valid = labels > 0;
    gfp = Seg.gfp(:)';
    sfreq = Seg.sfreq;
    n_samples_total = numel(labels);
    n_valid = nnz(valid);
    duration_s = n_samples_total / sfreq;

    file_path = repmat(manifest.file_path(file_index), K, 1);
    participant = repmat(manifest.participant(file_index), K, 1);
    group = repmat(manifest.group(file_index), K, 1);
    condition = repmat(manifest.condition(file_index), K, 1);
    file_index_col = repmat(file_index, K, 1);
    state = (1:K)';

    coverage = nan(K, 1);
    coverage_total = nan(K, 1);
    occurrence_per_s = nan(K, 1);
    mean_dwell_ms = nan(K, 1);
    median_dwell_ms = nan(K, 1);
    mean_gfp = nan(K, 1);
    median_gfp = nan(K, 1);
    mean_abs_corr = nan(K, 1);
    median_abs_corr = nan(K, 1);
    n_segments = nan(K, 1);
    n_samples_state = nan(K, 1);

    [run_values, run_lengths] = run_length_encode(labels);
    for k = 1:K
        sample_idx = labels == k;
        n_samples_state(k) = nnz(sample_idx);
        coverage(k) = n_samples_state(k) / max(1, n_valid);       % valid-sample denominator
        coverage_total(k) = n_samples_state(k) / max(1, n_samples_total);
        seg_lengths = run_lengths(run_values == k);
        n_segments(k) = numel(seg_lengths);
        occurrence_per_s(k) = n_segments(k) / duration_s;
        if isempty(seg_lengths)
            mean_dwell_ms(k) = 0;
            median_dwell_ms(k) = 0;
        else
            mean_dwell_ms(k) = mean(seg_lengths) / sfreq * 1000;
            median_dwell_ms(k) = median(seg_lengths) / sfreq * 1000;
        end
        if any(sample_idx)
            mean_gfp(k) = mean(gfp(sample_idx), 'omitnan');
            median_gfp(k) = median(gfp(sample_idx), 'omitnan');
            mean_abs_corr(k) = mean(Seg.best_corr(sample_idx), 'omitnan');
            median_abs_corr(k) = median(Seg.best_corr(sample_idx), 'omitnan');
        end
    end

    T = table(file_index_col, file_path, participant, group, condition, state, ...
        coverage, coverage_total, occurrence_per_s, mean_dwell_ms, median_dwell_ms, ...
        mean_gfp, median_gfp, mean_abs_corr, median_abs_corr, n_segments, n_samples_state, ...
        repmat(duration_s, K, 1), repmat(n_samples_total, K, 1), repmat(n_valid, K, 1), ...
        repmat(Seg.valid_fraction, K, 1), repmat(Seg.gfp_threshold, K, 1), repmat(Seg.gev_valid, K, 1), ...
        'VariableNames', {'file_index','file_path','participant','group','condition','state', ...
        'coverage','coverage_total','occurrence_per_s','mean_dwell_ms','median_dwell_ms', ...
        'mean_gfp','median_gfp','mean_abs_corr','median_abs_corr','n_segments','n_samples_state', ...
        'duration_s','n_samples','n_valid_samples','valid_fraction','gfp_threshold','gev_valid'});
end

function T = qc_table_from_segmentation(Seg, manifest, file_index, cfg, n_channels_used, n_template_channels_original, n_channels_after_reject)
    labels = Seg.labels(:)';
    valid = labels > 0;
    K = size(Seg.templates, 1);
    cov = zeros(1, K);
    for k = 1:K
        cov(k) = nnz(labels == k) / max(1, nnz(valid));
    end
    [dominant_coverage, dominant_state] = max(cov);
    n_states_used_1pct = nnz(cov >= 0.01);
    n_states_used_5pct = nnz(cov >= 0.05);

    C = abs(Seg.templates * Seg.templates');
    C(1:K+1:end) = NaN;
    max_template_abs_corr = max(C(:), [], 'omitnan');
    if isempty(max_template_abs_corr) || ~isfinite(max_template_abs_corr)
        max_template_abs_corr = NaN;
    end

    valid_corr = Seg.best_corr(valid);
    valid_margin = Seg.corr_margin(valid);
    median_assignment_corr = median(valid_corr, 'omitnan');
    median_margin = median(valid_margin, 'omitnan');
    mean_assignment_corr = mean(valid_corr, 'omitnan');

    flag_dominant_state = dominant_coverage >= cfg.dominance_qc_threshold;
    flag_low_valid_fraction = Seg.valid_fraction < cfg.low_valid_qc_threshold;
    flag_duplicate_templates = max_template_abs_corr >= cfg.duplicate_template_corr;
    flag_lt_two_states_used = n_states_used_1pct < 2;
    flag_low_median_assignment_corr = median_assignment_corr < cfg.min_abs_corr;
    flag_low_channels = n_channels_after_reject < max(8, round(0.5 * n_template_channels_original));
    flag_any = flag_dominant_state || flag_low_valid_fraction || flag_duplicate_templates || flag_lt_two_states_used || flag_low_median_assignment_corr || flag_low_channels;

    notes = strings(0,1);
    if flag_dominant_state, notes(end+1) = "dominant_state"; end %#ok<AGROW>
    if flag_low_valid_fraction, notes(end+1) = "low_valid_fraction"; end %#ok<AGROW>
    if flag_duplicate_templates, notes(end+1) = "duplicate_templates"; end %#ok<AGROW>
    if flag_lt_two_states_used, notes(end+1) = "lt_two_states_used"; end %#ok<AGROW>
    if flag_low_median_assignment_corr, notes(end+1) = "low_median_assignment_corr"; end %#ok<AGROW>
    if flag_low_channels, notes(end+1) = "low_channels"; end %#ok<AGROW>
    note = strjoin(cellstr(notes), ';');

    T = table(file_index, manifest.file_path(file_index), manifest.participant(file_index), ...
        manifest.group(file_index), manifest.condition(file_index), ...
        Seg.duration_s, Seg.n_samples, Seg.n_valid_samples, Seg.valid_fraction, ...
        dominant_state, dominant_coverage, n_states_used_1pct, n_states_used_5pct, ...
        median_assignment_corr, mean_assignment_corr, median_margin, Seg.gev_valid, ...
        max_template_abs_corr, n_channels_used, n_channels_after_reject, n_template_channels_original, ...
        flag_dominant_state, flag_low_valid_fraction, flag_duplicate_templates, ...
        flag_lt_two_states_used, flag_low_median_assignment_corr, flag_low_channels, flag_any, string(note), ...
        'VariableNames', {'file_index','file_path','participant','group','condition', ...
        'duration_s','n_samples','n_valid_samples','valid_fraction', ...
        'dominant_state','dominant_coverage','n_states_used_1pct','n_states_used_5pct', ...
        'median_assignment_corr','mean_assignment_corr','median_corr_margin','gev_valid', ...
        'max_template_abs_corr','n_channels_used','n_channels_after_reject','n_template_channels_original', ...
        'flag_dominant_state','flag_low_valid_fraction','flag_duplicate_templates', ...
        'flag_lt_two_states_used','flag_low_median_assignment_corr','flag_low_channels','flag_any','qc_note'});
end

function T = template_similarity_table(templates, manifest, file_index, cfg)
    K = size(templates, 1);
    rows = cell(max(0, K*(K-1)/2), 9);
    r = 0;
    for a = 1:K
        for b = a+1:K
            c = templates(a,:) * templates(b,:)';
            r = r + 1;
            rows(r,:) = {file_index, manifest.file_path{file_index}, manifest.participant{file_index}, ...
                manifest.group{file_index}, manifest.condition{file_index}, a, b, abs(c), abs(c) >= cfg.duplicate_template_corr}; %#ok<AGROW>
        end
    end
    if r == 0
        T = table([], string.empty(0,1), string.empty(0,1), string.empty(0,1), string.empty(0,1), [], [], [], [], ...
            'VariableNames', {'file_index','file_path','participant','group','condition','state_a','state_b','abs_corr','is_duplicate'});
    else
        T = cell2table(rows(1:r,:), 'VariableNames', {'file_index','file_path','participant','group','condition','state_a','state_b','abs_corr','is_duplicate'});
        T.file_path = string(T.file_path);
        T.participant = string(T.participant);
        T.group = string(T.group);
        T.condition = string(T.condition);
    end
end

% ======================================================================
% Template selection
% ======================================================================

function templates = get_templates_for_file(H, manifest, i, level)
    templates = [];
    switch level
        case 'global'
            if isfield(H, 'global') && isfield(H.global, 'centers')
                templates = H.global.centers;
            end
        case 'file'
            if isfield(H, 'files') && numel(H.files) >= i && isfield(H.files(i), 'centers')
                templates = H.files(i).centers;
            end
        case 'participant_condition'
            if isfield(H, 'participant_conditions')
                sub = char(manifest.participant{i});
                g = char(manifest.group{i});
                c = char(manifest.condition{i});
                nodes = H.participant_conditions;
                idx = find(strcmp({nodes.participant}, sub) & strcmp({nodes.group}, g) & strcmp({nodes.condition}, c), 1, 'first');
                if ~isempty(idx), templates = nodes(idx).centers; end
            end
        case 'group_condition'
            if isfield(H, 'group_conditions')
                g = char(manifest.group{i});
                c = char(manifest.condition{i});
                nodes = H.group_conditions;
                idx = find(strcmp({nodes.group}, g) & strcmp({nodes.condition}, c), 1, 'first');
                if ~isempty(idx), templates = nodes(idx).centers; end
            end
    end
    if isempty(templates) && isfield(H, 'global') && isfield(H.global, 'centers')
        templates = H.global.centers;
    end
end

% ======================================================================
% Loading and manifest handling
% ======================================================================

function manifest = read_manifest_robust(csv_file)
    opts = detectImportOptions(csv_file, 'FileType', 'text', 'Delimiter', ',', 'TextType', 'string');
    opts.VariableNamingRule = 'preserve';
    manifest = readtable(csv_file, opts);
end

function manifest = standardise_manifest_table(manifest)
    names = string(manifest.Properties.VariableNames);
    clean = lower(regexprep(strtrim(names), '[^a-zA-Z0-9]+', '_'));
    manifest.Properties.VariableNames = cellstr(clean);

    file_aliases = {'file_path','filepath','path','file','eeg_file','eeg_path','filename'};
    group_aliases = {'group','grp'};
    condition_aliases = {'condition','cond','state','task'};
    participant_aliases = {'participant','subject','subj','id','subject_id','participant_id'};

    file_col = first_matching_col(manifest, file_aliases);
    group_col = first_matching_col(manifest, group_aliases);
    condition_col = first_matching_col(manifest, condition_aliases);
    participant_col = first_matching_col(manifest, participant_aliases);

    if isempty(file_col) || isempty(group_col) || isempty(condition_col)
        error('Manifest must contain file_path, group, and condition columns.');
    end
    if isempty(participant_col)
        participant = strings(height(manifest), 1);
        for i = 1:height(manifest)
            [~, nm, ~] = fileparts(string(manifest.(file_col)(i)));
            participant(i) = nm;
        end
    else
        participant = string(manifest.(participant_col));
    end

    file_path = cellstr(string(manifest.(file_col)));
    group = cellstr(string(manifest.(group_col)));
    condition = cellstr(string(manifest.(condition_col)));
    participant = cellstr(participant);
    manifest = table(file_path, participant, group, condition);
end

function col = first_matching_col(T, aliases)
    names = T.Properties.VariableNames;
    col = '';
    for i = 1:numel(aliases)
        idx = find(strcmp(names, aliases{i}), 1, 'first');
        if ~isempty(idx)
            col = names{idx};
            return;
        end
    end
end

function [X, sfreq, labels, chanlocs] = load_eeg_matrix_local(eeg_file, fallback_sfreq)
    [~, ~, ext] = fileparts(eeg_file);
    ext = lower(ext);
    chanlocs = [];
    switch ext
        case '.set'
            if exist('pop_loadset', 'file') ~= 2
                error('EEGLAB pop_loadset not found. Add EEGLAB to the MATLAB path before loading .set files.');
            end
            EEG = pop_loadset(eeg_file);
            X = double(EEG.data);
            sfreq = double(EEG.srate);
            if isfield(EEG, 'chanlocs'), chanlocs = EEG.chanlocs; end
            labels = labels_from_chanlocs(chanlocs, size(X,1));
        case '.mat'
            S = load(eeg_file);
            if isfield(S, 'eeg_data')
                X = double(S.eeg_data);
            elseif isfield(S, 'data')
                X = double(S.data);
            elseif isfield(S, 'EEG') && isfield(S.EEG, 'data')
                X = double(S.EEG.data);
            elseif isfield(S, 'X_clean')
                X = double(S.X_clean);
            else
                error('Could not find EEG data in .mat file: %s', eeg_file);
            end
            if ndims(X) > 2, X = reshape(X, size(X,1), []); end
            if isfield(S, 'sfreq')
                sfreq = double(S.sfreq);
            elseif isfield(S, 'srate')
                sfreq = double(S.srate);
            elseif isfield(S, 'EEG') && isfield(S.EEG, 'srate')
                sfreq = double(S.EEG.srate);
            else
                sfreq = fallback_sfreq;
            end
            if isfield(S, 'chanlocs')
                chanlocs = S.chanlocs;
            elseif isfield(S, 'EEG') && isfield(S.EEG, 'chanlocs')
                chanlocs = S.EEG.chanlocs;
            end
            if isfield(S, 'channel_labels')
                labels = cellstr(string(S.channel_labels));
            elseif isfield(S, 'ch_labels')
                labels = cellstr(string(S.ch_labels));
            else
                labels = labels_from_chanlocs(chanlocs, size(X,1));
            end
        otherwise
            error('Unsupported EEG extension %s. This helper supports .set and .mat.', ext);
    end
end

function labels = labels_from_chanlocs(chanlocs, n)
    labels = cell(n, 1);
    if ~isempty(chanlocs) && isfield(chanlocs, 'labels') && numel(chanlocs) >= n
        for i = 1:n
            labels{i} = char(string(chanlocs(i).labels));
            if isempty(labels{i}), labels{i} = sprintf('Ch%03d', i); end
        end
    else
        for i = 1:n
            labels{i} = sprintf('Ch%03d', i);
        end
    end
end

function mask = scalp_channel_mask_local(chanlocs, n)
    mask = true(n,1);
    if isempty(chanlocs) || ~isfield(chanlocs, 'labels'), return; end
    bad_patterns = {'ecg','ekg','eog','emg','stim','status','trigger','resp','gsr','eda','misc'};
    for i = 1:min(n, numel(chanlocs))
        lab = lower(char(string(chanlocs(i).labels)));
        for j = 1:numel(bad_patterns)
            if contains(lab, bad_patterns{j})
                mask(i) = false;
                break;
            end
        end
    end
end

function [idx_eeg, idx_template] = match_channels(eeg_labels, template_labels)
    eeg_can = canonical_labels(eeg_labels);
    temp_can = canonical_labels(template_labels);
    idx_eeg = [];
    idx_template = [];
    for j = 1:numel(temp_can)
        ii = find(strcmp(eeg_can, temp_can{j}), 1, 'first');
        if ~isempty(ii)
            idx_eeg(end+1) = ii; %#ok<AGROW>
            idx_template(end+1) = j; %#ok<AGROW>
        end
    end
end

function c = canonical_labels(labels)
    labels = cellstr(string(labels));
    c = cell(size(labels));
    for i = 1:numel(labels)
        s = lower(strtrim(labels{i}));
        s = regexprep(s, '[^a-z0-9]', '');
        c{i} = s;
    end
end

% ======================================================================
% Numerical helpers
% ======================================================================

function good = robust_good_channel_mask(X, ratio)
    X = double(X);
    X(~isfinite(X)) = 0;
    sd = std(X, 0, 2);
    med_sd = median(sd(sd > 0 & isfinite(sd)), 'omitnan');
    if isempty(med_sd) || ~isfinite(med_sd) || med_sd <= 0
        good = true(size(X,1), 1);
        return;
    end
    good = isfinite(sd) & sd > med_sd / ratio & sd < med_sd * ratio;
end

function Xf = fft_bandpass_local(X, sfreq, band)
    band = sort(double(band(:)'));
    if band(1) <= 0 && band(2) >= sfreq/2
        Xf = X;
        return;
    end
    T = size(X, 2);
    F = fft(X, [], 2);
    freqs = (0:T-1) * (sfreq / T);
    mask = (freqs >= band(1) & freqs <= band(2)) | (freqs >= sfreq - band(2) & freqs <= sfreq - band(1));
    F(:, ~mask) = 0;
    Xf = real(ifft(F, [], 2));
end

function Xn = normalize_maps_local(X)
    X = double(X);
    X = X - mean(X, 2);
    Xn = X ./ (sqrt(sum(X.^2, 2)) + eps);
end

function q = quantile_fallback(x, p)
    x = sort(x(:));
    x = x(isfinite(x));
    if isempty(x)
        q = NaN;
        return;
    end
    p = min(max(p, 0), 1);
    pos = 1 + (numel(x) - 1) * p;
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        q = x(lo);
    else
        q = x(lo) + (x(hi) - x(lo)) * (pos - lo);
    end
end

function labels_s = smooth_labels_ignore_zero(labels, window)
    labels = labels(:)';
    labels_s = labels;
    if window <= 1, return; end
    half = floor(window / 2);
    for t = 1:numel(labels)
        a = max(1, t-half);
        b = min(numel(labels), t+half);
        vals = labels(a:b);
        vals = vals(vals > 0);
        if ~isempty(vals)
            labels_s(t) = mode(vals);
        end
    end
end

function labels_out = merge_short_positive_segments(labels, min_len, score)
    labels_out = labels(:)';
    if min_len <= 1, return; end
    changed = true;
    while changed
        changed = false;
        [vals, lens, starts, ends_] = run_length_encode(labels_out);
        for r = 1:numel(vals)
            if vals(r) == 0 || lens(r) >= min_len || numel(vals) == 1
                continue;
            end
            a = starts(r);
            b = ends_(r);
            candidates = [];
            if r > 1 && vals(r-1) > 0, candidates(end+1) = vals(r-1); end %#ok<AGROW>
            if r < numel(vals) && vals(r+1) > 0, candidates(end+1) = vals(r+1); end %#ok<AGROW>
            candidates = unique(candidates, 'stable');
            if isempty(candidates)
                labels_out(a:b) = 0;
            else
                sc = zeros(numel(candidates), 1);
                for ci = 1:numel(candidates)
                    idx = sub2ind(size(score), repmat(candidates(ci), 1, b-a+1), a:b);
                    sc(ci) = mean(score(idx), 'omitnan');
                end
                [~, best] = max(sc);
                labels_out(a:b) = candidates(best);
            end
            changed = true;
            break;
        end
    end
end

function [vals, lens, starts, ends_] = run_length_encode(labels)
    labels = labels(:)';
    if isempty(labels)
        vals = []; lens = []; starts = []; ends_ = [];
        return;
    end
    starts = [1, find(diff(labels) ~= 0) + 1];
    ends_ = [starts(2:end) - 1, numel(labels)];
    vals = labels(starts);
    lens = ends_ - starts + 1;
end

function s = safe_label(x)
    s = char(string(x));
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    if isempty(s), s = 'unknown'; end
end

function write_options_json(cfg, out_file)
    try
        txt = jsonencode(cfg);
        fid = fopen(out_file, 'w');
        fwrite(fid, txt, 'char');
        fclose(fid);
    catch
        % JSON is only a convenience; the .mat result still contains cfg.
    end
end
