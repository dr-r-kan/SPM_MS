function [HResults, results_mat] = fit_microstate_hierarchical_dataset(input_path, varargin)
% FIT_MICROSTATE_HIERARCHICAL_DATASET Dataset-wise hierarchical SPM-VB microstates.
%
% Pipeline:
%   preprocess/filter -> GFP peaks -> global SPM-VB -> group/condition/
%   participant SPM-VB fits seeded by parent template pseudo-counts.
%
% Example:
%   H = fit_microstate_hierarchical_dataset('LEMON');

    if nargin < 1 || isempty(input_path)
        input_path = 'LEMON';
    end

    t0 = tic;
    util = microstate_utilities();
    repo_cfg = util.load_config();
    path_defaults = repo_cfg.paths;
    pre_defaults = repo_cfg.preprocessing;
    hier_defaults = repo_cfg.hierarchical;

    p = inputParser;
    addRequired(p, 'input_path', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', path_defaults.hierarchical_output_dir, @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', path_defaults.template_file, @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', double(hier_defaults.K_candidates(:)'), @isnumeric);
    addParameter(p, 'criterion', char(hier_defaults.criterion), @(x) ischar(x) || isstring(x));
    addParameter(p, 'apply_average_reference', logical(pre_defaults.apply_average_reference), @islogical);
    addParameter(p, 'spatial_filter', char(pre_defaults.spatial_filter), @(x) ischar(x) || isstring(x));
    addParameter(p, 'spatial_filter_neighbours', double(pre_defaults.spatial_filter_neighbours), @isnumeric);
    addParameter(p, 'spatial_filter_strength', double(pre_defaults.spatial_filter_strength), @isnumeric);
    addParameter(p, 'filter_band', double(pre_defaults.filter_band(:)'), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'use_scalp_channels', true, @islogical);
    addParameter(p, 'exclude_channels', {'PO9', 'PO10'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'interpolate_missing_channels', false, @islogical);
    addParameter(p, 'gfp_peak_min_distance', 3, @isnumeric);
    addParameter(p, 'gfp_peak_threshold_schedule', [0.50 0.60 0.70 0.80 0.90], @isnumeric);
    addParameter(p, 'min_gfp_peaks_per_file', 20, @isnumeric);
    addParameter(p, 'reject_gfp_peak_outliers', logical(pre_defaults.reject_gfp_peak_outliers), @islogical);
    addParameter(p, 'gfp_outlier_mad_multiplier', double(pre_defaults.gfp_outlier_mad_multiplier), @isnumeric);
    addParameter(p, 'max_maps_per_file', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_global_maps', double(hier_defaults.max_global_maps), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'spm_prior_pseudocount', double(hier_defaults.spm_prior_pseudocount), @isnumeric);
    addParameter(p, 'run_backfit', true, @islogical);
    addParameter(p, 'backfit_state_metrics_csv', 'participant_condition_state_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_pairwise_metrics_csv', 'participant_condition_state_pairwise_backfit_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_record_summary_csv', 'participant_condition_record_backfit_summary.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'run_backfit_axis_dynamics', false, @islogical);
    addParameter(p, 'backfit_axis_dynamics_csv', 'participant_condition_state_axis_dynamics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'backfit_axis_min_samples', 16, @isnumeric);
    addParameter(p, 'random_seed', 1, @isnumeric);
    addParameter(p, 'save_plots', true, @islogical);
    addParameter(p, 'verbose', true, @islogical);
    parse(p, input_path, varargin{:});
    cfg = p.Results;

    cfg.input_path = char(cfg.input_path);
    cfg.output_dir = util.resolve_path(cfg.output_dir, util.project_root());
    cfg.template_file = util.resolve_path(cfg.template_file, util.project_root());
    cfg.criterion = lower(char(cfg.criterion));
    cfg.spatial_filter = char(cfg.spatial_filter);
    cfg.exclude_channels = cellstr(string(cfg.exclude_channels));
    cfg.backfit_state_metrics_csv = char(cfg.backfit_state_metrics_csv);
    cfg.backfit_pairwise_metrics_csv = char(cfg.backfit_pairwise_metrics_csv);
    cfg.backfit_record_summary_csv = char(cfg.backfit_record_summary_csv);
    cfg.backfit_axis_dynamics_csv = char(cfg.backfit_axis_dynamics_csv);
    cfg.K_candidates = unique(round(double(cfg.K_candidates(:)')));
    cfg.K_candidates = cfg.K_candidates(cfg.K_candidates >= 2);
    if isempty(cfg.K_candidates)
        error('K_candidates must contain at least one value >= 2.');
    end
    rng(cfg.random_seed, 'twister');

    ensure_dir(cfg.output_dir);
    dirs = make_output_dirs(cfg.output_dir);

    [spm_ok, spm_info] = util.ensure_spm_mix('', path_defaults.spm_mixture_paths, cfg.verbose);
    if ~spm_ok
        error('SPM spm_mix not found. Checked: %s', strjoin(spm_info.attempted, ', '));
    end

    manifest = read_or_build_manifest(cfg.input_path);
    writetable(manifest, fullfile(cfg.output_dir, 'normalised_input_manifest.csv'));
    if cfg.verbose
        fprintf('\nHierarchical SPM-VB microstates\n');
        fprintf('Input rows: %d\nOutput: %s\n', height(manifest), cfg.output_dir);
        fprintf('K candidates: %s\n', mat2str(cfg.K_candidates));
        fprintf('Excluded channels: %s\n', strjoin(cfg.exclude_channels, ', '));
        fprintf('Channel interpolation: %s\n\n', onoff(cfg.interpolate_missing_channels));
    end

    meta = inspect_files(manifest, cfg, util);
    [common_labels, common_can, common_chanlocs, common_pos] = common_channel_set(meta, cfg);
    if numel(common_labels) < 8
        error('Only %d common channels remain after filtering/exclusion.', numel(common_labels));
    end
    writetable(table((1:numel(common_labels))', string(common_labels(:)), ...
        'VariableNames', {'index', 'label'}), fullfile(cfg.output_dir, 'common_channels.csv'));

    peak_records = cell(height(manifest), 1);
    pooled_maps = [];
    pooled_gfp = [];
    pooled_rows = table();
    for i = 1:height(manifest)
        if cfg.verbose
            fprintf('[%d/%d] GFP peaks: %s %s\n', i, height(manifest), manifest.participant(i), manifest.condition(i));
        end
        rec = extract_file_peaks(manifest.file_path(i), meta(i), common_can, common_pos, cfg, util);
        peak_records{i} = rec;
        save(fullfile(dirs.peaks, sprintf('%03d_%s_%s_gfp_peaks.mat', i, clean_key(manifest.participant(i)), clean_key(manifest.condition(i)))), ...
            'rec', 'cfg', '-v7.3');

        n = size(rec.maps_norm, 1);
        pooled_maps = [pooled_maps; double(rec.maps_norm)]; %#ok<AGROW>
        pooled_gfp = [pooled_gfp; double(rec.gfp_peak(:))]; %#ok<AGROW>
        rows = table();
        rows.file_index = repmat(i, n, 1);
        rows.participant = repmat(manifest.participant(i), n, 1);
        rows.condition = repmat(manifest.condition(i), n, 1);
        rows.group = repmat(manifest.group(i), n, 1);
        rows.file_path = repmat(manifest.file_path(i), n, 1);
        rows.peak_sample = rec.peak_sample(:);
        rows.gfp = double(rec.gfp_peak(:));
        pooled_rows = [pooled_rows; rows]; %#ok<AGROW>
    end
    writetable(pooled_rows, fullfile(cfg.output_dir, 'pooled_gfp_peak_manifest.csv'));

    fit_idx = deterministic_subsample(size(pooled_maps, 1), cfg.max_global_maps);
    global_maps = pooled_maps(fit_idx, :);
    global_gfp = pooled_gfp(fit_idx);
    if cfg.verbose && numel(fit_idx) < size(pooled_maps, 1)
        fprintf('Global fit uses deterministic cap: %d/%d peaks\n', numel(fit_idx), size(pooled_maps, 1));
    end

    if cfg.verbose
        fprintf('\nGlobal SPM-VB fit...\n');
    end
    global_fit = fit_spm_vb_peak_maps(global_maps, global_gfp, cfg.K_candidates, cfg, 'global', []);
    global_fit = attach_template_alignment_if_available(global_fit, cfg, common_labels);
    global_rows = pooled_rows(fit_idx, :);
    global_rows.cluster_label = assign_by_abs_correlation(global_maps, global_fit.centers);
    save_fit_bundle(dirs.global, 'global', global_fit, global_rows, common_labels, common_chanlocs, cfg);

    K = global_fit.K_estimated;
    child_cfg = cfg;
    child_cfg.K_candidates = K;

    [group_nodes, group_summary] = fit_group_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, child_cfg, dirs.groups, common_labels, common_chanlocs);
    [condition_nodes, condition_summary] = fit_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, child_cfg, dirs.conditions, common_labels, common_chanlocs);
    [participant_nodes, participant_summary] = fit_participant_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, child_cfg, dirs.participants, common_labels, common_chanlocs);
    [participant_condition_nodes, participant_condition_summary] = fit_participant_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, participant_nodes, child_cfg, dirs.participant_conditions, common_labels, common_chanlocs);

    hierarchy_summary = [ ...
        make_summary_row('global', "global", "", "", "", global_fit, height(manifest), size(global_maps, 1), dirs.global); ...
        group_summary; condition_summary; participant_summary; participant_condition_summary];
    writetable(hierarchy_summary, fullfile(cfg.output_dir, 'hierarchical_fit_summary.csv'));

    backfit_state_metrics = table();
    backfit_pairwise_metrics = table();
    backfit_record_summary = table();
    backfit_axis_dynamics = table();
    backfit_state_metrics_csv = '';
    backfit_pairwise_metrics_csv = '';
    backfit_record_summary_csv = '';
    backfit_axis_dynamics_csv = '';
    if cfg.run_backfit
        if cfg.verbose
            fprintf('\nBackfitting participant-condition templates to full records...\n');
        end
        [backfit_state_metrics, backfit_pairwise_metrics, backfit_record_summary, backfit_axis_dynamics] = ...
            compile_backfit_metric_tables(manifest, participant_nodes, participant_condition_nodes, common_can, common_pos, cfg, util);
        backfit_state_metrics_csv = output_csv_path(cfg.output_dir, cfg.backfit_state_metrics_csv);
        backfit_pairwise_metrics_csv = output_csv_path(cfg.output_dir, cfg.backfit_pairwise_metrics_csv);
        backfit_record_summary_csv = output_csv_path(cfg.output_dir, cfg.backfit_record_summary_csv);
        backfit_axis_dynamics_csv = output_csv_path(cfg.output_dir, cfg.backfit_axis_dynamics_csv);
        if ~isempty(backfit_state_metrics)
            writetable(backfit_state_metrics, backfit_state_metrics_csv);
        end
        if ~isempty(backfit_pairwise_metrics)
            writetable(backfit_pairwise_metrics, backfit_pairwise_metrics_csv);
        end
        if ~isempty(backfit_record_summary)
            writetable(backfit_record_summary, backfit_record_summary_csv);
        end
        if cfg.run_backfit_axis_dynamics && ~isempty(backfit_axis_dynamics)
            writetable(backfit_axis_dynamics, backfit_axis_dynamics_csv);
        end
    end

    HResults = struct();
    HResults.source = 'fit_microstate_hierarchical_dataset';
    HResults.created = datestr(now, 30);
    HResults.cfg = cfg;
    HResults.manifest = manifest;
    HResults.common_labels = common_labels;
    HResults.common_channel_labels = common_labels;
    HResults.common_chanlocs = common_chanlocs;
    HResults.common_pos = common_pos;
    HResults.selected_K = K;
    HResults.global = build_node('global', "", "all", "", "global", global_fit, size(global_maps, 1), dirs.global);
    HResults.groups = group_nodes;
    HResults.conditions = condition_nodes;
    HResults.participants = participant_nodes;
    HResults.participant_level = participant_nodes;
    HResults.participant_conditions = participant_condition_nodes;
    HResults.participant_condition_level = participant_condition_nodes;
    HResults.hierarchy_summary = hierarchy_summary;
    HResults.peak_records = peak_records;
    HResults.backfit_state_metrics = backfit_state_metrics;
    HResults.backfit_state_metrics_csv = backfit_state_metrics_csv;
    HResults.backfit_pairwise_metrics = backfit_pairwise_metrics;
    HResults.backfit_pairwise_metrics_csv = backfit_pairwise_metrics_csv;
    HResults.backfit_record_summary = backfit_record_summary;
    HResults.backfit_record_summary_csv = backfit_record_summary_csv;
    HResults.backfit_axis_dynamics = backfit_axis_dynamics;
    HResults.backfit_axis_dynamics_csv = backfit_axis_dynamics_csv;
    HResults.runtime_s = toc(t0);

    results_mat = fullfile(cfg.output_dir, 'hierarchical_microstate_results.mat');
    save(results_mat, 'HResults', '-v7.3');

    if cfg.verbose
        fprintf('\nDone in %.1f s\nResults: %s\n', HResults.runtime_s, results_mat);
        if ~isempty(backfit_state_metrics_csv)
            fprintf('Backfit state metrics CSV: %s\n', backfit_state_metrics_csv);
            fprintf('Backfit pairwise metrics CSV: %s\n', backfit_pairwise_metrics_csv);
            fprintf('Backfit record summary CSV: %s\n', backfit_record_summary_csv);
            if cfg.run_backfit_axis_dynamics && ~isempty(backfit_axis_dynamics_csv)
                fprintf('Backfit axis dynamics CSV: %s\n', backfit_axis_dynamics_csv);
            end
        end
        print_alignment_line('Global', global_fit);
    end
end

function dirs = make_output_dirs(root_dir)
    dirs = struct();
    dirs.root = root_dir;
    dirs.peaks = fullfile(root_dir, 'gfp_peaks');
    dirs.global = fullfile(root_dir, 'global');
    dirs.groups = fullfile(root_dir, 'groups');
    dirs.conditions = fullfile(root_dir, 'conditions');
    dirs.participants = fullfile(root_dir, 'participants');
    dirs.participant_conditions = fullfile(root_dir, 'participant_conditions');
    names = fieldnames(dirs);
    for i = 1:numel(names)
        ensure_dir(dirs.(names{i}));
    end
end

function manifest = read_or_build_manifest(input_path)
    input_path = char(input_path);
    if isfolder(input_path)
        files = [dir(fullfile(input_path, '**', '*.set')); dir(fullfile(input_path, '**', '*.mat'))];
        if isempty(files)
            error('No .set or .mat files found under %s.', input_path);
        end
        file_path = strings(numel(files), 1);
        participant = strings(numel(files), 1);
        condition = strings(numel(files), 1);
        group = repmat("all", numel(files), 1);
        for i = 1:numel(files)
            file_path(i) = string(fullfile(files(i).folder, files(i).name));
            participant(i) = infer_participant(file_path(i), i);
            condition(i) = infer_condition(file_path(i));
        end
    elseif isfile(input_path)
        opts = detectImportOptions(input_path, 'FileType', 'text', 'TextType', 'string');
        opts.VariableNamingRule = 'preserve';
        T = readtable(input_path, opts);
        names = normalise_names(T.Properties.VariableNames);
        file_col = find_first(names, ["filepath", "file_path", "file", "path", "filename"]);
        if isempty(file_col)
            error('Manifest must contain a file_path column.');
        end
        file_path = string(T{:, file_col});
        base_dir = fileparts(input_path);
        for i = 1:numel(file_path)
            if ~is_absolute_path(file_path(i))
                file_path(i) = string(fullfile(base_dir, file_path(i)));
            end
        end
        p_col = find_first(names, ["participant", "subject", "sub", "id", "subjectid", "participantid"]);
        c_col = find_first(names, ["condition", "state", "task", "eyes", "session"]);
        g_col = find_first(names, ["group", "diagnosis", "cohort", "class"]);
        participant = column_or_infer(T, p_col, file_path, "participant");
        condition = column_or_default(T, c_col, "condition", numel(file_path));
        group = column_or_default(T, g_col, "all", numel(file_path));
    else
        error('Input path not found: %s', input_path);
    end

    [file_path, ord] = sort(file_path);
    participant = participant(ord);
    condition = condition(ord);
    group = group(ord);
    manifest = table(participant(:), condition(:), group(:), file_path(:), ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path'});
end

function participant = column_or_infer(T, col, file_path, fallback)
    if isempty(col)
        participant = strings(numel(file_path), 1);
        for i = 1:numel(file_path)
            participant(i) = infer_participant(file_path(i), i);
        end
        return;
    end
    participant = string(T{:, col});
    bad = ismissing(participant) | strlength(strtrim(participant)) == 0;
    for i = find(bad(:))'
        participant(i) = infer_participant(file_path(i), i);
    end
    participant(strlength(strtrim(participant)) == 0) = fallback;
end

function values = column_or_default(T, col, default_value, n)
    if isempty(col)
        values = repmat(string(default_value), n, 1);
    else
        values = string(T{:, col});
        values(ismissing(values) | strlength(strtrim(values)) == 0) = string(default_value);
    end
end

function names = normalise_names(names_in)
    names = lower(regexprep(string(names_in), '[^a-zA-Z0-9]', ''));
end

function idx = find_first(names, choices)
    idx = [];
    choices = lower(regexprep(string(choices), '[^a-zA-Z0-9]', ''));
    for i = 1:numel(choices)
        hit = find(names == choices(i), 1, 'first');
        if ~isempty(hit)
            idx = hit;
            return;
        end
    end
end

function tf = is_absolute_path(pth)
    pth = char(pth);
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function pid = infer_participant(file_path, row_index)
    [parent, stem] = fileparts(char(file_path));
    hit = regexp(stem, '(sub-[A-Za-z0-9]+)', 'match', 'once');
    if isempty(hit)
        [~, hit] = fileparts(parent);
    end
    if isempty(hit)
        hit = sprintf('row%03d', row_index);
    end
    pid = string(hit);
end

function condition = infer_condition(file_path)
    [~, stem] = fileparts(char(file_path));
    s = upper(stem);
    if contains(s, '_EC') || contains(s, 'EYESCLOSED') || contains(s, 'CLOSED')
        condition = "EC";
    elseif contains(s, '_EO') || contains(s, 'EYESOPEN') || contains(s, 'OPEN')
        condition = "EO";
    else
        condition = "condition";
    end
end

function meta = inspect_files(manifest, cfg, util)
    meta = repmat(struct('file_path', "", 'labels', {{}}, 'canonical', {{}}, ...
        'keep_index', [], 'chanlocs', [], 'pos', []), height(manifest), 1);
    for i = 1:height(manifest)
        [data, ~, chanlocs, labels, pos] = load_eeg_file(manifest.file_path(i), util);
        n = size(data, 1);
        keep = true(n, 1);
        if cfg.use_scalp_channels
            keep = util.scalp_channel_mask(chanlocs, n);
        end
        can = util.canonical_channel_labels(labels);
        exclude_can = util.canonical_channel_labels(cfg.exclude_channels);
        keep = keep & ~ismember(can(:), exclude_can(:));
        if nnz(keep) < 8
            error('Too few channels remain for %s.', manifest.file_path(i));
        end
        meta(i).file_path = manifest.file_path(i);
        meta(i).labels = labels(keep);
        meta(i).canonical = can(keep);
        meta(i).keep_index = find(keep);
        if ~isempty(chanlocs)
            meta(i).chanlocs = chanlocs(keep);
        end
        if ~isempty(pos)
            meta(i).pos = pos(keep, :);
        end
    end
end

function [common_labels, common_can, common_chanlocs, common_pos] = common_channel_set(meta, cfg)
    if cfg.interpolate_missing_channels
        [~, ref] = max(arrayfun(@(m) numel(m.canonical), meta));
        common_can = meta(ref).canonical(:);
    else
        common_can = meta(1).canonical(:);
        for i = 2:numel(meta)
            common_can = common_can(ismember(common_can, meta(i).canonical));
        end
        if isempty(common_can)
            error('No common channels across files. Rerun with ''interpolate_missing_channels'', true to use the densest montage and fill missing channels.');
        end
        ref = 1;
    end
    common_labels = cell(numel(common_can), 1);
    common_chanlocs = [];
    common_pos = [];
    idx_first = zeros(numel(common_can), 1);
    for c = 1:numel(common_can)
        idx_first(c) = find(strcmp(meta(ref).canonical, common_can{c}), 1, 'first');
        common_labels{c} = meta(ref).labels{idx_first(c)};
    end
    if ~isempty(meta(ref).chanlocs)
        common_chanlocs = meta(ref).chanlocs(idx_first);
    end
    if ~isempty(meta(ref).pos)
        common_pos = meta(ref).pos(idx_first, :);
    end
end

function rec = extract_file_peaks(file_path, meta, common_can, common_pos, cfg, util)
    [data, sfreq, chanlocs, labels, pos] = load_eeg_file(file_path, util);
    can = util.canonical_channel_labels(labels);
    target_data = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            target_data(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if ~cfg.interpolate_missing_channels
            error('Missing channel %s in %s.', common_can{missing(1)}, file_path);
        end
        target_data = fill_missing_channels(target_data, missing, pos, idx, common_pos, file_path);
    end
    data = target_data;
    good_t = all(isfinite(data), 1);
    data = data(:, good_t);
    Sim = struct();
    Sim.X_noisy = data;
    Sim.X_clean = data;
    Sim.sfreq = sfreq;
    Sim.chanlocs = target_chanlocs(chanlocs, idx, common_can);
    Sim.channel_labels = common_can(:)';
    Sim.pos = common_pos;
    Sim.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule, ...
        'reject_gfp_peak_outliers', cfg.reject_gfp_peak_outliers, ...
        'gfp_outlier_mad_multiplier', cfg.gfp_outlier_mad_multiplier, ...
        'min_peak_count_after_gfp_rejection', cfg.min_gfp_peaks_per_file);
    [maps_norm, peak_sample, gfp_vec, ~, ~, maps_raw, preproc_info] = util.preprocess_maps(Sim);
    gfp_peak = gfp_vec(peak_sample(:));
    keep = deterministic_subsample(size(maps_norm, 1), cfg.max_maps_per_file);

    rec = struct();
    rec.file_path = char(file_path);
    rec.common_labels = Sim.channel_labels;
    rec.sfreq = sfreq;
    rec.maps_norm = single(maps_norm(keep, :));
    rec.maps_raw = single(maps_raw(keep, :));
    rec.peak_sample = peak_sample(keep);
    rec.gfp_peak = gfp_peak(keep);
    rec.n_peaks_raw = numel(peak_sample);
    rec.n_peaks_used = numel(keep);
    rec.preprocessing_info = preproc_info;
    rec.n_interpolated_channels = numel(missing);
    rec.interpolated_channel_labels = common_can(missing);
    rec.excluded_channel_labels = setdiff(meta.labels, Sim.channel_labels, 'stable');
end

function chanlocs_out = target_chanlocs(chanlocs, idx, common_can)
    if isempty(chanlocs) || any(~isfinite(idx))
        chanlocs_out = [];
    else
        chanlocs_out = chanlocs(idx);
    end
    if isempty(chanlocs_out)
        chanlocs_out = repmat(struct('labels', ''), 1, numel(common_can));
        for i = 1:numel(common_can)
            chanlocs_out(i).labels = common_can{i};
        end
    end
end

function data = fill_missing_channels(data, missing, source_pos, direct_idx, target_pos, file_path)
    if isempty(source_pos) || isempty(target_pos)
        error('Cannot interpolate missing channels in %s without channel positions.', file_path);
    end
    available = find(isfinite(direct_idx));
    if numel(available) < 4
        error('Cannot interpolate missing channels in %s with fewer than 4 observed target channels.', file_path);
    end
    source_xyz = source_pos(direct_idx(available), :);
    if size(source_xyz, 1) ~= numel(available) || size(target_pos, 1) < max(missing)
        error('Cannot interpolate missing channels in %s because channel positions are incomplete.', file_path);
    end
    for i = 1:numel(missing)
        t = missing(i);
        d = sqrt(sum((source_xyz - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        m = min(6, numel(ord));
        ord = ord(1:m);
        ds = ds(1:m);
        if ds(1) <= eps
            w = zeros(m, 1);
            w(1) = 1;
        else
            w = 1 ./ (ds.^2 + eps);
            w = w ./ sum(w);
        end
        data(t, :) = w' * data(available(ord), :);
    end
end

function [data, sfreq, chanlocs, labels, pos] = load_eeg_file(file_path, util)
    file_path = char(file_path);
    if ~isfile(file_path)
        error('EEG file not found: %s', file_path);
    end
    [~, ~, ext] = fileparts(file_path);
    ext = lower(ext);
    chanlocs = [];
    sfreq = 250;
    switch ext
        case '.set'
            if exist('pop_loadset', 'file') ~= 2
                error('EEGLAB pop_loadset not found on MATLAB path.');
            end
            EEG = pop_loadset(file_path);
            data = double(EEG.data);
            sfreq = double(EEG.srate);
            chanlocs = EEG.chanlocs;
            labels = util.channel_labels_from_chanlocs(chanlocs, size(data, 1));
            pos = util.positions_from_chanlocs(chanlocs, size(data, 1));
        case '.mat'
            S = load(file_path);
            if isfield(S, 'EEG')
                data = double(S.EEG.data);
                if isfield(S.EEG, 'srate'), sfreq = double(S.EEG.srate); end
                if isfield(S.EEG, 'chanlocs'), chanlocs = S.EEG.chanlocs; end
            elseif isfield(S, 'eeg_data')
                data = double(S.eeg_data);
            elseif isfield(S, 'data')
                data = double(S.data);
            else
                error('No EEG data field found in %s.', file_path);
            end
            if isfield(S, 'sfreq'), sfreq = double(S.sfreq); end
            if isfield(S, 'srate'), sfreq = double(S.srate); end
            if isfield(S, 'chanlocs'), chanlocs = S.chanlocs; end
            labels = util.channel_labels_from_chanlocs(chanlocs, size(data, 1));
            pos = util.positions_from_chanlocs(chanlocs, size(data, 1));
        otherwise
            error('Unsupported EEG extension: %s', ext);
    end
    data = squeeze(data);
    if size(data, 1) > size(data, 2) && size(data, 2) < 128
        data = data';
    end
end

function [nodes, summary] = fit_group_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, cfg, out_root, common_labels, common_chanlocs)
    levels = meaningful_levels(pooled_rows.group, "all");
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        level = levels(i);
        mask = pooled_rows.group == level;
        out_dir = fullfile(out_root, clean_key(level));
        ensure_dir(out_dir);
        Fit = fit_spm_vb_peak_maps(pooled_maps(mask, :), pooled_gfp(mask), cfg.K_candidates, cfg, ['group_' char(level)], global_fit.centers);
        Fit = attach_template_alignment_if_available(Fit, cfg, common_labels);
        rows = pooled_rows(mask, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(mask, :), Fit.centers);
        save_fit_bundle(out_dir, 'group', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('group', "", level, "", level, Fit, sum(mask), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('group', level, level, "", "", Fit, numel(unique(rows.file_index)), sum(mask), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Group ' char(level)], Fit);
    end
end

function [nodes, summary] = fit_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, cfg, out_root, common_labels, common_chanlocs)
    levels = meaningful_levels(pooled_rows.condition, "condition");
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        level = levels(i);
        mask = pooled_rows.condition == level;
        out_dir = fullfile(out_root, clean_key(level));
        ensure_dir(out_dir);
        Fit = fit_spm_vb_peak_maps(pooled_maps(mask, :), pooled_gfp(mask), cfg.K_candidates, cfg, ['condition_' char(level)], global_fit.centers);
        Fit = attach_template_alignment_if_available(Fit, cfg, common_labels);
        rows = pooled_rows(mask, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(mask, :), Fit.centers);
        save_fit_bundle(out_dir, 'condition', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('condition', "", "", level, level, Fit, sum(mask), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('condition', level, "", level, "", Fit, numel(unique(rows.file_index)), sum(mask), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Condition ' char(level)], Fit);
    end
end

function [nodes, summary] = fit_participant_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, cfg, out_root, common_labels, common_chanlocs)
    levels = unique(pooled_rows.participant, 'stable');
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        participant = levels(i);
        mask = pooled_rows.participant == participant;
        row_mask = manifest.participant == participant;
        group_value = manifest.group(find(row_mask, 1, 'first'));
        cond_values = unique(pooled_rows.condition(mask), 'stable');
        prior_maps = parent_prior_maps(global_fit, group_nodes, condition_nodes, group_value, cond_values);
        out_dir = fullfile(out_root, clean_key(participant));
        ensure_dir(out_dir);
        Fit = fit_spm_vb_peak_maps(pooled_maps(mask, :), pooled_gfp(mask), cfg.K_candidates, cfg, ['participant_' char(participant)], prior_maps);
        Fit = attach_template_alignment_if_available(Fit, cfg, common_labels);
        rows = pooled_rows(mask, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(mask, :), Fit.centers);
        save_fit_bundle(out_dir, 'participant', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('participant', participant, group_value, "", participant, Fit, sum(mask), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('participant', participant, group_value, "", participant, Fit, numel(unique(rows.file_index)), sum(mask), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Participant ' char(participant)], Fit);
    end
end

function [nodes, summary] = fit_participant_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, participant_nodes, cfg, out_root, common_labels, common_chanlocs)
    key = string(pooled_rows.participant) + "|" + string(pooled_rows.condition);
    levels = unique(key, 'stable');
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        parts = split(levels(i), "|");
        participant = parts(1);
        condition = parts(2);
        mask = pooled_rows.participant == participant & pooled_rows.condition == condition;
        if ~any(mask)
            continue;
        end

        row_mask = manifest.participant == participant & manifest.condition == condition;
        if ~any(row_mask)
            row_mask = manifest.participant == participant;
        end
        group_value = "all";
        if any(row_mask)
            group_value = manifest.group(find(row_mask, 1, 'first'));
        end

        prior_maps = global_fit.centers;
        c = find_node(condition_nodes, "", "", condition);
        if ~isempty(c)
            prior_maps = c.fit.centers;
        end
        g = find_node(group_nodes, "", group_value, "");
        if ~isempty(g)
            prior_maps = [prior_maps; g.fit.centers]; %#ok<AGROW>
        end
        p = find_node(participant_nodes, participant, "", "");
        if ~isempty(p)
            prior_maps = [prior_maps; p.fit.centers]; %#ok<AGROW>
        end

        out_dir = fullfile(out_root, clean_key(participant), clean_key(condition));
        ensure_dir(out_dir);
        fit_name = ['participant_condition_' char(participant) '_' char(condition)];
        Fit = fit_spm_vb_peak_maps(pooled_maps(mask, :), pooled_gfp(mask), cfg.K_candidates, cfg, fit_name, prior_maps);
        Fit = attach_template_alignment_if_available(Fit, cfg, common_labels);
        rows = pooled_rows(mask, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(mask, :), Fit.centers);
        save_fit_bundle(out_dir, 'participant_condition', Fit, rows, common_labels, common_chanlocs, cfg);

        node_name = participant + "_" + condition;
        nodes(end+1) = build_node('participant_condition', participant, group_value, condition, node_name, Fit, sum(mask), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('participant_condition', node_name, group_value, condition, participant, Fit, numel(unique(rows.file_index)), sum(mask), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Participant-condition ' char(participant) ' ' char(condition)], Fit);
    end
end

function prior_maps = parent_prior_maps(global_fit, group_nodes, condition_nodes, group_value, cond_values)
    prior_maps = global_fit.centers;
    g = find_node(group_nodes, "", group_value, "");
    if ~isempty(g)
        prior_maps = g.fit.centers;
    end
    for i = 1:numel(cond_values)
        c = find_node(condition_nodes, "", "", cond_values(i));
        if ~isempty(c)
            prior_maps = [prior_maps; c.fit.centers]; %#ok<AGROW>
        end
    end
end

function rec = find_node(nodes, participant, group, condition)
    rec = [];
    for i = 1:numel(nodes)
        if (strlength(participant) == 0 || nodes(i).participant == participant) && ...
                (strlength(group) == 0 || nodes(i).group == group) && ...
                (strlength(condition) == 0 || nodes(i).condition == condition)
            rec = nodes(i);
            return;
        end
    end
end

function out = output_csv_path(output_dir, csv_name)
    out = char(string(csv_name));
    if isempty(out)
        out = output_dir;
    elseif ~is_absolute_path(out)
        out = fullfile(output_dir, out);
    end
end

function [T_state, T_pairwise, T_record, T_axis] = compile_backfit_metric_tables(manifest, participant_nodes, participant_condition_nodes, common_can, common_pos, cfg, util)
    state_rows = {};
    pair_rows = {};
    record_rows = {};
    axis_rows = {};
    for i = 1:height(manifest)
        participant = manifest.participant(i);
        condition = manifest.condition(i);
        node = find_node(participant_condition_nodes, participant, "", condition);
        if isempty(node)
            node = find_node(participant_nodes, participant, "", "");
        end
        if isempty(node)
            warning('No participant-condition or participant fit found for %s %s; backfit skipped.', participant, condition);
            continue;
        end
        try
            [Sim_backfit, gfp, X_metric] = prepare_backfit_record(manifest.file_path(i), common_can, common_pos, cfg, util);
            Results = node.fit;
            Results.name = sprintf('backfit_%s_%s', char(manifest.participant(i)), char(manifest.condition(i)));
            backfit = backfit_microstate_timecourse(Sim_backfit, Results);
            if ~isstruct(backfit) || ~isfield(backfit, 'ok') || ~backfit.ok
                warning('Backfit failed for %s: %s', manifest.file_path(i), backfit.message);
                continue;
            end
            [Ts, Tp, Tr] = summarise_backfit_record(backfit, Results, gfp, Sim_backfit.sfreq, ...
                manifest.participant(i), manifest.condition(i), manifest.group(i), manifest.file_path(i), node.level);
            state_rows{end+1, 1} = Ts; %#ok<AGROW>
            pair_rows{end+1, 1} = Tp; %#ok<AGROW>
            record_rows{end+1, 1} = Tr; %#ok<AGROW>
            if cfg.run_backfit_axis_dynamics
                Ta = summarise_backfit_axis_dynamics(backfit, Results, X_metric, Sim_backfit.sfreq, ...
                    manifest.participant(i), manifest.condition(i), manifest.group(i), manifest.file_path(i), ...
                    cfg.backfit_axis_min_samples, node.level);
                axis_rows{end+1, 1} = Ta; %#ok<AGROW>
            end
        catch ME
            warning('Backfit metric extraction failed for %s: %s', manifest.file_path(i), ME.message);
        end
    end

    T_state = vertcat_nonempty(state_rows);
    T_pairwise = vertcat_nonempty(pair_rows);
    T_record = vertcat_nonempty(record_rows);
    T_axis = vertcat_nonempty(axis_rows);
    if ~isempty(T_state)
        T_state = sortrows(T_state, {'participant', 'condition', 'backfit_method', 'template_label', 'state_index'});
    end
    if ~isempty(T_pairwise)
        T_pairwise = sortrows(T_pairwise, {'participant', 'condition', 'backfit_method', 'state_i_label', 'state_j_label'});
    end
    if ~isempty(T_record)
        T_record = sortrows(T_record, {'participant', 'condition', 'backfit_method'});
    end
    if ~isempty(T_axis)
        T_axis = sortrows(T_axis, {'participant', 'condition', 'template_label', 'state_index'});
    end
end

function T = vertcat_nonempty(rows)
    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        T = table();
    else
        T = vertcat(rows{:});
    end
end

function [Sim_backfit, gfp, X_metric] = prepare_backfit_record(file_path, common_can, common_pos, cfg, util)
    [data, sfreq, ~, labels, pos] = load_eeg_file(file_path, util);
    can = util.canonical_channel_labels(labels);
    target_data = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            target_data(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if ~cfg.interpolate_missing_channels
            error('Missing channel %s in %s.', common_can{missing(1)}, file_path);
        end
        target_data = fill_missing_channels(target_data, missing, pos, idx, common_pos, file_path);
    end
    good_t = all(isfinite(target_data), 1);
    target_data = target_data(:, good_t);

    Sim_backfit = struct();
    Sim_backfit.X_noisy = target_data;
    Sim_backfit.sfreq = sfreq;
    Sim_backfit.channel_labels = common_can(:)';
    Sim_backfit.chanlocs = [];
    Sim_backfit.pos = common_pos;
    Sim_backfit.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule);
    X_metric = preprocess_backfit_metric_matrix(Sim_backfit, util);
    gfp = std(X_metric, 0, 1, 'omitnan')';
end

function X = preprocess_backfit_metric_matrix(Sim_backfit, util)
    X = double(Sim_backfit.X_noisy);
    if isfield(Sim_backfit.preprocessing, 'apply_average_reference') && Sim_backfit.preprocessing.apply_average_reference
        X = X - mean(X, 1, 'omitnan');
    end
    if isfield(Sim_backfit.preprocessing, 'filter_band') && ~isempty(Sim_backfit.preprocessing.filter_band)
        X = util.bandpass_filter(X, Sim_backfit.sfreq, Sim_backfit.preprocessing.filter_band);
    end
end

function [T_state, T_pairwise, T_record] = summarise_backfit_record(backfit, Results, gfp, sfreq, participant, condition, group, file_path, result_source)
    [state_labels, state_order, template_corr] = state_labels_for_metrics(Results);
    K = size(Results.centers, 1);
    n_samples = min(numel(gfp), backfit.n_samples);
    gfp = gfp(1:n_samples);

    mix_available = isfield(backfit, 'mixture') && isstruct(backfit.mixture) && ...
        isfield(backfit.mixture, 'available') && backfit.mixture.available;
    modes = { ...
        struct('name', 'hard', 'available', true, 'weights', backfit.hard.weights, 'assignments', backfit.hard.assignments), ...
        struct('name', 'gaussian_mixture', 'available', mix_available, 'weights', get_backfit_weights(backfit, 'mixture'), 'assignments', get_backfit_assignments(backfit, 'mixture'))};

    state_rows = {};
    pair_rows = {};
    record_rows = {};
    for m = 1:numel(modes)
        spec = modes{m};
        weights = double(spec.weights);
        assignments = double(spec.assignments(:));
        gfp_mode = gfp;
        if spec.available
            if size(weights, 1) > n_samples
                weights = weights(1:n_samples, :);
            elseif size(weights, 1) < n_samples
                gfp_mode = gfp(1:size(weights, 1));
            else
                gfp_mode = gfp;
            end
            if numel(assignments) < size(weights, 1)
                [~, assignments] = max(weights, [], 2);
            else
                assignments = assignments(1:size(weights, 1));
            end
        else
            weights = nan(n_samples, K);
            assignments = nan(n_samples, 1);
            gfp_mode = gfp;
        end
        n_mode = size(weights, 1);
        duration_mode_s = n_mode / max(sfreq, eps);

        Wn = normalize_weight_matrix_rows(weights);
        if spec.available && strcmpi(spec.name, 'hard')
            record_differential_entropy_bits = NaN;
            record_shannon_entropy_bits = shannon_entropy_bits_from_state_distribution(mean(Wn, 1, 'omitnan'));
        elseif spec.available
            record_differential_entropy_bits = joint_differential_entropy_bits_from_weight_matrix(Wn);
            record_shannon_entropy_bits = NaN;
        else
            record_differential_entropy_bits = NaN;
            record_shannon_entropy_bits = NaN;
        end

        record_rows{end+1, 1} = table( ...
            string(participant), string(condition), string(group), string(file_path), ...
            string(result_source), string(spec.name), logical(spec.available), ...
            double(K), double(n_mode), double(sfreq), double(duration_mode_s), ...
            double(record_differential_entropy_bits), double(record_shannon_entropy_bits), ...
            'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
            'backfit_result_source', 'backfit_method', 'backfit_available', ...
            'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
            'record_differential_entropy_bits', 'record_shannon_entropy_bits'}); %#ok<AGROW>

        for j = 1:numel(state_order)
            k = state_order(j);
            wk = weights(:, k);
            wk(~isfinite(wk)) = 0;
            if spec.available && sum(wk, 'omitnan') > eps
                occupancy = mean(wk, 'omitnan');
                pct_present = 100 * mean(assignments == k, 'omitnan');
                mean_gfp = sum(wk .* gfp_mode, 'omitnan') / sum(wk, 'omitnan');
                occurrence_count = count_state_occurrences(assignments, k);
                occurrence_rate_hz = occurrence_count / max(duration_mode_s, eps);
            else
                occupancy = NaN;
                pct_present = NaN;
                mean_gfp = NaN;
                occurrence_count = NaN;
                occurrence_rate_hz = NaN;
            end
            state_rows{end+1, 1} = table( ...
                string(participant), string(condition), string(group), string(file_path), ...
                string(result_source), string(spec.name), logical(spec.available), ...
                double(k), string(state_labels{k}), double(K), double(n_mode), double(sfreq), double(duration_mode_s), ...
                double(occupancy), double(pct_present), double(occupancy), double(mean_gfp), ...
                double(occurrence_count), double(occurrence_rate_hz), double(template_corr(k)), ...
                'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
                'backfit_result_source', 'backfit_method', 'backfit_available', ...
                'state_index', 'template_label', 'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
                'occupancy', 'percentage_record_present', 'mean_quantity', 'mean_gfp', ...
                'occurrence_count', 'occurrence_rate_hz', 'template_match_abs_correlation'}); %#ok<AGROW>
        end

        if spec.available
            for a = 1:(numel(state_order) - 1)
                k1 = state_order(a);
                for b = (a + 1):numel(state_order)
                    k2 = state_order(b);
                    if strcmpi(spec.name, 'hard')
                        [mi_bits, nmi_bits] = binary_mutual_information_bits(weights(:, k1) > 0.5, weights(:, k2) > 0.5);
                    else
                        [mi_bits, nmi_bits] = normalized_quantity_mutual_information_bits(weights(:, k1), weights(:, k2));
                    end
                    pair_rows{end+1, 1} = table( ...
                        string(participant), string(condition), string(group), string(file_path), ...
                        string(result_source), string(spec.name), logical(spec.available), ...
                        double(K), double(n_mode), double(sfreq), double(duration_mode_s), ...
                        double(k1), string(state_labels{k1}), double(k2), string(state_labels{k2}), ...
                        double(mi_bits), double(nmi_bits), ...
                        'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
                        'backfit_result_source', 'backfit_method', 'backfit_available', ...
                        'K_estimated', 'n_samples', 'sfreq', 'duration_s', ...
                        'state_i_index', 'state_i_label', 'state_j_index', 'state_j_label', ...
                        'mutual_information_bits', 'normalized_mutual_information'}); %#ok<AGROW>
                end
            end
        end
    end

    T_state = vertcat_nonempty(state_rows);
    T_pairwise = vertcat_nonempty(pair_rows);
    T_record = vertcat_nonempty(record_rows);
end

function T_axis = summarise_backfit_axis_dynamics(backfit, Results, X_metric, sfreq, participant, condition, group, file_path, min_samples, result_source)
    [state_labels, state_order, template_corr] = state_labels_for_metrics(Results);
    K = size(Results.centers, 1);
    rows = {};

    if ~isfield(backfit, 'hard') || ~isfield(backfit.hard, 'assignments') || isempty(backfit.hard.assignments)
        T_axis = table();
        return;
    end

    assignments = double(backfit.hard.assignments(:));
    n_samples = min([size(X_metric, 2), numel(assignments), backfit.n_samples]);
    if n_samples < 2
        T_axis = table();
        return;
    end

    X_metric = double(X_metric(:, 1:n_samples));
    assignments = assignments(1:n_samples);
    maps_norm = normalize_maps(X_metric');
    centers = normalize_maps(Results.centers);
    min_samples = max(2, round(double(min_samples)));

    for j = 1:numel(state_order)
        k = state_order(j);
        active = assignments == k & isfinite(assignments);
        n_active = sum(active);
        active_fraction = n_active / n_samples;
        active_duration_s = n_active / max(sfreq, eps);
        run_lengths = active_run_lengths(active);
        active_run_count = numel(run_lengths);
        mean_run_s = mean_or_nan(run_lengths) / max(sfreq, eps);
        median_run_s = median_or_nan(run_lengths) / max(sfreq, eps);

        axis_projection_rms = NaN;
        axis_projection_std = NaN;
        axis_projection_mean_abs = NaN;
        axis_zero_crossing_rate_hz = NaN;
        axis_energy_fraction = NaN;
        pca_pc1_variance_pct = NaN;
        pca_pc2_variance_pct = NaN;
        pca_pc1_abs_corr_with_state_axis = NaN;
        angular_deviation_mean_deg = NaN;
        angular_deviation_sd_deg = NaN;
        angular_deviation_median_deg = NaN;
        perpendicular_energy_fraction = NaN;
        residual_pc1_variance_pct = NaN;
        residual_pc2_variance_pct = NaN;
        residual_phase_resultant_length = NaN;
        precession_signed_mean_deg_per_s = NaN;
        precession_mean_abs_deg_per_s = NaN;
        precession_median_abs_deg_per_s = NaN;
        precession_abs_cycles_per_s = NaN;
        precession_net_cycles_per_s = NaN;
        precession_directionality_index = NaN;
        precession_positive_fraction = NaN;
        precession_negative_fraction = NaN;

        if n_active >= min_samples && k <= size(centers, 1) && size(maps_norm, 2) == size(centers, 2)
            axis_norm = centers(k, :);
            axis_col = axis_norm(:);
            axis_signal_full = X_metric' * axis_col;
            axis_signal = axis_signal_full(active);
            axis_projection_rms = sqrt(mean(axis_signal.^2, 'omitnan'));
            axis_projection_std = std(axis_signal, 0, 'omitnan');
            axis_projection_mean_abs = mean(abs(axis_signal), 'omitnan');
            axis_zero_crossing_rate_hz = zero_crossing_rate_active(axis_signal_full, active, sfreq);

            maps_k = maps_norm(active, :);
            proj = maps_k * axis_col;
            proj_abs = abs(proj);
            axis_energy_fraction = mean(clamp_unit(proj).^2, 'omitnan');
            angles = acosd(clamp_unit(proj_abs));
            angular_deviation_mean_deg = mean(angles, 'omitnan');
            angular_deviation_sd_deg = std(angles, 0, 'omitnan');
            angular_deviation_median_deg = median(angles, 'omitnan');
            perpendicular_energy_fraction = mean(max(0, 1 - proj_abs.^2), 'omitnan');

            Xc = maps_k - mean(maps_k, 1, 'omitnan');
            [pca_pc1_variance_pct, pca_pc2_variance_pct, pca_pc1_abs_corr_with_state_axis] = ...
                pca_axis_summary(Xc, axis_col);

            signs = sign(proj);
            signs(signs == 0) = 1;
            maps_aligned = maps_k .* signs;
            residual_active = maps_aligned - proj_abs * axis_norm;
            Rc = residual_active - mean(residual_active, 1, 'omitnan');
            [residual_pc1_variance_pct, residual_pc2_variance_pct, ~, residual_basis] = ...
                pca_axis_summary(Rc, axis_col);

            if size(residual_basis, 2) >= 2
                proj_full = maps_norm * axis_col;
                signs_full = sign(proj_full);
                signs_full(signs_full == 0) = 1;
                maps_aligned_full = maps_norm .* signs_full;
                residual_full = maps_aligned_full - abs(proj_full) * axis_norm;
                phase = atan2(residual_full * residual_basis(:, 2), residual_full * residual_basis(:, 1));
                residual_phase_resultant_length = abs(mean(exp(1i * phase(active)), 'omitnan'));
                [precession_signed_mean_deg_per_s, precession_mean_abs_deg_per_s, ...
                    precession_median_abs_deg_per_s, precession_abs_cycles_per_s, ...
                    precession_net_cycles_per_s, precession_directionality_index, ...
                    precession_positive_fraction, precession_negative_fraction] = ...
                    precession_metrics_active(phase, active, sfreq);
            end
        end

        rows{end+1, 1} = table( ...
            string(participant), string(condition), string(group), string(file_path), ...
            string(result_source), "hard", true, double(k), string(state_labels{k}), ...
            double(K), double(n_samples), double(sfreq), double(n_active), double(active_fraction), ...
            double(active_duration_s), double(active_run_count), double(mean_run_s), double(median_run_s), ...
            double(axis_projection_rms), double(axis_projection_std), double(axis_projection_mean_abs), ...
            double(axis_zero_crossing_rate_hz), double(axis_energy_fraction), ...
            double(pca_pc1_variance_pct), double(pca_pc2_variance_pct), double(pca_pc1_abs_corr_with_state_axis), ...
            double(angular_deviation_mean_deg), double(angular_deviation_sd_deg), double(angular_deviation_median_deg), ...
            double(perpendicular_energy_fraction), double(residual_pc1_variance_pct), double(residual_pc2_variance_pct), ...
            double(residual_phase_resultant_length), double(precession_signed_mean_deg_per_s), ...
            double(precession_mean_abs_deg_per_s), double(precession_median_abs_deg_per_s), ...
            double(precession_abs_cycles_per_s), double(precession_net_cycles_per_s), ...
            double(precession_directionality_index), double(precession_positive_fraction), ...
            double(precession_negative_fraction), double(template_corr(k)), ...
            'VariableNames', {'participant', 'condition', 'group', 'file_path', ...
            'backfit_result_source', 'backfit_method', 'backfit_available', ...
            'state_index', 'template_label', 'K_estimated', 'n_samples', 'sfreq', ...
            'n_active_samples', 'active_fraction', 'active_duration_s', 'active_run_count', ...
            'mean_active_run_duration_s', 'median_active_run_duration_s', ...
            'axis_projection_rms', 'axis_projection_std', 'axis_projection_mean_abs', ...
            'axis_zero_crossing_rate_hz', 'axis_energy_fraction', ...
            'pca_pc1_variance_pct', 'pca_pc2_variance_pct', 'pca_pc1_abs_corr_with_state_axis', ...
            'angular_deviation_mean_deg', 'angular_deviation_sd_deg', 'angular_deviation_median_deg', ...
            'perpendicular_energy_fraction', 'residual_pc1_variance_pct', 'residual_pc2_variance_pct', ...
            'residual_phase_resultant_length', 'precession_signed_mean_deg_per_s', ...
            'precession_mean_abs_deg_per_s', 'precession_median_abs_deg_per_s', ...
            'precession_abs_cycles_per_s', 'precession_net_cycles_per_s', ...
            'precession_directionality_index', 'precession_positive_fraction', ...
            'precession_negative_fraction', ...
            'template_match_abs_correlation'}); %#ok<AGROW>
    end

    T_axis = vertcat_nonempty(rows);
end

function lengths = active_run_lengths(mask)
    mask = logical(mask(:));
    d = diff([false; mask; false]);
    starts = find(d == 1);
    stops = find(d == -1) - 1;
    lengths = stops - starts + 1;
end

function x = mean_or_nan(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v)
        x = NaN;
    else
        x = mean(v);
    end
end

function x = median_or_nan(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v)
        x = NaN;
    else
        x = median(v);
    end
end

function x = clamp_unit(x)
    x = max(-1, min(1, double(x)));
end

function [pc1_pct, pc2_pct, pc1_axis_corr, basis] = pca_axis_summary(Xc, axis_col)
    pc1_pct = NaN;
    pc2_pct = NaN;
    pc1_axis_corr = NaN;
    basis = zeros(size(Xc, 2), 0);
    if size(Xc, 1) < 2 || isempty(Xc)
        return;
    end
    Xc(~isfinite(Xc)) = 0;
    [~, S, V] = svd(Xc, 'econ');
    latent = diag(S).^2 ./ max(1, size(Xc, 1) - 1);
    total = sum(latent);
    if total <= eps || isempty(V)
        return;
    end
    basis = V;
    pc1_pct = 100 * latent(1) / total;
    if numel(latent) >= 2
        pc2_pct = 100 * latent(2) / total;
    end
    pc1_axis_corr = abs(dot(V(:, 1), axis_col));
end

function rate_hz = zero_crossing_rate_active(signal, active, sfreq)
    idx = find(active(:));
    if numel(idx) < 2
        rate_hz = NaN;
        return;
    end
    a = double(signal(idx(1:end-1)));
    b = double(signal(idx(2:end)));
    adjacent = idx(2:end) == idx(1:end-1) + 1;
    valid = adjacent & isfinite(a) & isfinite(b) & a ~= 0 & b ~= 0;
    flips = (a < 0 & b > 0) | (a > 0 & b < 0);
    active_duration_s = numel(idx) / max(sfreq, eps);
    rate_hz = sum(flips(:) & valid(:)) / max(active_duration_s, eps);
end

function [signed_mean_deg_s, mean_abs_deg_s, median_abs_deg_s, abs_cycles_s, net_cycles_s, directionality, positive_fraction, negative_fraction] = precession_metrics_active(phase, active, sfreq)
    idx = find(active(:));
    if numel(idx) < 2
        signed_mean_deg_s = NaN;
        mean_abs_deg_s = NaN;
        median_abs_deg_s = NaN;
        abs_cycles_s = NaN;
        net_cycles_s = NaN;
        directionality = NaN;
        positive_fraction = NaN;
        negative_fraction = NaN;
        return;
    end
    p1 = phase(idx(1:end-1));
    p2 = phase(idx(2:end));
    adjacent = idx(2:end) == idx(1:end-1) + 1;
    dphi = angle(exp(1i * (p2 - p1)));
    dphi = dphi(adjacent & isfinite(dphi));
    if isempty(dphi)
        signed_mean_deg_s = NaN;
        mean_abs_deg_s = NaN;
        median_abs_deg_s = NaN;
        abs_cycles_s = NaN;
        net_cycles_s = NaN;
        directionality = NaN;
        positive_fraction = NaN;
        negative_fraction = NaN;
        return;
    end
    abs_dphi = abs(dphi);
    duration_s = numel(idx) / max(sfreq, eps);
    signed_mean_deg_s = mean(dphi) * sfreq * 180 / pi;
    mean_abs_deg_s = mean(abs_dphi) * sfreq * 180 / pi;
    median_abs_deg_s = median(abs_dphi) * sfreq * 180 / pi;
    abs_cycles_s = sum(abs_dphi) / (2 * pi) / max(duration_s, eps);
    net_cycles_s = sum(dphi) / (2 * pi) / max(duration_s, eps);
    directionality = abs(sum(dphi)) / max(sum(abs_dphi), eps);
    positive_fraction = mean(dphi > 0);
    negative_fraction = mean(dphi < 0);
end

function weights = get_backfit_weights(backfit, mode)
    weights = [];
    if isfield(backfit, mode) && isfield(backfit.(mode), 'weights')
        weights = backfit.(mode).weights;
    end
end

function assignments = get_backfit_assignments(backfit, mode)
    assignments = [];
    if isfield(backfit, mode) && isfield(backfit.(mode), 'assignments')
        assignments = backfit.(mode).assignments;
    end
end

function [state_labels, state_order, template_corr] = state_labels_for_metrics(Results)
    K = size(Results.centers, 1);
    state_labels = arrayfun(@(k) sprintf('state_%02d', k), 1:K, 'UniformOutput', false);
    template_corr = nan(K, 1);
    if isfield(Results, 'template_alignment') && isstruct(Results.template_alignment)
        A = Results.template_alignment;
        if isfield(A, 'labels') && numel(A.labels) >= K
            state_labels = cellstr(string(A.labels(:)));
        end
        if isfield(A, 'correlations') && numel(A.correlations) >= K
            template_corr = abs(double(A.correlations(:)));
        end
    end
    [~, state_order] = sort(lower(string(state_labels)));
end

function n = count_state_occurrences(assignments, state_idx)
    assignments = double(assignments(:));
    assignments = assignments(isfinite(assignments));
    if isempty(assignments)
        n = NaN;
        return;
    end
    hit = assignments == state_idx;
    n = double(hit(1)) + sum(hit(2:end) & ~hit(1:end-1));
end

function W = normalize_weight_matrix_rows(W)
    W = double(W);
    W(~isfinite(W)) = 0;
    row_sum = sum(W, 2);
    valid = row_sum > eps;
    W(valid, :) = W(valid, :) ./ row_sum(valid);
    W(~valid, :) = 0;
end

function h_bits = joint_differential_entropy_bits_from_weight_matrix(W)
    W = normalize_weight_matrix_rows(W);
    if isempty(W) || size(W, 1) < 16
        h_bits = NaN;
        return;
    end
    d = size(W, 2);
    n_bins = min(5, max(3, round(size(W, 1) .^ (1 / max(d + 1, 2)))));
    edges = linspace(0, 1, n_bins + 1);
    bin_subs = zeros(size(W));
    for j = 1:d
        bin_subs(:, j) = discretize(min(max(W(:, j), 0), 1), edges);
    end
    valid = all(bin_subs >= 1 & bin_subs <= n_bins, 2);
    bin_subs = bin_subs(valid, :);
    if size(bin_subs, 1) < 16
        h_bits = NaN;
        return;
    end
    strides = [1, cumprod(repmat(n_bins, 1, d - 1))];
    linear_idx = 1 + sum((bin_subs - 1) .* strides, 2);
    counts = accumarray(linear_idx, 1, [n_bins ^ d, 1]);
    p = counts / sum(counts);
    p = p(p > 0);
    h_bits = -sum(p .* log2(p)) + d * log2(edges(2) - edges(1));
end

function h_bits = shannon_entropy_bits_from_state_distribution(p)
    p = double(p(:));
    p(~isfinite(p)) = 0;
    if sum(p) <= eps
        h_bits = NaN;
        return;
    end
    p = p / sum(p);
    p = p(p > 0);
    h_bits = -sum(p .* log2(p));
end

function [mi_bits, nmi] = binary_mutual_information_bits(x, y)
    x = logical(x(:));
    y = logical(y(:));
    n = min(numel(x), numel(y));
    if n == 0
        mi_bits = NaN;
        nmi = NaN;
        return;
    end
    x = x(1:n);
    y = y(1:n);
    joint = zeros(2, 2);
    for xi = 0:1
        for yi = 0:1
            joint(xi + 1, yi + 1) = mean((x == logical(xi)) & (y == logical(yi)));
        end
    end
    [mi_bits, nmi] = mutual_information_from_joint(joint);
end

function [mi_bits, nmi] = normalized_quantity_mutual_information_bits(x, y)
    x = normalize_single_quantity_trace(x);
    y = normalize_single_quantity_trace(y);
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(x) < 16 || max(x) - min(x) <= eps || max(y) - min(y) <= eps
        mi_bits = NaN;
        nmi = NaN;
        return;
    end
    n_bins = min(16, max(6, round(sqrt(numel(x)) / 2)));
    joint = histcounts2(min(max(x, 0), 1), min(max(y, 0), 1), linspace(0, 1, n_bins + 1), linspace(0, 1, n_bins + 1));
    [mi_bits, nmi] = mutual_information_from_joint(joint);
end

function [mi_bits, nmi] = mutual_information_from_joint(joint_counts)
    total = sum(joint_counts, 'all');
    if total <= 0
        mi_bits = NaN;
        nmi = NaN;
        return;
    end
    pxy = joint_counts / total;
    px = sum(pxy, 2);
    py = sum(pxy, 1);
    mi_bits = 0;
    for i = 1:size(pxy, 1)
        for j = 1:size(pxy, 2)
            if pxy(i, j) > 0
                mi_bits = mi_bits + pxy(i, j) * log2(pxy(i, j) / max(px(i) * py(j), eps));
            end
        end
    end
    nmi = mi_bits / max(sqrt(discrete_entropy_bits(px) * discrete_entropy_bits(py(:))), eps);
end

function h = discrete_entropy_bits(p)
    p = double(p(:));
    p = p(isfinite(p) & p > 0);
    if isempty(p)
        h = 0;
    else
        h = -sum(p .* log2(p));
    end
end

function x = normalize_single_quantity_trace(x)
    x = double(x(:));
    x(~isfinite(x)) = 0;
    x = x - min(x);
    xmax = max(x);
    if xmax > eps
        x = x ./ xmax;
    else
        x(:) = 0;
    end
end

function levels = meaningful_levels(values, default_value)
    levels = unique(string(values), 'stable');
    levels = levels(~ismissing(levels) & strlength(strtrim(levels)) > 0);
    levels = levels(levels ~= string(default_value));
    levels = levels(:)';
end

function Fit = fit_spm_vb_peak_maps(maps, gfp, K_candidates, cfg, fit_name, prior_maps)
    t0 = tic;
    X = normalize_maps(double(maps));
    gfp = double(gfp(:));
    if numel(gfp) ~= size(X, 1)
        gfp = ones(size(X, 1), 1);
    end
    ok = all(isfinite(X), 2) & isfinite(gfp) & gfp > 0;
    X = X(ok, :);
    gfp = gfp(ok);
    K_candidates = unique(round(double(K_candidates(:)')));
    K_candidates = K_candidates(K_candidates >= 2 & K_candidates < size(X, 1));
    if isempty(K_candidates)
        error('No valid K candidates for %s with %d maps.', fit_name, size(X, 1));
    end

    prior_maps = normalize_maps(double(prior_maps));
    nK = numel(K_candidates);
    free_energy = -inf(nK, 1);
    silhouette_vals = nan(nK, 1);
    gev_vals = nan(nK, 1);
    wss_vals = inf(nK, 1);
    centers_by_K = cell(nK, 1);
    labels_by_K = cell(nK, 1);
    summaries = repmat(empty_mix_summary(), nK, 1);

    for iK = 1:nK
        K = K_candidates(iK);
        try
            prior_K = ensure_k_prior_maps(prior_maps, X, K);
            X_aug = augment_with_priors(X, prior_K, cfg.spm_prior_pseudocount);
            [features, feature_info] = spm_features(X_aug);
            evalc('mix = spm_mix(features, K, 0);');
            if ~valid_mix(mix, K)
                continue;
            end
            labels_aug = assign_mix(features, mix);
            centers0 = recover_centers(X_aug, labels_aug, K, prior_K);
            centers = refine_centers(X, centers0, prior_K, 20);
            labels = assign_by_abs_correlation(X, centers);
            metrics = topo_metrics(X, labels, centers, gfp);

            free_energy(iK) = double(real(mix.fm));
            silhouette_vals(iK) = metrics.silhouette;
            gev_vals(iK) = metrics.gev;
            wss_vals(iK) = metrics.wss;
            centers_by_K{iK} = centers;
            labels_by_K{iK} = labels;
            summaries(iK) = summarise_mix(mix, K, size(features, 2), feature_info);
        catch ME
            warning('SPM-VB fit failed for %s K=%d: %s', fit_name, K, ME.message);
        end
    end

    valid = isfinite(free_energy) & ~cellfun(@isempty, centers_by_K);
    if ~any(valid)
        error('All SPM-VB fits failed for %s.', fit_name);
    end
    selection_payload = struct('K_candidates', K_candidates(:), ...
        'free_energy_vals', free_energy(:), ...
        'silhouette_vals', silhouette_vals(:), ...
        'gev_vals', gev_vals(:), ...
        'calinski_harabasz_vals', nan(nK, 1), ...
        'spm_mix_model_summaries', summaries);
    [K_est, best_score, score_by_k, selection_details] = select_spm_vb_k_by_criterion(selection_payload, cfg.criterion);
    if ~isfinite(K_est)
        [~, best_idx] = max(gev_vals);
        K_est = K_candidates(best_idx);
        best_score = gev_vals(best_idx);
        score_by_k = gev_vals;
        selection_details = struct('criterion', 'gev_fallback');
    else
        best_idx = find(K_candidates == K_est, 1, 'first');
    end

    model_comparison = table(K_candidates(:), free_energy(:), silhouette_vals(:), gev_vals(:), wss_vals(:), score_by_k(:), ...
        'VariableNames', {'K', 'free_energy', 'silhouette', 'gev', 'wss', 'selection_score'});

    Fit = struct();
    Fit.name = char(fit_name);
    Fit.method = 'spm_vb_hierarchical';
    Fit.criterion = cfg.criterion;
    Fit.K_candidates = K_candidates;
    Fit.K_estimated = K_est;
    Fit.best_index = best_idx;
    Fit.best_criterion_value = best_score;
    Fit.centers = centers_by_K{best_idx};
    Fit.labels = labels_by_K{best_idx};
    Fit.cluster_weights = cluster_weights(Fit.labels, K_est);
    Fit.maps_nc = single(X);
    Fit.backfit_peak_labels = double(Fit.labels(:));
    Fit.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule);
    Fit.free_energy_vals = free_energy;
    Fit.silhouette_vals = silhouette_vals;
    Fit.gev_vals = gev_vals;
    Fit.within_ss = wss_vals;
    Fit.selection_score_by_k = score_by_k;
    Fit.selection_details = selection_details;
    Fit.model_comparison = model_comparison;
    Fit.spm_mix_model_summaries = summaries;
    Fit.selected_spm_mix_model = summaries(best_idx);
    Fit.centers_by_K = centers_by_K;
    Fit.labels_by_K = labels_by_K;
    Fit.n_maps = size(X, 1);
    Fit.n_channels = size(X, 2);
    Fit.prior_n_maps = size(prior_maps, 1);
    Fit.spm_prior_pseudocount = cfg.spm_prior_pseudocount;
    Fit.runtime = toc(t0);
end

function X_aug = augment_with_priors(X, prior_maps, pseudocount)
    X_aug = X;
    r = max(0, round(double(pseudocount)));
    if isempty(prior_maps) || r == 0
        return;
    end
    X_aug = [X_aug; repmat(prior_maps, r, 1)];
end

function prior_K = ensure_k_prior_maps(prior_maps, X, K)
    if isempty(prior_maps)
        prior_K = [];
        return;
    end
    prior_K = normalize_maps(prior_maps);
    if size(prior_K, 1) > K
        prior_K = prior_K(1:K, :);
    end
    while size(prior_K, 1) < K
        if isempty(prior_K)
            prior_K = X(1, :);
        else
            sim = abs(X * prior_K');
            [~, idx] = max(1 - max(sim, [], 2));
            prior_K(end+1, :) = X(idx, :); %#ok<AGROW>
        end
    end
    prior_K = normalize_maps(prior_K);
end

function [features, info] = spm_features(X)
    X = normalize_maps(X);
    Xc = X - mean(X, 1);
    [~, S, V] = svd(Xc, 'econ');
    latent = diag(S).^2 ./ max(1, size(Xc, 1) - 1);
    total = sum(latent);
    if total <= eps
        linear = Xc(:, 1:min(size(Xc, 2), 2));
        rank_est = size(linear, 2);
        var_pct = 0;
    else
        rank_est = nnz(latent > max(size(Xc)) * eps(max(latent)));
        cumv = cumsum(latent) ./ total;
        nd = find(cumv >= 0.999, 1, 'first');
        if isempty(nd), nd = rank_est; end
        nd = max(1, min([nd, rank_est, size(Xc, 1) - 1, size(Xc, 2) - 1, 8]));
        linear = Xc * V(:, 1:nd);
        var_pct = 100 * cumv(nd);
    end
    sd = std(linear, 0, 1);
    sd(sd < eps) = 1;
    linear = linear ./ sd;

    Y = normalize_rows(linear);
    D = size(Y, 2);
    features = zeros(size(Y, 1), D * (D + 1) / 2);
    col = 0;
    for a = 1:D
        for b = a:D
            col = col + 1;
            if a == b
                scale = 1;
            else
                scale = sqrt(2);
            end
            features(:, col) = scale * Y(:, a) .* Y(:, b);
        end
    end
    features = features - mean(features, 1);
    fsd = std(features, 0, 1);
    keep = fsd > 10 * eps;
    features = features(:, keep);
    fsd = fsd(keep);
    if isempty(features)
        features = linear;
        fsd = std(features, 0, 1);
    end
    fsd(fsd < eps) = 1;
    features = features ./ fsd;
    info = struct('rank_est', rank_est, 'linear_dims', size(linear, 2), ...
        'feature_dim', size(features, 2), 'variance_explained_pct', var_pct);
end

function tf = valid_mix(mix, K)
    tf = isstruct(mix) && isfield(mix, 'state') && numel(mix.state) >= K && ...
        isfield(mix, 'fm') && isfinite(double(real(mix.fm)));
end

function labels = assign_mix(X, mix)
    K = mix.m;
    logp = -inf(size(X, 1), K);
    for k = 1:K
        mu = double(real(mix.state(k).m(:)'));
        C = double(real(mix.state(k).C));
        if isfield(mix.state(k), 'prior') && ~isempty(mix.state(k).prior)
            prior = double(real(mix.state(k).prior));
        else
            prior = 1 / K;
        end
        logp(:, k) = log(prior + eps) + log_gaussian(X, mu, C);
    end
    [~, labels] = max(logp, [], 2);
end

function y = log_gaussian(X, mu, C)
    D = size(X, 2);
    mu = mu(1:min(numel(mu), D));
    if numel(mu) < D
        mu = [mu zeros(1, D - numel(mu))];
    end
    C = coerce_cov(C, D);
    C = (C + C') / 2;
    ridge = max(1e-6, mean(diag(C), 'omitnan') * 1e-6);
    [R, p] = chol(C + ridge * eye(D));
    tries = 0;
    while p ~= 0 && tries < 5
        ridge = ridge * 10;
        [R, p] = chol(C + ridge * eye(D));
        tries = tries + 1;
    end
    if p ~= 0
        C = diag(max(var(X, 0, 1), 1e-6));
        R = chol(C);
    end
    Z = bsxfun(@minus, X, mu) / R;
    q = sum(Z.^2, 2);
    logdet = 2 * sum(log(diag(R)));
    y = -0.5 * (D * log(2*pi) + logdet + q);
end

function C = coerce_cov(C, D)
    if isscalar(C)
        C = C * eye(D);
    elseif isvector(C)
        v = C(:);
        C = diag(v(1:min(D, numel(v))));
    end
    if size(C, 1) ~= D || size(C, 2) ~= D
        out = eye(D);
        n = min([D, size(C, 1), size(C, 2)]);
        out(1:n, 1:n) = C(1:n, 1:n);
        C = out;
    end
end

function centers = recover_centers(X, labels, K, fallback)
    X = normalize_maps(X);
    fallback = normalize_maps(fallback);
    centers = zeros(K, size(X, 2));
    for k = 1:K
        idx = labels == k;
        if any(idx)
            Xk = X(idx, :);
            if size(Xk, 1) == 1
                c = Xk(1, :);
            else
                [~, ~, V] = svd(Xk, 'econ');
                c = V(:, 1)';
                if dot(c, mean(Xk, 1)) < 0
                    c = -c;
                end
            end
        elseif size(fallback, 1) >= k
            c = fallback(k, :);
        else
            c = X(mod(k - 1, size(X, 1)) + 1, :);
        end
        centers(k, :) = c;
    end
    centers = normalize_maps(centers);
end

function centers = refine_centers(X, centers, fallback, max_iter)
    centers = normalize_maps(centers);
    labels = assign_by_abs_correlation(X, centers);
    for it = 1:max_iter
        old = labels;
        centers = recover_centers(X, labels, size(centers, 1), fallback);
        labels = assign_by_abs_correlation(X, centers);
        if isequal(old, labels)
            break;
        end
    end
end

function metrics = topo_metrics(X, labels, centers, gfp)
    X = normalize_maps(X);
    centers = normalize_maps(centers);
    sim = abs(X * centers');
    [maxsim, ~] = max(sim, [], 2);
    w = double(gfp(:)).^2;
    if numel(w) ~= size(X, 1) || sum(w) <= eps
        w = ones(size(X, 1), 1);
    end
    metrics = struct();
    metrics.wss = sum(w .* (1 - maxsim.^2));
    metrics.gev = sum(w .* maxsim.^2) / sum(w);
    metrics.silhouette = silhouette_centroid(X, labels, centers);
end

function sil = silhouette_centroid(X, labels, centers)
    K = size(centers, 1);
    if K < 2 || size(X, 1) <= K
        sil = NaN;
        return;
    end
    dist = 1 - abs(X * centers');
    own = labels(:);
    a = dist(sub2ind(size(dist), (1:size(dist, 1))', own));
    dist(sub2ind(size(dist), (1:size(dist, 1))', own)) = Inf;
    b = min(dist, [], 2);
    vals = (b - a) ./ max(a, b);
    vals(~isfinite(vals)) = NaN;
    sil = mean(vals, 'omitnan');
end

function w = cluster_weights(labels, K)
    w = zeros(1, K);
    for k = 1:K
        w(k) = mean(labels == k);
    end
    w = w ./ max(sum(w), eps);
end

function s = empty_mix_summary()
    s = struct('K_candidate', NaN, 'feature_dim', NaN, 'free_energy', NaN, ...
        'priors', [], 'means', [], 'covariance_traces', [], 'feature_info', struct());
end

function s = summarise_mix(mix, K, feature_dim, feature_info)
    s = empty_mix_summary();
    s.K_candidate = K;
    s.feature_dim = feature_dim;
    s.free_energy = double(real(mix.fm));
    s.priors = nan(K, 1);
    s.means = nan(K, feature_dim);
    s.covariance_traces = nan(K, 1);
    s.feature_info = feature_info;
    for k = 1:K
        if isfield(mix.state(k), 'prior')
            s.priors(k) = double(real(mix.state(k).prior));
        end
        if isfield(mix.state(k), 'm')
            m = double(real(mix.state(k).m(:)'));
            s.means(k, 1:min(feature_dim, numel(m))) = m(1:min(feature_dim, numel(m)));
        end
        if isfield(mix.state(k), 'C')
            C = coerce_cov(double(real(mix.state(k).C)), feature_dim);
            s.covariance_traces(k) = trace(C);
        end
    end
end

function Fit = attach_template_alignment_if_available(Fit, cfg, common_labels)
    if isempty(cfg.template_file) || ~isfile(cfg.template_file)
        return;
    end
    try
        alignment = align_microstates_to_template(Fit.centers, cfg.template_file, ...
            'estimated_channel_labels', common_labels, ...
            'strong_threshold', 0.5);
        Fit.template_alignment = alignment;
        Fit.centers = alignment.aligned_maps;
    catch ME
        Fit.template_alignment = struct('labels', {{}}, 'mean_correlation', NaN, ...
            'n_strong_matches', NaN, 'message', ME.message);
    end
end

function save_fit_bundle(out_dir, prefix, Fit, rows, common_labels, common_chanlocs, cfg)
    ensure_dir(out_dir);
    save(fullfile(out_dir, [prefix '_solution.mat']), 'Fit', 'rows', 'common_labels', 'common_chanlocs', 'cfg', '-v7.3');
    write_matrix_csv(fullfile(out_dir, [prefix '_centers.csv']), Fit.centers);
    writetable(Fit.model_comparison, fullfile(out_dir, [prefix '_model_comparison.csv']));
    if istable(rows) && ~isempty(rows)
        writetable(rows, fullfile(out_dir, [prefix '_gfp_peak_manifest.csv']));
    end
    if cfg.save_plots
        plot_centers(Fit, common_chanlocs, fullfile(out_dir, [prefix '_centers.png']));
    end
end

function plot_centers(Fit, chanlocs, out_file)
    try
        centers = Fit.centers;
        K = size(centers, 1);
        fig = figure('Visible', 'off', 'Color', 'white');
        n_cols = min(K, 4);
        n_rows = ceil(K / n_cols);
        for k = 1:K
            subplot(n_rows, n_cols, k);
            if ~isempty(chanlocs) && exist('topoplot', 'file') == 2
                try
                    topoplot(centers(k, :), chanlocs, 'electrodes', 'off', 'numcontour', 6);
                catch
                    imagesc(centers(k, :)); axis tight; colorbar;
                end
            else
                imagesc(centers(k, :)); axis tight; colorbar;
            end
            label = sprintf('State %d', k);
            if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels') && numel(Fit.template_alignment.labels) >= k
                label = sprintf('%s r=%.2f', Fit.template_alignment.labels{k}, Fit.template_alignment.correlations(k));
            end
            title(label, 'Interpreter', 'none');
        end
        sgtitle(Fit.name, 'Interpreter', 'none');
        saveas(fig, out_file);
        close(fig);
    catch ME
        warning('Plot failed for %s: %s', out_file, ME.message);
    end
end

function write_matrix_csv(file_path, X)
    T = array2table(double(X));
    names = cell(1, size(X, 2));
    for i = 1:size(X, 2)
        names{i} = sprintf('ch_%03d', i);
    end
    T.Properties.VariableNames = names;
    writetable(T, file_path);
end

function row = make_summary_row(level, name, group, condition, participant, Fit, n_units, n_maps, fit_path)
    [mean_corr, strong] = alignment_score(Fit);
    row = table(string(level), string(name), string(group), string(condition), string(participant), ...
        double(Fit.K_estimated), double(n_units), double(n_maps), double(mean_corr), double(strong), string(fit_path), ...
        'VariableNames', {'level', 'name', 'group', 'condition', 'participant', 'K_estimated', 'n_units', 'n_maps', 'template_mean_corr', 'template_strong_matches', 'fit_path'});
end

function [mean_corr, strong] = alignment_score(Fit)
    mean_corr = NaN;
    strong = NaN;
    if isfield(Fit, 'template_alignment')
        A = Fit.template_alignment;
        if isfield(A, 'mean_correlation'), mean_corr = A.mean_correlation; end
        if isfield(A, 'n_strong_matches'), strong = A.n_strong_matches; end
    end
end

function node = build_node(level, participant, group, condition, name, Fit, n_maps, fit_path)
    node = struct();
    node.level = string(level);
    node.participant = string(participant);
    node.group = string(group);
    node.condition = string(condition);
    node.name = string(name);
    node.centers = Fit.centers;
    node.K_estimated = Fit.K_estimated;
    node.n_maps = n_maps;
    node.fit = Fit;
    node.fit_path = string(fit_path);
end

function nodes = empty_nodes(n_channels)
    nodes = repmat(build_node("", "", "", "", "", struct('centers', zeros(1, n_channels), 'K_estimated', 0), 0, ""), 0, 1);
end

function print_alignment_line(label, Fit)
    if ~isfield(Fit, 'template_alignment')
        return;
    end
    A = Fit.template_alignment;
    if isfield(A, 'mean_correlation') && isfinite(A.mean_correlation)
        fprintf('%s template fit: mean r=%.3f, strong=%d/%d\n', label, A.mean_correlation, A.n_strong_matches, Fit.K_estimated);
    end
end

function labels = assign_by_abs_correlation(maps, centers)
    X = normalize_maps(double(maps));
    C = normalize_maps(double(centers));
    [~, labels] = max(abs(X * C'), [], 2);
end

function Xn = normalize_maps(X)
    X = double(X);
    if isempty(X)
        Xn = X;
        return;
    end
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function Xn = normalize_rows(X)
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function idx = deterministic_subsample(N, maxN)
    if isempty(maxN) || ~isfinite(maxN) || N <= maxN
        idx = (1:N)';
        return;
    end
    idx = unique(round(linspace(1, N, round(maxN))))';
    if numel(idx) > maxN
        idx = idx(1:maxN);
    end
end

function ensure_dir(pth)
    if ~exist(pth, 'dir')
        mkdir(pth);
    end
end

function s = onoff(tf)
    if tf
        s = 'ON';
    else
        s = 'OFF';
    end
end

function key = clean_key(value)
    key = char(string(value));
    key = regexprep(key, '[^A-Za-z0-9_-]', '_');
    if isempty(key)
        key = 'empty';
    end
end
