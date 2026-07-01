function [HResults, results_mat] = fit_microstates_heppy_global_group_condition(root_path, varargin)
% FIT_MICROSTATES_HEPPY_GLOBAL_GROUP_CONDITION
% HEPPy-specific global/group/condition microstate fitting and global-template backfitting.
%
% Current HEPPy layout expected by this wrapper:
%
%   <root>/heppy/raw_fif/suj_1_pp_raw.fif             CFA-removed preprocessed EEG
%   <root>/heppy/raw_fif/suj_1_pp_raw_keepcfa.fif     same preprocessing, CFA kept
%   <root>/heppy/raw_fif/suj_1_ica.fif                ICA solution for CFA-removed raw
%   <root>/heppy/raw_fif/suj_1_keepcfa_ica.fif        ICA solution for keep-CFA raw
%
% The microstate fit uses only exact suj_N_pp_raw.fif files. It deliberately
% excludes *_keepcfa* and *_ica.fif files from the EEG input set. Because the
% current layout no longer contains pre-split condition FIF files, this wrapper
% calls prepare_heppy_condition_fifs.py to crop/concatenate condition-specific
% FIF files from annotations before running fit_microstate_hierarchical_dataset.m.
%
% Example:
%   [H, matfile] = fit_microstates_heppy_global_group_condition( ...
%       '/home/rohan/EEG/Microstates_and_Interoception', ...
%       'output_dir', '/home/rohan/EEG/Microstates_and_Interoception/output/heppy_microstates_matlab', ...
%       'K_candidates', 4:7, ...
%       'conditions', {'intero','extero','feedback','intero_fb'}, ...
%       'prepare_condition_fifs', true);

    if nargin < 1 || isempty(root_path)
        root_path = fullfile(getenv('HOME'), 'EEG', 'Microstates_and_Interoception');
    end

    p = inputParser;
    addRequired(p, 'root_path', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'conditions', {'intero','extero','feedback','intero_fb'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'include_full_record', false, @islogical);
    addParameter(p, 'group_table', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'condition_markers_json', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'prepare_condition_fifs', true, @islogical);
    addParameter(p, 'condition_fif_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'python_executable', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'condition_padding_s', 1.0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'condition_max_marker_gap_s', 5.0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'condition_min_duration_s', 2.0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'overwrite_condition_fifs', false, @islogical);
    addParameter(p, 'K_candidates', 4:7, @isnumeric);
    addParameter(p, 'criterion', 'free_energy_covariance', @(x) ischar(x) || isstring(x));
    addParameter(p, 'force_k', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_label_override', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'apply_average_reference', true, @islogical);
    addParameter(p, 'filter_band', [1 40], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'spatial_filter', 'smoothing', @(x) ischar(x) || isstring(x));
    addParameter(p, 'spatial_filter_neighbours', 12, @isnumeric);
    addParameter(p, 'spatial_filter_strength', 0.75, @isnumeric);
    addParameter(p, 'exclude_channels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'max_maps_per_file', 4000, @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_global_maps', 120000, @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_child_maps', 120000, @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'random_seed', 42, @isnumeric);
    addParameter(p, 'save_plots', true, @islogical);
    addParameter(p, 'verbose', true, @islogical);
    parse(p, root_path, varargin{:});
    cfg = p.Results;

    cfg.root_path = char(cfg.root_path);
    cfg.conditions = cellstr(string(cfg.conditions));
    cfg.conditions = cellfun(@(x) lower(strtrim(x)), cfg.conditions, 'UniformOutput', false);
    cfg.exclude_channels = cellstr(string(cfg.exclude_channels));
    cfg.template_label_override = cellstr(string(cfg.template_label_override));
    cfg.spatial_filter = char(cfg.spatial_filter);
    if isempty(char(cfg.output_dir))
        cfg.output_dir = fullfile(cfg.root_path, 'output', 'heppy_microstates_matlab');
    else
        cfg.output_dir = char(cfg.output_dir);
    end
    ensure_dir_local(cfg.output_dir);

    heppy_root = resolve_heppy_root(cfg.root_path);
    raw_dir = fullfile(heppy_root, 'raw_fif');
    if ~isfolder(raw_dir)
        error('HEPPy raw_fif directory not found: %s', raw_dir);
    end

    dirs = make_heppy_dirs(cfg.output_dir);
    if isempty(char(cfg.condition_fif_dir))
        cfg.condition_fif_dir = dirs.condition_fifs;
    else
        cfg.condition_fif_dir = char(cfg.condition_fif_dir);
        ensure_dir_local(cfg.condition_fif_dir);
    end

    manifest_csv = fullfile(dirs.manifest, 'normalised_heppy_manifest.csv');
    if cfg.prepare_condition_fifs
        helper = fullfile(fileparts(mfilename('fullpath')), 'prepare_heppy_condition_fifs.py');
        if ~isfile(helper)
            error(['prepare_heppy_condition_fifs.py not found next to this MATLAB file. ', ...
                   'Copy both files into the same analysis directory. Missing: %s'], helper);
        end
        run_condition_fif_helper(helper, raw_dir, cfg.condition_fif_dir, manifest_csv, cfg);
        manifest = read_heppy_manifest_csv(manifest_csv);
    else
        manifest = build_full_record_manifest(raw_dir, cfg.include_full_record, cfg.group_table);
        if isempty(manifest) || height(manifest) == 0
            error(['prepare_condition_fifs=false but no manifest rows were available. ', ...
                   'Either set prepare_condition_fifs=true or include_full_record=true.']);
        end
        writetable(manifest, manifest_csv);
    end

    if isempty(manifest) || height(manifest) == 0
        error('No HEPPy condition FIF files were found/prepared from %s.', raw_dir);
    end

    if manifest_uses_fif(manifest)
        [manifest, manifest_csv] = convert_fif_manifest_to_mat(manifest_csv, dirs.condition_mats, cfg);
    end

    if cfg.verbose
        fprintf('\nHEPPy raw_fif input: %s\n', raw_dir);
        fprintf('Prepared/manifest records: %d\n', height(manifest));
        try
            disp(groupsummary(manifest, {'study','group','condition'}));
        catch
            disp(manifest(:, intersect({'study','group','participant','condition','file_path'}, manifest.Properties.VariableNames, 'stable')));
        end
        fprintf('Output: %s\n\n', cfg.output_dir);
    end

    fit_args = { ...
        'output_dir', cfg.output_dir, ...
        'K_candidates', cfg.K_candidates, ...
        'criterion', char(cfg.criterion), ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength, ...
        'max_maps_per_file', cfg.max_maps_per_file, ...
        'max_global_maps', cfg.max_global_maps, ...
        'max_child_maps', cfg.max_child_maps, ...
        'run_backfit', false, ...
        'save_plots', cfg.save_plots, ...
        'random_seed', cfg.random_seed, ...
        'verbose', cfg.verbose, ...
        'exclude_channels', cfg.exclude_channels};

    if ~isempty(char(cfg.template_file))
        fit_args = [fit_args, {'template_file', char(cfg.template_file)}]; %#ok<AGROW>
    end
    if ~isempty(cfg.template_label_override)
        fit_args = [fit_args, {'template_label_override', cfg.template_label_override}]; %#ok<AGROW>
    end
    if ~isempty(cfg.force_k)
        fit_args{find(strcmp(fit_args, 'K_candidates')) + 1} = double(cfg.force_k);
    end

    [HResults, results_mat] = fit_microstate_hierarchical_dataset(manifest_csv, fit_args{:});

    save_cluster_export_bundle(HResults, dirs, cfg);
    if isfield(HResults, 'hierarchy_summary') && istable(HResults.hierarchy_summary)
        writetable(HResults.hierarchy_summary, fullfile(dirs.summary, 'hierarchical_fit_summary.csv'));
    end

    if cfg.verbose
        fprintf('\nBackfitting GLOBAL templates to participant-condition records...\n');
    end
    [state_metrics, record_metrics] = backfit_global_templates_to_manifest(HResults, manifest, dirs, cfg);
    writetable(state_metrics, fullfile(dirs.summary, 'global_backfit_state_metrics.csv'));
    writetable(record_metrics, fullfile(dirs.summary, 'global_backfit_record_metrics.csv'));

    HResults.heppy = struct();
    HResults.heppy.root = heppy_root;
    HResults.heppy.raw_fif_dir = raw_dir;
    HResults.heppy.condition_fif_dir = cfg.condition_fif_dir;
    HResults.heppy.manifest_csv = manifest_csv;
    HResults.heppy.global_backfit_state_metrics_csv = fullfile(dirs.summary, 'global_backfit_state_metrics.csv');
    HResults.heppy.global_backfit_record_metrics_csv = fullfile(dirs.summary, 'global_backfit_record_metrics.csv');
    HResults.heppy.backfit_dir = dirs.backfit_global;
    HResults.heppy.cluster_export_dir = dirs.clusters;
    save(results_mat, 'HResults', '-v7.3');

    if cfg.verbose
        fprintf('\nFinished HEPPy microstate fitting/backfitting.\n');
        fprintf('Results MAT: %s\n', results_mat);
        fprintf('Global backfit sequences: %s\n', dirs.backfit_global);
        fprintf('State metrics: %s\n', HResults.heppy.global_backfit_state_metrics_csv);
    end
end

function heppy_root = resolve_heppy_root(root_path)
    root_path = char(root_path);
    if strcmpi(get_last_path_part(root_path), 'heppy')
        heppy_root = root_path;
    else
        heppy_root = fullfile(root_path, 'heppy');
    end
end

function name = get_last_path_part(pth)
    parts = regexp(char(pth), '[\/]', 'split');
    parts = parts(~cellfun(@isempty, parts));
    if isempty(parts), name = ''; else, name = parts{end}; end
end

function dirs = make_heppy_dirs(output_dir)
    dirs = struct();
    dirs.root = output_dir;
    dirs.manifest = fullfile(output_dir, 'manifest');
    dirs.condition_fifs = fullfile(output_dir, 'condition_fifs');
    dirs.condition_mats = fullfile(output_dir, 'condition_mats');
    dirs.summary = fullfile(output_dir, 'summary');
    dirs.clusters = fullfile(output_dir, 'clusters');
    dirs.clusters_global = fullfile(dirs.clusters, 'global');
    dirs.clusters_groups = fullfile(dirs.clusters, 'groups');
    dirs.clusters_conditions = fullfile(dirs.clusters, 'conditions');
    dirs.plots = fullfile(output_dir, 'plots');
    dirs.topoplots = fullfile(dirs.plots, 'topoplots');
    dirs.backfit = fullfile(output_dir, 'backfit');
    dirs.backfit_global = fullfile(dirs.backfit, 'global');
    names = fieldnames(dirs);
    for i = 1:numel(names)
        ensure_dir_local(dirs.(names{i}));
    end
end

function run_condition_fif_helper(helper, raw_dir, out_dir, manifest_csv, cfg)
    py = default_python_executable(cfg);
    conds = strjoin(cellfun(@shell_quote, cfg.conditions, 'UniformOutput', false), ' ');
    cmd = [shell_quote(py) ' ' shell_quote(helper) ...
        ' --raw-dir ' shell_quote(raw_dir) ...
        ' --out-dir ' shell_quote(out_dir) ...
        ' --manifest ' shell_quote(manifest_csv) ...
        ' --conditions ' conds ...
        ' --padding-s ' num2str(double(cfg.condition_padding_s), '%.12g') ...
        ' --max-marker-gap-s ' num2str(double(cfg.condition_max_marker_gap_s), '%.12g') ...
        ' --min-duration-s ' num2str(double(cfg.condition_min_duration_s), '%.12g')];
    if cfg.include_full_record
        cmd = [cmd ' --include-full-record'];
    end
    if cfg.overwrite_condition_fifs
        cmd = [cmd ' --overwrite'];
    end
    if ~isempty(char(cfg.group_table))
        cmd = [cmd ' --group-table ' shell_quote(char(cfg.group_table))];
    end
    if ~isempty(char(cfg.condition_markers_json))
        cmd = [cmd ' --condition-markers-json ' shell_quote(char(cfg.condition_markers_json))];
    end
    [status, out] = system(cmd);
    if status ~= 0
        error('Condition FIF preparation failed with status %d. Command output:\n%s', status, out);
    end
    if ~isfile(manifest_csv)
        error('Condition FIF helper completed but did not create manifest: %s\nOutput:\n%s', manifest_csv, out);
    end
end

function tf = manifest_uses_fif(manifest)
    tf = false;
    if ~istable(manifest) || ~ismember('file_path', manifest.Properties.VariableNames)
        return;
    end
    [~, ~, ext] = cellfun(@fileparts, cellstr(string(manifest.file_path)), 'UniformOutput', false);
    tf = any(ismember(lower(string(ext)), [".fif"; ".fiff"]));
end

function T = read_heppy_manifest_csv(path)
    T = readtable(path, 'TextType', 'string', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
end

function [manifest, converted_manifest_csv] = convert_fif_manifest_to_mat(manifest_csv, out_dir, cfg)
    helper = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'convert_fif_manifest_to_set.py');
    if ~isfile(helper)
        error('FIF conversion helper not found: %s', helper);
    end
    ensure_dir_local(out_dir);
    converted_manifest_csv = fullfile(out_dir, 'converted_heppy_manifest.csv');
    py = default_python_executable(cfg);
    cmd = [shell_quote(py) ' ' shell_quote(helper) ' ' shell_quote(manifest_csv) ' ' shell_quote(out_dir) ...
        ' --format mat' ...
        ' --output-manifest ' shell_quote(converted_manifest_csv)];
    if cfg.overwrite_condition_fifs
        cmd = [cmd ' --overwrite'];
    end
    [status, out] = system(cmd);
    if status ~= 0
        error(['MATLAB cannot read the prepared FIF files and automatic FIF-to-MAT conversion failed. ', ...
               'Command output:\n%s'], out);
    end
    manifest = read_heppy_manifest_csv(converted_manifest_csv);
end

function py = default_python_executable(cfg)
    py = char(cfg.python_executable);
    if ~isempty(py)
        return;
    end
    if ispc
        py = 'python';
        return;
    end
    candidates = {'/opt/homebrew/anaconda3/bin/python3', '/opt/homebrew/bin/python3', 'python3', 'python'};
    py = candidates{find(cellfun(@python_candidate_exists, candidates), 1, 'first')};
end

function tf = python_candidate_exists(py)
    if contains(py, filesep)
        tf = isfile(py);
    else
        [status, ~] = system(['command -v ' py]);
        tf = status == 0;
    end
end

function q = shell_quote(x)
    x = char(x);
    % Double quoting is adequate for the file paths used here and avoids
    % fragile platform-specific single-quote escaping inside MATLAB strings.
    q = ['"' strrep(x, '"', '\"') '"'];
end

function manifest = build_full_record_manifest(raw_dir, include_full, group_table)
    if ~include_full
        manifest = table();
        return;
    end
    files = dir(fullfile(raw_dir, 'suj_*_pp_raw.fif'));
    override = read_group_override_table(group_table);
    rows = {};
    for i = 1:numel(files)
        if isempty(regexp(files(i).name, '^suj_\d+_pp_raw\.fif$', 'once'))
            continue;
        end
        fpath = fullfile(files(i).folder, files(i).name);
        participant = regexp(lower(files(i).name), '^(suj_\d+)', 'match', 'once');
        [group, study] = infer_group_from_participant(participant);
        if isKey(override.group_by_participant, lower(participant))
            group = override.group_by_participant(lower(participant));
            study = study_from_group(group);
        end
        rows(end+1,:) = {string(participant), "full", string(group), string(study), string(fpath), string(fpath), string(fpath), "", "", ""}; %#ok<AGROW>
    end
    if isempty(rows)
        manifest = table();
    else
        manifest = cell2table(rows, 'VariableNames', {'participant','condition','group','study','file_path','raw_fif','cfa_removed_fif','keepcfa_fif','ica_fif','keepcfa_ica_fif'});
        manifest = sortrows(manifest, {'study','group','participant','condition'});
    end
end

function override = read_group_override_table(group_table)
    override = struct();
    override.group_by_participant = containers.Map('KeyType', 'char', 'ValueType', 'char');
    group_table = char(group_table);
    if isempty(group_table) || ~isfile(group_table)
        return;
    end
    T = readtable(group_table, 'TextType', 'string');
    names = lower(regexprep(string(T.Properties.VariableNames), '[^a-z0-9]', ''));
    pcol = find(ismember(names, ["participant","subject","subjectid","id"]), 1, 'first');
    gcol = find(ismember(names, ["group","condition","diagnosis"]), 1, 'first');
    if isempty(pcol) || isempty(gcol)
        warning('Group table ignored; participant/group columns were not found: %s', group_table);
        return;
    end
    for r = 1:height(T)
        sid = lower(strtrim(char(string(T{r, pcol}))));
        grp = strtrim(char(string(T{r, gcol})));
        if ~isempty(sid) && ~isempty(grp)
            override.group_by_participant(sid) = grp;
        end
    end
end

function [group, study] = infer_group_from_participant(participant)
    tok = regexp(char(participant), '(\d+)', 'tokens', 'once');
    if isempty(tok)
        group = 'unknown'; study = 'unknown'; return;
    end
    n = str2double(tok{1});
    if n >= 1 && n < 100
        group = 'ANX'; study = 'ANX';
    elseif n >= 100 && n < 200
        group = 'NANX'; study = 'ANX';
    elseif n >= 700 && n < 800
        group = 'HTN'; study = 'HTN';
    elseif n >= 800 && n < 900
        group = 'NHTN'; study = 'HTN';
    else
        group = 'unknown'; study = 'unknown';
    end
end

function study = study_from_group(group)
    g = upper(char(group));
    if ismember(g, {'ANX','NANX','ANXIETY','NONANXIOUS','NON_ANXIOUS'})
        study = 'ANX';
    elseif ismember(g, {'HTN','NHTN','HYPERTENSION','NONHYPERTENSION','NON_HTN'})
        study = 'HTN';
    else
        study = 'unknown';
    end
end

function save_cluster_export_bundle(HResults, dirs, cfg)
    if ~isfield(HResults, 'global') || ~isfield(HResults.global, 'fit')
        warning('HResults.global.fit is missing; cluster export skipped.');
        return;
    end
    global_fit = HResults.global.fit;
    save(fullfile(dirs.clusters_global, 'global_cluster.mat'), 'global_fit', '-v7.3');
    write_matrix_csv_local(fullfile(dirs.clusters_global, 'global_centers.csv'), global_fit.centers);
    if cfg.save_plots
        chanlocs = [];
        if isfield(HResults, 'common_chanlocs'), chanlocs = HResults.common_chanlocs; end
        plot_topoplot_centers_local(global_fit, chanlocs, fullfile(dirs.topoplots, 'global_microstates_topoplot.png'), 'Global microstates');
    end

    if isfield(HResults, 'groups')
        group_summary_rows = {};
        for i = 1:numel(HResults.groups)
            node = HResults.groups(i);
            if ~isfield(node, 'fit') || isempty(node.fit), continue; end
            key = clean_key_local(node.group);
            out_dir = fullfile(dirs.clusters_groups, key);
            ensure_dir_local(out_dir);
            Fit = node.fit; %#ok<NASGU>
            save(fullfile(out_dir, 'group_cluster.mat'), 'Fit', 'node', '-v7.3');
            write_matrix_csv_local(fullfile(out_dir, 'group_centers.csv'), node.fit.centers);
            if cfg.save_plots
                plot_topoplot_centers_local(node.fit, chanlocs_from_results(HResults), fullfile(out_dir, 'group_topoplot.png'), ['Group ' char(node.group)]);
            end
            group_summary_rows(end+1,:) = {string(node.group), string(out_dir), double(node.fit.K_estimated), double(node.n_maps)}; %#ok<AGROW>
        end
        if ~isempty(group_summary_rows)
            writetable(cell2table(group_summary_rows, 'VariableNames', {'group','output_dir','K_estimated','n_maps'}), fullfile(dirs.summary, 'group_cluster_exports.csv'));
        end
    end

    if isfield(HResults, 'conditions')
        condition_summary_rows = {};
        for i = 1:numel(HResults.conditions)
            node = HResults.conditions(i);
            if ~isfield(node, 'fit') || isempty(node.fit), continue; end
            key = clean_key_local(node.condition);
            out_dir = fullfile(dirs.clusters_conditions, key);
            ensure_dir_local(out_dir);
            Fit = node.fit; %#ok<NASGU>
            save(fullfile(out_dir, 'condition_cluster.mat'), 'Fit', 'node', '-v7.3');
            write_matrix_csv_local(fullfile(out_dir, 'condition_centers.csv'), node.fit.centers);
            if cfg.save_plots
                plot_topoplot_centers_local(node.fit, chanlocs_from_results(HResults), fullfile(out_dir, 'condition_topoplot.png'), ['Condition ' char(node.condition)]);
            end
            condition_summary_rows(end+1,:) = {string(node.condition), string(out_dir), double(node.fit.K_estimated), double(node.n_maps)}; %#ok<AGROW>
        end
        if ~isempty(condition_summary_rows)
            writetable(cell2table(condition_summary_rows, 'VariableNames', {'condition','output_dir','K_estimated','n_maps'}), fullfile(dirs.summary, 'condition_cluster_exports.csv'));
        end
    end
end

function chanlocs = chanlocs_from_results(HResults)
    chanlocs = [];
    if isfield(HResults, 'common_chanlocs')
        chanlocs = HResults.common_chanlocs;
    end
end

function [state_metrics, record_metrics] = backfit_global_templates_to_manifest(HResults, manifest, dirs, cfg)
    state_rows = {};
    record_rows = {};
    global_fit = HResults.global.fit;
    centers = normalize_rows_local(double(global_fit.centers));
    labels = state_labels_from_fit(global_fit, size(centers, 1));

    common_labels = {};
    if isfield(HResults, 'common_channel_labels')
        common_labels = cellstr(string(HResults.common_channel_labels));
    elseif isfield(HResults, 'common_labels')
        common_labels = cellstr(string(HResults.common_labels));
    end
    common_pos = [];
    if isfield(HResults, 'common_pos')
        common_pos = HResults.common_pos;
    end

    for i = 1:height(manifest)
        participant = char(manifest.participant(i));
        condition = char(manifest.condition(i));
        group = char(manifest.group(i));
        study = char(manifest.study(i));
        file_path = char(manifest.file_path(i));
        out_dir = fullfile(dirs.backfit_global, clean_key_local(study), clean_key_local(group), clean_key_local(participant), clean_key_local(condition));
        ensure_dir_local(out_dir);

        try
            [X, sfreq, channel_labels, pos] = load_heppy_eeg_matrix(file_path);
            if ~isempty(common_labels)
                X = reorder_to_common_channels(X, channel_labels, pos, common_labels, common_pos, file_path);
            end
            pre_pos = common_pos;
            if isempty(pre_pos), pre_pos = pos; end
            X = preprocess_for_backfit_local(X, sfreq, pre_pos, cfg);
            [assign_1based, corr_signed, corr_abs, gfp] = hard_backfit_maps(X, centers);
            n_samples = numel(assign_1based);
            time_s = (0:(n_samples-1))' ./ sfreq;
            names = strings(n_samples, 1);
            for s = 1:n_samples
                names(s) = string(labels{assign_1based(s)});
            end

            seq = table((0:(n_samples-1))', time_s, assign_1based(:)-1, assign_1based(:), names(:), ...
                corr_signed(:), corr_abs(:), gfp(:), ...
                'VariableNames', {'sample_index','time_s','microstate_label','state_index_1based','microstate_name','corr_signed','corr_abs','gfp'});
            seq.participant = repmat(string(participant), n_samples, 1);
            seq.group = repmat(string(group), n_samples, 1);
            seq.study = repmat(string(study), n_samples, 1);
            seq.condition = repmat(string(condition), n_samples, 1);
            seq = movevars(seq, {'study','group','participant','condition'}, 'Before', 'sample_index');

            csv_path = fullfile(out_dir, sprintf('%s_%s_global_sequence.csv', participant, condition));
            mat_path = fullfile(out_dir, sprintf('%s_%s_global_backfit.mat', participant, condition));
            writetable(seq, csv_path);
            save(mat_path, 'assign_1based', 'corr_signed', 'corr_abs', 'gfp', 'sfreq', 'labels', 'centers', 'participant', 'condition', 'group', 'study', 'file_path', '-v7.3');

            [Ts, Tr] = summarise_global_backfit(assign_1based, gfp, sfreq, labels, participant, condition, group, study, file_path, csv_path);
            state_rows{end+1,1} = Ts; %#ok<AGROW>
            record_rows{end+1,1} = Tr; %#ok<AGROW>
        catch ME
            warning('Global backfit failed for %s %s (%s): %s', participant, condition, file_path, ME.message);
        end
    end

    state_metrics = vertcat_nonempty_local(state_rows);
    record_metrics = vertcat_nonempty_local(record_rows);
end

function labels = state_labels_from_fit(Fit, K)
    labels = arrayfun(@(k) sprintf('%c', char('A' + k - 1)), 1:K, 'UniformOutput', false);
    if isfield(Fit, 'template_alignment') && isstruct(Fit.template_alignment) && isfield(Fit.template_alignment, 'labels')
        tmp = cellstr(string(Fit.template_alignment.labels(:)));
        if numel(tmp) >= K
            labels = tmp(1:K);
        end
    end
end

function [assign_1based, corr_signed, corr_abs, gfp] = hard_backfit_maps(X, centers)
    X = double(X);
    gfp = std(X, 0, 1, 'omitnan')';
    X = X - mean(X, 1, 'omitnan');
    denom = sqrt(sum(X.^2, 1));
    denom(denom <= eps | ~isfinite(denom)) = NaN;
    Xn = bsxfun(@rdivide, X, denom);
    Xn(~isfinite(Xn)) = 0;
    C = normalize_rows_local(centers);
    R = Xn' * C';
    [corr_abs, assign_1based] = max(abs(R), [], 2);
    row_idx = (1:size(R,1))';
    corr_signed = R(sub2ind(size(R), row_idx, assign_1based));
end

function [Ts, Tr] = summarise_global_backfit(assign_1based, gfp, sfreq, labels, participant, condition, group, study, file_path, sequence_csv)
    K = numel(labels);
    n_samples = numel(assign_1based);
    duration_s = n_samples / max(sfreq, eps);
    rows = cell(K, 1);
    for k = 1:K
        hit = assign_1based(:) == k;
        run_lengths = active_run_lengths_local(hit);
        rows{k} = table(string(study), string(group), string(participant), string(condition), string(file_path), string(sequence_csv), ...
            double(k-1), double(k), string(labels{k}), double(n_samples), double(sfreq), double(duration_s), ...
            double(mean(hit)), double(100 * mean(hit)), double(mean_or_nan_local(gfp(hit))), ...
            double(numel(run_lengths)), double(numel(run_lengths) / max(duration_s, eps)), ...
            double(mean_or_nan_local(run_lengths) * 1000 / max(sfreq, eps)), ...
            double(median_or_nan_local(run_lengths) * 1000 / max(sfreq, eps)), ...
            'VariableNames', {'study','group','participant','condition','file_path','sequence_csv', ...
            'microstate_label','state_index_1based','microstate_name','n_samples','sfreq','duration_s', ...
            'coverage','percentage_record_present','gfp','occurrence_count','occurrence_rate_hz', ...
            'mean_duration_ms','median_duration_ms'});
    end
    Ts = vertcat(rows{:});
    Tr = table(string(study), string(group), string(participant), string(condition), string(file_path), string(sequence_csv), ...
        double(K), double(n_samples), double(sfreq), double(duration_s), double(mean(gfp, 'omitnan')), ...
        'VariableNames', {'study','group','participant','condition','file_path','sequence_csv','K_estimated','n_samples','sfreq','duration_s','mean_gfp'});
end

function [X, sfreq, labels, pos] = load_heppy_eeg_matrix(file_path)
    file_path = char(file_path);
    [~, ~, ext] = fileparts(file_path);
    ext = lower(ext);
    X = [];
    sfreq = 250;
    labels = {};
    pos = [];

    if any(strcmp(ext, {'.fif', '.fiff'})) && exist('ft_read_header', 'file') == 2 && exist('ft_read_data', 'file') == 2
        hdr = ft_read_header(file_path);
        X = double(ft_read_data(file_path));
        sfreq = double(hdr.Fs);
        labels = cellstr(string(hdr.label(:)));
        keep = true(numel(labels), 1);
        if isfield(hdr, 'chantype') && numel(hdr.chantype) >= numel(labels)
            types = lower(string(hdr.chantype(:)));
            eeg = types == "eeg" | contains(types, "eeg");
            if any(eeg), keep = eeg; end
        end
        X = squeeze(X);
        if ndims(X) > 2, X = reshape(X, size(X,1), []); end
        X = X(keep, :);
        labels = labels(keep);
        pos = positions_from_fieldtrip_header(hdr, labels);
        return;
    end

    if any(strcmp(ext, {'.fif', '.fiff'})) && exist('fiff_setup_read_raw', 'file') == 2
        raw = fiff_setup_read_raw(file_path);
        [Xraw, ~] = fiff_read_raw_segment(raw, raw.first_samp, raw.last_samp);
        sfreq = double(raw.info.sfreq);
        labels_all = mne_info_channel_labels_local(raw.info);
        keep = mne_eeg_channel_mask_local(raw.info, numel(labels_all));
        X = double(Xraw(keep, :));
        labels = labels_all(keep);
        pos = mne_info_positions_local(raw.info, keep);
        return;
    end

    if strcmp(ext, '.mat')
        S = load(file_path);
        if isfield(S, 'EEG')
            X = double(S.EEG.data);
            if isfield(S.EEG, 'srate'), sfreq = double(S.EEG.srate); end
            if isfield(S.EEG, 'chanlocs') && ~isempty(S.EEG.chanlocs)
                labels = cellstr(string({S.EEG.chanlocs.labels}));
                pos = positions_from_chanlocs_local(S.EEG.chanlocs, numel(labels));
            end
        elseif isfield(S, 'data')
            X = double(S.data);
        elseif isfield(S, 'eeg_data')
            X = double(S.eeg_data);
        else
            error('No EEG data field found in %s.', file_path);
        end
        if isfield(S, 'sfreq'), sfreq = double(S.sfreq); end
        if isfield(S, 'srate'), sfreq = double(S.srate); end
        if isempty(labels) && isfield(S, 'labels')
            labels = cellstr(string(S.labels(:)));
        end
        if isempty(pos) && isfield(S, 'pos')
            pos = double(squeeze(S.pos));
            if size(pos, 2) ~= 3 && size(pos, 1) == 3
                pos = pos';
            end
        end
        if isempty(labels)
            labels = arrayfun(@(i) sprintf('Ch%d', i), 1:size(X,1), 'UniformOutput', false);
        end
        [labels, pos] = apply_heppy_chanloc_sidecar(file_path, labels, pos);
        return;
    end

    if strcmp(ext, '.set') && exist('pop_loadset', 'file') == 2
        EEG = pop_loadset(file_path);
        X = double(EEG.data);
        if isfield(EEG, 'srate'), sfreq = double(EEG.srate); end
        if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
            labels = cellstr(string({EEG.chanlocs.labels}));
            pos = positions_from_chanlocs_local(EEG.chanlocs, numel(labels));
        else
            labels = arrayfun(@(i) sprintf('Ch%d', i), 1:size(X,1), 'UniformOutput', false);
            pos = [];
        end
        [labels, pos] = apply_heppy_chanloc_sidecar(file_path, labels, pos);
        return;
    end

    if exist('pop_fileio', 'file') == 2
        EEG = pop_fileio(file_path);
        X = double(EEG.data);
        if isfield(EEG, 'srate'), sfreq = double(EEG.srate); end
        if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
            labels = cellstr(string({EEG.chanlocs.labels}));
            pos = positions_from_chanlocs_local(EEG.chanlocs, numel(labels));
        else
            labels = arrayfun(@(i) sprintf('Ch%d', i), 1:size(X,1), 'UniformOutput', false);
            pos = [];
        end
        [labels, pos] = apply_heppy_chanloc_sidecar(file_path, labels, pos);
        return;
    end

    error('No supported reader found for %s. Add FieldTrip, MNE-MATLAB, or EEGLAB FileIO to the MATLAB path.', file_path);
end

function X = reorder_to_common_channels(X, labels, pos, common_labels, common_pos, file_path)
    can = canonical_labels_local(labels);
    common_can = canonical_labels_local(common_labels);
    out = nan(numel(common_can), size(X, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            out(c, :) = double(X(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if isempty(pos) || isempty(common_pos)
            error('Missing channels in %s and no positions available for interpolation. First missing: %s', file_path, common_labels{missing(1)});
        end
        available = find(isfinite(idx));
        if numel(available) < 4
            error('Too few observed common channels in %s for interpolation.', file_path);
        end
        source_xyz = pos(idx(available), :);
        for ii = 1:numel(missing)
            m = missing(ii);
            d = sqrt(sum((source_xyz - common_pos(m,:)).^2, 2));
            [ds, ord] = sort(d, 'ascend');
            n = min(6, numel(ord));
            ord = ord(1:n);
            ds = ds(1:n);
            if ds(1) <= eps
                w = zeros(n, 1); w(1) = 1;
            else
                w = 1 ./ (ds.^2 + eps);
                w = w ./ sum(w);
            end
            out(m, :) = w' * out(available(ord), :);
        end
    end
    X = out;
end

function X = preprocess_for_backfit_local(X, sfreq, pos, cfg)
    X = double(X);
    if cfg.apply_average_reference
        X = X - mean(X, 1, 'omitnan');
    end
    util = microstate_utilities();
    Sim = struct();
    Sim.X_noisy = X;
    Sim.sfreq = sfreq;
    Sim.pos = pos;
    Sim.preprocessing = struct( ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength);
    [X, ~] = util.apply_spatial_filter(X, Sim);
    if ~isempty(cfg.filter_band) && numel(cfg.filter_band) == 2 && all(isfinite(cfg.filter_band)) && exist('butter', 'file') == 2 && exist('filtfilt', 'file') == 2
        band = double(cfg.filter_band(:)');
        band(1) = max(band(1), 0.001);
        band(2) = min(band(2), 0.99 * sfreq / 2);
        if band(1) < band(2)
            [b, a] = butter(4, band / (sfreq / 2), 'bandpass');
            X = filtfilt(b, a, X')';
        end
    end
end

function pos = positions_from_fieldtrip_header(hdr, labels)
    pos = nan(numel(labels), 3);
    if ~isfield(hdr, 'elec') || isempty(hdr.elec)
        return;
    end
    elec = hdr.elec;
    coords = [];
    for f = {'chanpos','elecpos','pnt'}
        if isfield(elec, f{1}) && size(elec.(f{1}), 2) >= 3
            coords = double(elec.(f{1})(:, 1:3));
            break;
        end
    end
    if isempty(coords), return; end
    if isfield(elec, 'label') && ~isempty(elec.label)
        elec_labels = cellstr(string(elec.label(:)));
        can_elec = canonical_labels_local(elec_labels);
        can_labels = canonical_labels_local(labels);
        for i = 1:numel(labels)
            j = find(strcmp(can_elec, can_labels{i}), 1, 'first');
            if ~isempty(j) && j <= size(coords, 1), pos(i,:) = coords(j,:); end
        end
    elseif size(coords, 1) >= numel(labels)
        pos = coords(1:numel(labels), :);
    end
end

function labels = mne_info_channel_labels_local(info)
    if isfield(info, 'ch_names') && ~isempty(info.ch_names)
        labels = cellstr(string(info.ch_names(:)));
        return;
    end
    n = numel(info.chs);
    labels = cell(n, 1);
    for i = 1:n
        if isfield(info.chs(i), 'ch_name') && ~isempty(info.chs(i).ch_name)
            labels{i} = char(info.chs(i).ch_name);
        else
            labels{i} = sprintf('Ch%03d', i);
        end
    end
end

function keep = mne_eeg_channel_mask_local(info, n)
    keep = true(n, 1);
    if ~isfield(info, 'chs') || numel(info.chs) < n || exist('fiff_define_constants', 'file') ~= 2
        return;
    end
    FIFF = fiff_define_constants();
    if ~isfield(FIFF, 'FIFFV_EEG_CH'), return; end
    kinds = nan(n, 1);
    for i = 1:n
        if isfield(info.chs(i), 'kind') && ~isempty(info.chs(i).kind)
            kinds(i) = double(info.chs(i).kind);
        end
    end
    eeg = kinds == double(FIFF.FIFFV_EEG_CH);
    if any(eeg), keep = eeg; end
end

function pos = mne_info_positions_local(info, keep)
    labels_all = mne_info_channel_labels_local(info); %#ok<NASGU>
    idx = find(keep);
    pos = nan(numel(idx), 3);
    if ~isfield(info, 'chs'), return; end
    for ii = 1:numel(idx)
        i = idx(ii);
        if isfield(info.chs(i), 'loc') && numel(info.chs(i).loc) >= 3
            pos(ii,:) = double(info.chs(i).loc(1:3));
        end
    end
end

function pos = positions_from_chanlocs_local(chanlocs, n)
    pos = nan(n, 3);
    for i = 1:min(n, numel(chanlocs))
        if isfield(chanlocs(i), 'X') && isfield(chanlocs(i), 'Y') && isfield(chanlocs(i), 'Z')
            xyz = [double_or_nan(chanlocs(i).X), double_or_nan(chanlocs(i).Y), double_or_nan(chanlocs(i).Z)];
            if all(isfinite(xyz)), pos(i,:) = xyz; end
        end
    end
end

function [labels, pos] = apply_heppy_chanloc_sidecar(file_path, labels, pos)
    [side_labels, side_pos] = load_heppy_chanloc_sidecar(file_path);
    if isempty(side_labels) || isempty(side_pos) || sum(all(isfinite(side_pos), 2)) < 4
        return;
    end
    if isempty(labels)
        labels = side_labels;
        pos = side_pos;
        return;
    end
    side_can = canonical_labels_local(side_labels);
    label_can = canonical_labels_local(labels);
    mapped = nan(numel(labels), 3);
    for i = 1:numel(labels)
        j = find(strcmp(side_can, label_can{i}), 1, 'first');
        if ~isempty(j)
            mapped(i, :) = side_pos(j, :);
        elseif numel(side_labels) == numel(labels)
            mapped(i, :) = side_pos(i, :);
        end
    end
    if sum(all(isfinite(mapped), 2)) >= 4
        pos = mapped;
    end
end

function [labels, pos] = load_heppy_chanloc_sidecar(file_path)
    labels = {};
    pos = [];
    [folder, stem] = fileparts(file_path);
    sidecar = fullfile(folder, [stem '_chanlocs.mat']);
    if ~isfile(sidecar)
        return;
    end
    S = load(sidecar);
    if isfield(S, 'labels') && isfield(S, 'pos')
        labels = cellstr(string(S.labels(:)));
        pos = double(squeeze(S.pos));
        if size(pos, 2) ~= 3 && size(pos, 1) == 3
            pos = pos';
        end
        return;
    end
    if isfield(S, 'chanlocs') && ~isempty(S.chanlocs)
        labels = cellstr(string({S.chanlocs.labels}));
        pos = positions_from_chanlocs_local(S.chanlocs, numel(labels));
    end
end

function x = double_or_nan(v)
    if isempty(v), x = NaN; else, x = double(v(1)); end
end

function can = canonical_labels_local(labels)
    labels = cellstr(string(labels(:)));
    can = cell(size(labels));
    for i = 1:numel(labels)
        s = lower(strtrim(labels{i}));
        s = regexprep(s, '\s+', '');
        s = regexprep(s, '^eeg', '');
        can{i} = regexprep(s, '[^a-z0-9]', '');
    end
end

function C = normalize_rows_local(X)
    C = double(X);
    C = C - mean(C, 2, 'omitnan');
    denom = sqrt(sum(C.^2, 2));
    denom(denom <= eps | ~isfinite(denom)) = 1;
    C = bsxfun(@rdivide, C, denom);
end

function lengths = active_run_lengths_local(mask)
    mask = logical(mask(:));
    d = diff([false; mask; false]);
    starts = find(d == 1);
    stops = find(d == -1) - 1;
    lengths = stops - starts + 1;
end

function x = mean_or_nan_local(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v), x = NaN; else, x = mean(v); end
end

function x = median_or_nan_local(v)
    v = double(v(:));
    v = v(isfinite(v));
    if isempty(v), x = NaN; else, x = median(v); end
end

function T = vertcat_nonempty_local(rows)
    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        T = table();
    else
        T = vertcat(rows{:});
    end
end

function write_matrix_csv_local(path, X)
    ensure_dir_local(fileparts(path));
    writematrix(double(X), path);
end

function plot_topoplot_centers_local(Fit, chanlocs, out_png, title_text)
    ensure_dir_local(fileparts(out_png));
    centers = double(Fit.centers);
    K = size(centers, 1);
    labels = state_labels_from_fit(Fit, K);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 max(900, 180*K) 260]);
    for k = 1:K
        subplot(1, K, k);
        if exist('topoplot', 'file') == 2 && ~isempty(chanlocs)
            try
                topoplot(centers(k, :), chanlocs, 'electrodes', 'off');
            catch
                imagesc(centers(k, :)); axis tight off;
            end
        else
            imagesc(centers(k, :)); axis tight off;
        end
        title(labels{k}, 'Interpreter', 'none', 'Color', [0 0 0], 'FontWeight', 'bold');
    end
    if exist('sgtitle', 'file') == 2
        st = sgtitle(title_text, 'Interpreter', 'none');
        set(st, 'Color', [0 0 0], 'FontWeight', 'bold');
    elseif exist('suptitle', 'file') == 2
        st = suptitle(title_text);
        set(st, 'Color', [0 0 0], 'FontWeight', 'bold');
    end
    saveas(fig, out_png);
    close(fig);
end

function key = clean_key_local(x)
    key = char(string(x));
    key = regexprep(key, '[^A-Za-z0-9_.-]+', '_');
    if isempty(key), key = 'unknown'; end
end

function ensure_dir_local(pathstr)
    if isempty(pathstr), return; end
    if ~isfolder(pathstr), mkdir(pathstr); end
end
