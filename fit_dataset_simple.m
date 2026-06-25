function [Results, results_mat] = fit_dataset_simple(eeg_list, varargin)
%FIT_DATASET_SIMPLE Global SPM-VB microstates plus per-file backfit sequences.
%
% Usage:
%   R = fit_dataset_simple({'/path/a.set', '/path/b.set'});
%   R = fit_dataset_simple('/path/manifest.csv', 'criterion', 'free_energy_covariance');
%   R = fit_dataset_simple('LEMON');

    if nargin < 1 || isempty(eeg_list)
        eeg_list = 'LEMON';
    end

    util = microstate_utilities();
    repo_cfg = util.load_config();
    hcfg = repo_cfg.hierarchical;
    pcfg = repo_cfg.preprocessing;

    p = inputParser;
    addRequired(p, 'eeg_list');
    addParameter(p, 'output_dir', fullfile('outputs', 'simple_microstates'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'criterion', 'free_energy_covariance', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', double(hcfg.K_candidates(:)'), @isnumeric);
    addParameter(p, 'max_maps_per_file', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_global_maps', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'filter_band', double(pcfg.filter_band(:)'), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'apply_average_reference', logical(pcfg.apply_average_reference), @islogical);
    addParameter(p, 'spatial_filter', char(pcfg.spatial_filter), @(x) ischar(x) || isstring(x));
    addParameter(p, 'localizer_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'save_plots', true, @islogical);
    addParameter(p, 'verbose', true, @islogical);
    parse(p, eeg_list, varargin{:});
    cfg = p.Results;

    cfg.output_dir = util.resolve_path(cfg.output_dir, util.project_root());
    ensure_dir(cfg.output_dir);

    manifest_path = resolve_input_manifest(eeg_list, cfg.output_dir);
    [H, hresults_mat] = fit_microstate_hierarchical_dataset(manifest_path, ...
        'output_dir', cfg.output_dir, ...
        'criterion', cfg.criterion, ...
        'K_candidates', cfg.K_candidates, ...
        'max_maps_per_file', cfg.max_maps_per_file, ...
        'max_global_maps', cfg.max_global_maps, ...
        'filter_band', cfg.filter_band, ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'spatial_filter', cfg.spatial_filter, ...
        'localizer_dir', cfg.localizer_dir, ...
        'global_only', true, ...
        'run_backfit', false, ...
        'save_plots', cfg.save_plots, ...
        'verbose', cfg.verbose);

    Fit = H.global.fit;
    print_centers(Fit.centers, H.common_labels);
    write_center_table(fullfile(cfg.output_dir, 'global_microstate_centers.csv'), Fit.centers, H.common_labels);

    backfit_dir = fullfile(cfg.output_dir, 'backfit_sequences');
    ensure_dir(backfit_dir);
    sequence_files = strings(height(H.manifest), 1);
    backfit_files = strings(height(H.manifest), 1);
    backfit_summary = table();
    for i = 1:height(H.manifest)
        row = H.manifest(i, :);
        if cfg.verbose
            fprintf('Backfitting [%d/%d] %s %s\n', i, height(H.manifest), row.participant, row.condition);
        end
        [Sim, kept_samples] = load_common_record(row.file_path, H, util);
        backfit = backfit_microstate_timecourse(Sim, Fit);
        if ~backfit.ok
            error('Backfit failed for %s: %s', row.file_path, backfit.message);
        end

        labels = state_labels(Fit);
        seq = backfit_sequence_table(backfit, labels, Sim.sfreq, kept_samples);
        stem = sprintf('%03d_%s_%s', i, clean_key(row.participant), clean_key(row.condition));
        sequence_files(i) = string(fullfile(backfit_dir, [stem '_sequence.csv']));
        backfit_files(i) = string(fullfile(backfit_dir, [stem '_backfit.mat']));
        writetable(seq, sequence_files(i));
        save(backfit_files(i), 'backfit', 'seq', 'row', 'Sim', '-v7.3');

        backfit_summary = [backfit_summary; make_backfit_summary_row(row, sequence_files(i), backfit_files(i), backfit)]; %#ok<AGROW>
    end
    writetable(backfit_summary, fullfile(cfg.output_dir, 'backfit_sequence_summary.csv'));

    Results = struct();
    Results.source = 'fit_dataset_simple';
    Results.created = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
    Results.cfg = cfg;
    Results.hierarchical_results_mat = hresults_mat;
    Results.manifest = H.manifest;
    Results.common_labels = H.common_labels;
    Results.common_chanlocs = H.common_chanlocs;
    Results.common_pos = H.common_pos;
    Results.global_fit = Fit;
    Results.centers = Fit.centers;
    Results.K_estimated = Fit.K_estimated;
    Results.backfit_summary = backfit_summary;
    Results.sequence_files = sequence_files;
    Results.backfit_files = backfit_files;

    results_mat = fullfile(cfg.output_dir, 'simple_dataset_results.mat');
    save(results_mat, 'Results', '-v7.3');
    if cfg.verbose
        fprintf('\nSimple dataset fit complete.\nResults: %s\nSequences: %s\n', results_mat, backfit_dir);
    end
end

function manifest_path = resolve_input_manifest(eeg_list, output_dir)
    if isstring(eeg_list) && numel(eeg_list) > 1
        manifest_path = write_list_manifest(cellstr(eeg_list(:)), output_dir);
        return;
    end
    if iscell(eeg_list)
        manifest_path = write_list_manifest(cellstr(string(eeg_list(:))), output_dir);
        return;
    end

    manifest_path = char(string(eeg_list));
    if strcmpi(strtrim(manifest_path), 'LEMON')
        manifest_path = fullfile(getenv('HOME'), 'EEG', 'LEMON');
    end
end

function manifest_path = write_list_manifest(files, output_dir)
    files = files(:);
    participant = strings(numel(files), 1);
    condition = strings(numel(files), 1);
    group = repmat("all", numel(files), 1);
    file_path = strings(numel(files), 1);
    for i = 1:numel(files)
        file_path(i) = string(abs_path(files{i}));
        participant(i) = infer_participant(file_path(i), i);
        condition(i) = infer_condition(file_path(i));
    end
    T = table(participant, condition, group, file_path, ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path'});
    manifest_path = fullfile(output_dir, 'simple_input_manifest.csv');
    writetable(T, manifest_path);
end

function out = abs_path(path_in)
    out = char(string(path_in));
    if startsWith(out, filesep) || ~isempty(regexp(out, '^[A-Za-z]:[\\/]', 'once'))
        return;
    end
    out = fullfile(pwd, out);
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

function [Sim, kept_samples] = load_common_record(file_path, H, util)
    [data, sfreq, ~, labels, pos] = load_eeg_file_simple(file_path, H.cfg, util);
    common_labels = cellstr(string(H.common_labels(:)));
    common_can = util.canonical_channel_labels(common_labels);
    can = util.canonical_channel_labels(labels);
    X = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            X(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        X = fill_missing_channels_simple(X, missing, pos, idx, H.common_pos, file_path);
    end
    good_t = all(isfinite(X), 1);
    X = X(:, good_t);
    kept_samples = find(good_t)';

    Sim = struct();
    Sim.X_noisy = X;
    Sim.sfreq = sfreq;
    Sim.channel_labels = common_labels(:)';
    Sim.chanlocs = H.common_chanlocs;
    Sim.pos = H.common_pos;
    Sim.preprocessing = struct( ...
        'apply_average_reference', H.cfg.apply_average_reference, ...
        'filter_band', H.cfg.filter_band, ...
        'gfp_peak_min_distance', H.cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', H.cfg.gfp_peak_threshold_schedule);
end

function [data, sfreq, chanlocs, labels, pos] = load_eeg_file_simple(file_path, cfg, util)
    file_path = char(file_path);
    [~, ~, ext] = fileparts(file_path);
    chanlocs = [];
    sfreq = 250;
    switch lower(ext)
        case '.set'
            EEG = pop_loadset(file_path);
            data = double(EEG.data);
            sfreq = double(EEG.srate);
            chanlocs = EEG.chanlocs;
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
        otherwise
            error('Unsupported EEG extension: %s', ext);
    end
    data = squeeze(data);
    if size(data, 1) > size(data, 2) && size(data, 2) < 128
        data = data';
    end
    chanlocs = maybe_apply_localizer(chanlocs, file_path, cfg, util);
    labels = util.channel_labels_from_chanlocs(chanlocs, size(data, 1));
    pos = util.positions_from_chanlocs(chanlocs, size(data, 1));
end

function chanlocs = maybe_apply_localizer(chanlocs, file_path, cfg, util)
    if isempty(chanlocs) || ~isfield(cfg, 'localizer_dir') || isempty(cfg.localizer_dir)
        return;
    end
    participant = regexp(char(file_path), 'sub-[A-Za-z0-9]+', 'match', 'once');
    if isempty(participant)
        return;
    end
    loc_file = fullfile(char(cfg.localizer_dir), participant, [participant '.mat']);
    if ~isfile(loc_file)
        return;
    end
    S = load(loc_file, 'Channel');
    if ~isfield(S, 'Channel') || isempty(S.Channel)
        return;
    end
    loc_labels = cell(1, numel(S.Channel));
    loc_xyz = nan(numel(S.Channel), 3);
    for i = 1:numel(S.Channel)
        parts = regexp(char(S.Channel(i).Name), '_', 'split');
        loc_labels{i} = parts{end};
        if isfield(S.Channel(i), 'Loc') && ~isempty(S.Channel(i).Loc)
            xyz = double(S.Channel(i).Loc);
            loc_xyz(i, :) = xyz(1:3, 1)';
        end
    end
    loc_can = util.canonical_channel_labels(loc_labels);
    labels = util.channel_labels_from_chanlocs(chanlocs, numel(chanlocs));
    can = util.canonical_channel_labels(labels);
    for i = 1:numel(chanlocs)
        hit = find(strcmp(loc_can, can{i}), 1, 'first');
        if isempty(hit) || any(~isfinite(loc_xyz(hit, :)))
            continue;
        end
        chanlocs(i).X = loc_xyz(hit, 1);
        chanlocs(i).Y = loc_xyz(hit, 2);
        chanlocs(i).Z = loc_xyz(hit, 3);
    end
end

function data = fill_missing_channels_simple(data, missing, source_pos, direct_idx, target_pos, file_path)
    if isempty(source_pos) || isempty(target_pos)
        error('Cannot interpolate missing channels in %s without channel positions.', file_path);
    end
    available = find(isfinite(direct_idx));
    source_xyz = source_pos(direct_idx(available), :);
    valid = all(isfinite(source_xyz), 2);
    available = available(valid);
    source_xyz = source_xyz(valid, :);
    if numel(available) < 4
        error('Cannot interpolate missing channels in %s with fewer than 4 positioned source channels.', file_path);
    end
    for i = 1:numel(missing)
        t = missing(i);
        d = sqrt(sum((source_xyz - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        ord = ord(1:min(6, numel(ord)));
        ds = ds(1:numel(ord));
        if ds(1) <= eps
            w = zeros(numel(ord), 1);
            w(1) = 1;
        else
            w = 1 ./ (ds.^2 + eps);
            w = w ./ sum(w);
        end
        data(t, :) = w' * data(available(ord), :);
    end
end

function labels = state_labels(Fit)
    K = size(Fit.centers, 1);
    labels = arrayfun(@(k) sprintf('state_%02d', k), 1:K, 'UniformOutput', false);
    if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels') && numel(Fit.template_alignment.labels) >= K
        labels = cellstr(string(Fit.template_alignment.labels(:)));
    end
end

function T = backfit_sequence_table(backfit, labels, sfreq, kept_samples)
    n = numel(backfit.hard.assignments);
    hard_idx = double(backfit.hard.assignments(:));
    T = table((1:n)', kept_samples(:), ((0:n-1)' ./ sfreq), hard_idx, ...
        string(labels(hard_idx(:))), double(backfit.hard.confidence(:)), ...
        'VariableNames', {'sample', 'original_sample', 'time_s', 'hard_state_index', 'hard_state_label', 'hard_confidence'});
    if isfield(backfit, 'mixture') && backfit.mixture.available
        mix_idx = double(backfit.mixture.assignments(:));
        T.mixture_state_index = mix_idx;
        T.mixture_state_label = string(labels(mix_idx(:)));
        T.mixture_confidence = double(backfit.mixture.confidence(:));
    end
end

function row = make_backfit_summary_row(manifest_row, seq_file, mat_file, backfit)
    row = table(string(manifest_row.participant), string(manifest_row.condition), ...
        string(manifest_row.group), string(manifest_row.file_path), double(backfit.n_samples), ...
        logical(isfield(backfit, 'mixture') && backfit.mixture.available), string(seq_file), string(mat_file), ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path', 'n_samples', ...
        'mixture_available', 'sequence_csv', 'backfit_mat'});
end

function print_centers(centers, labels)
    fprintf('\nGlobal microstate cluster centres (%d states x %d channels):\n', size(centers, 1), size(centers, 2));
    fprintf('Channels: %s\n', strjoin(cellstr(string(labels(:)')), ', '));
    disp(centers);
end

function write_center_table(file_path, centers, labels)
    T = array2table(double(centers));
    T.Properties.VariableNames = matlab.lang.makeValidName(cellstr(string(labels(:)')));
    T.state = (1:size(centers, 1))';
    T = movevars(T, 'state', 'Before', 1);
    writetable(T, file_path);
end

function ensure_dir(pth)
    if ~exist(pth, 'dir')
        mkdir(pth);
    end
end

function key = clean_key(value)
    key = char(string(value));
    key = regexprep(key, '[^A-Za-z0-9_-]', '_');
    if isempty(key)
        key = 'empty';
    end
end
