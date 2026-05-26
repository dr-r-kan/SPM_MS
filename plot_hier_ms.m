function [plot_file, PlotInfo] = plot_hier_ms(hierarchical_mat, varargin)
% PLOT_HIERARCHICAL_MICROSTATE_TOPOGRAPHIES
%
% Create aligned topographic plots of hierarchical microstate maps.
%
% This is a plotting helper for the output of fit_microstate_hierarchical_dataset.m.
% By default it writes three aligned figures:
%   1) a single-row global microstate figure
%   2) a groupwise microstate figure
%   3) a group-vs-condition microstate figure
%
% Each row is a hierarchical node and each column is one microstate class.
% Lower-level maps are re-ordered and polarity-aligned to the global solution
% before plotting. Alignment always uses mean-centred unit-norm maps, while
% plotting can optionally use a display-only z-score scaling so the topoplots
% are easier to read.
%
% REQUIRED INPUT
%   hierarchical_mat : path to hierarchical_microstate_results.mat, or an
%                      already-loaded HResults structure.
%
% NAME-VALUE OPTIONS
%   'output_file'          output image path or base path. If empty, a default
%                          PNG base name is written beside hierarchical_mat.
%   'format'               'png', 'tiff', 'pdf', or 'fig'. Used only when
%                          output_file is empty. Default: 'png'.
%   'output_mode'          'split' [default] or 'combined'. 'split' writes
%                          separate global, groupwise, and group-vs-condition
%                          figures. 'combined' recreates the old combined grid.
%   'include_global'       include global row. Default: true.
%   'include_groups'       include group-level rows. Default: true.
%   'include_conditions'   include group x condition rows. Default: true.
%   'condition_mode'       'group_condition' or 'collapsed_condition'. Default:
%                          'group_condition'. The fitted hierarchy contains
%                          group x condition nodes, so this is the most direct
%                          and least assumptive mode. 'collapsed_condition'
%                          averages condition maps across groups after alignment.
%   'group_filter'         cellstr/string array of groups to include. Default: all.
%   'condition_filter'     cellstr/string array of conditions to include. Default: all.
%   'reference_eeg_file'   optional EEG .set file used only if HResults does not
%                          contain usable common_chanlocs.
%   'display_normalisation' 'zscore' [default], 'l2', or 'none'. This only
%                          affects plotting, not alignment or saved aligned maps.
%   'maplimits'            'global', 'row', 'global_percentile',
%                          'row_percentile', or numeric [lo hi]. Default:
%                          'global_percentile'.
%   'map_percentile'       percentile used by *_percentile scale modes.
%                          Default: 75.
%   'colormap_name'        MATLAB/EEGLAB colormap name. Default: 'jet'.
%   'electrodes'           topoplot electrode display option. Default: 'off'.
%   'numcontour'           topoplot contour count. Default: 6.
%   'visible'              show figure. Default: false.
%   'resolution'           export resolution in DPI. Default: 300.
%   'max_rows_per_figure'  split into multiple files if many rows. Default: Inf.
%   'save_aligned_maps'    save aligned templates and row metadata. Default: true.
%
% OUTPUTS
%   plot_file : struct of written plot file(s). In split mode the fields are
%               .global, .groupwise, and .group_vs_condition when present.
%               In combined mode the field is .combined.
%   PlotInfo  : struct containing row metadata and aligned template maps.
%
% EXAMPLE
%   plot_hier_ms( ...
%       'hierarchical_microstates/hierarchical_microstate_results.mat', ...
%       'output_file', 'hierarchical_microstates/hierarchical_topographies.png');
%
% NOTES
%   This function is only a visualisation/alignment helper.  It should not be
%   used as the inferential TANOVA step.  The inferential unit for TANOVA should
%   be participant-condition maps, not just these fitted group-level summaries.

    util = microstate_utilities();
    repo_cfg = util.load_config();
    plot_defaults = repo_cfg.plotting;

    p = inputParser;
    addRequired(p, 'hierarchical_mat', @(x) ischar(x) || isstring(x) || isstruct(x));
    addParameter(p, 'output_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'format', 'png', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_mode', 'split', @(x) ischar(x) || isstring(x));
    addParameter(p, 'include_global', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'include_groups', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'include_conditions', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'condition_mode', 'group_condition', @(x) ischar(x) || isstring(x));
    addParameter(p, 'group_filter', {}, @(x) iscell(x) || isstring(x) || ischar(x));
    addParameter(p, 'condition_filter', {}, @(x) iscell(x) || isstring(x) || ischar(x));
    addParameter(p, 'reference_eeg_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'display_normalisation', char(plot_defaults.display_normalisation), @(x) ischar(x) || isstring(x));
    addParameter(p, 'maplimits', char(plot_defaults.maplimits));
    addParameter(p, 'map_percentile', double(plot_defaults.map_percentile), @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 100);
    addParameter(p, 'colormap_name', char(plot_defaults.colormap_name), @(x) ischar(x) || isstring(x));
    addParameter(p, 'electrodes', char(plot_defaults.electrodes), @(x) ischar(x) || isstring(x));
    addParameter(p, 'numcontour', 6, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'visible', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'resolution', double(plot_defaults.resolution), @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'max_rows_per_figure', Inf, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'save_aligned_maps', true, @(x) islogical(x) && isscalar(x));
    parse(p, hierarchical_mat, varargin{:});
    cfg = p.Results;

    cfg.format = lower(char(cfg.format));
    cfg.output_mode = lower(char(cfg.output_mode));
    cfg.condition_mode = lower(char(cfg.condition_mode));
    cfg.display_normalisation = lower(char(cfg.display_normalisation));
    cfg.colormap_name = char(cfg.colormap_name);
    cfg.electrodes = char(cfg.electrodes);
    cfg.reference_eeg_file = char(cfg.reference_eeg_file);
    cfg.output_file = char(cfg.output_file);
    cfg.group_filter = normalise_filter_values(cfg.group_filter);
    cfg.condition_filter = normalise_filter_values(cfg.condition_filter);

    if ~ismember(cfg.format, {'png', 'tif', 'tiff', 'pdf', 'fig'})
        error('format must be png, tif, tiff, pdf, or fig.');
    end
    if ~ismember(cfg.output_mode, {'split', 'combined'})
        error('output_mode must be ''split'' or ''combined''.');
    end
    if ~ismember(cfg.condition_mode, {'group_condition', 'collapsed_condition'})
        error('condition_mode must be ''group_condition'' or ''collapsed_condition''.');
    end
    if ~ismember(cfg.display_normalisation, {'zscore', 'l2', 'none'})
        error('display_normalisation must be ''zscore'', ''l2'', or ''none''.');
    end
    if exist('topoplot', 'file') ~= 2
        error('EEGLAB topoplot is not on the MATLAB path. Start EEGLAB or add it to the path first.');
    end

    [H, base_dir, base_name] = load_hresults(hierarchical_mat);
    K = infer_selected_K(H);
    if K < 2
        error('Could not infer a valid selected K from HResults.');
    end

    [chanlocs, channel_labels] = get_common_chanlocs(H, cfg.reference_eeg_file);
    if numel(chanlocs) < 4
        error('Fewer than four channel locations are available for plotting.');
    end

    ref_centers = get_global_centers(H, K);
    [ref_centers, chanlocs, keep_ch] = restrict_maps_to_usable_channels(ref_centers, chanlocs);
    channel_labels = channel_labels(keep_ch);
    if size(ref_centers, 2) ~= numel(chanlocs)
        error('Global template width does not match usable channel-location count after filtering.');
    end
    ref_centers = normalise_maps_for_plot(ref_centers);

    rows = build_plot_rows(H, ref_centers, chanlocs, keep_ch, cfg, K);
    if isempty(rows)
        error('No hierarchical rows survived the requested filters.');
    end
    rows = add_display_maps(rows, cfg.display_normalisation);
    ref_display = apply_display_normalisation(ref_centers, cfg.display_normalisation);

    PlotInfo = struct();
    PlotInfo.K = K;
    PlotInfo.channel_labels = channel_labels(:)';
    PlotInfo.rows = rows;
    PlotInfo.reference_centers = ref_centers;
    PlotInfo.reference_display = ref_display;
    PlotInfo.global_template_alignment = get_global_template_alignment(H);
    PlotInfo.maplimits = cfg.maplimits;
    PlotInfo.map_percentile = cfg.map_percentile;
    PlotInfo.display_normalisation = cfg.display_normalisation;
    PlotInfo.output_mode = cfg.output_mode;
    PlotInfo.created = datestr(now, 30);

    if isempty(cfg.output_file)
        cfg.output_file = fullfile(base_dir, sprintf('%s_hierarchical_topographies.%s', base_name, cfg.format));
    end

    out_dir = fileparts(cfg.output_file);
    if isempty(out_dir)
        out_dir = pwd;
        cfg.output_file = fullfile(out_dir, cfg.output_file);
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    if cfg.save_aligned_maps
        save_aligned_outputs(cfg.output_file, PlotInfo);
    end

    plot_file = write_plot_files(rows, chanlocs, cfg, K, H);
    PlotInfo.output_files = plot_file;
end

function [H, base_dir, base_name] = load_hresults(input_arg)
    if isstruct(input_arg)
        H = input_arg;
        base_dir = pwd;
        base_name = 'HResults';
        return;
    end
    mat_file = char(input_arg);
    if ~isfile(mat_file)
        error('Hierarchical results file not found: %s', mat_file);
    end
    S = load(mat_file);
    if isfield(S, 'HResults')
        H = S.HResults;
    else
        names = fieldnames(S);
        hit = '';
        for i = 1:numel(names)
            if isstruct(S.(names{i})) && isfield(S.(names{i}), 'global') && isfield(S.(names{i}), 'selected_K')
                hit = names{i};
                break;
            end
        end
        if isempty(hit)
            error('No HResults-like structure found in %s.', mat_file);
        end
        H = S.(hit);
    end
    [base_dir, base_name, ~] = fileparts(mat_file);
    if isempty(base_dir)
        base_dir = pwd;
    end
end

function K = infer_selected_K(H)
    K = NaN;
    if isfield(H, 'selected_K') && isnumeric(H.selected_K) && isscalar(H.selected_K)
        K = H.selected_K;
    elseif isfield(H, 'global') && isfield(H.global, 'centers') && ~isempty(H.global.centers)
        K = size(H.global.centers, 1);
    elseif isfield(H, 'global') && isfield(H.global, 'K_estimated')
        K = H.global.K_estimated;
    end
    K = round(K);
end

function centers = get_global_centers(H, K)
    if ~isfield(H, 'global') || ~isfield(H.global, 'centers') || isempty(H.global.centers)
        error('HResults.global.centers is missing or empty.');
    end
    centers = double(H.global.centers);
    if size(centers, 1) ~= K
        error('Global centres have %d rows but selected K is %d.', size(centers, 1), K);
    end
end

function [chanlocs, labels] = get_common_chanlocs(H, reference_eeg_file)
    labels = {};
    if isfield(H, 'common_channel_labels') && ~isempty(H.common_channel_labels)
        labels = cellstr(H.common_channel_labels(:));
    end

    chanlocs = [];
    if isfield(H, 'common_chanlocs') && ~isempty(H.common_chanlocs)
        chanlocs = H.common_chanlocs;
    elseif isfield(H, 'common_pos') && ~isempty(H.common_pos)
        chanlocs = chanlocs_from_positions(H.common_pos, labels);
    elseif ~isempty(reference_eeg_file)
        chanlocs = load_reference_chanlocs(reference_eeg_file, labels);
    else
        error(['No common_chanlocs/common_pos found in HResults. Provide ', ...
               '''reference_eeg_file'', pointing to a .set file with matching channel labels.']);
    end

    chanlocs = chanlocs(:)';
    if isempty(labels)
        labels = cell(1, numel(chanlocs));
        for i = 1:numel(chanlocs)
            if isfield(chanlocs(i), 'labels') && ~isempty(chanlocs(i).labels)
                labels{i} = char(chanlocs(i).labels);
            else
                labels{i} = sprintf('Ch%d', i);
            end
        end
    end

    if numel(labels) ~= numel(chanlocs) && ~isempty(reference_eeg_file)
        chanlocs = load_reference_chanlocs(reference_eeg_file, labels);
    end
    if numel(labels) ~= numel(chanlocs)
        error('Number of channel labels (%d) does not match number of chanlocs (%d).', numel(labels), numel(chanlocs));
    end
end

function chanlocs = load_reference_chanlocs(reference_eeg_file, labels)
    reference_eeg_file = char(reference_eeg_file);
    if ~isfile(reference_eeg_file)
        error('reference_eeg_file not found: %s', reference_eeg_file);
    end
    if exist('pop_loadset', 'file') ~= 2
        error('pop_loadset is required to use reference_eeg_file, but EEGLAB is not on the path.');
    end
    EEG = pop_loadset('filename', reference_eeg_file);
    all_chanlocs = EEG.chanlocs;
    if isempty(labels)
        chanlocs = all_chanlocs;
        return;
    end
    all_labels = cellfun(@(x) lower(strtrim(char(x))), {all_chanlocs.labels}, 'UniformOutput', false);
    want = cellfun(@(x) lower(strtrim(char(x))), labels(:)', 'UniformOutput', false);
    keep = nan(1, numel(want));
    for i = 1:numel(want)
        hit = find(strcmp(want{i}, all_labels), 1, 'first');
        if isempty(hit)
            hit = find(strcmp(canonical_label(want{i}), cellfun(@canonical_label, all_labels, 'UniformOutput', false)), 1, 'first');
        end
        if isempty(hit)
            error('Could not match channel %s in reference EEG file.', labels{i});
        end
        keep(i) = hit;
    end
    chanlocs = all_chanlocs(keep);
end

function chanlocs = chanlocs_from_positions(pos, labels)
    pos = double(pos);
    if size(pos, 2) ~= 3 && size(pos, 1) == 3
        pos = pos';
    end
    if size(pos, 2) ~= 3
        error('common_pos must be n_channels x 3 or 3 x n_channels.');
    end
    n = size(pos, 1);
    if isempty(labels)
        labels = arrayfun(@(i) sprintf('Ch%d', i), 1:n, 'UniformOutput', false);
    end
    chanlocs = repmat(struct('labels', '', 'X', [], 'Y', [], 'Z', [], 'theta', [], 'radius', []), 1, n);
    for i = 1:n
        chanlocs(i).labels = char(labels{i});
        chanlocs(i).X = pos(i, 1);
        chanlocs(i).Y = pos(i, 2);
        chanlocs(i).Z = pos(i, 3);
        try
            [theta, radius] = cart_to_eeglab_polar(pos(i, :));
            chanlocs(i).theta = theta;
            chanlocs(i).radius = radius;
        catch
            chanlocs(i).theta = [];
            chanlocs(i).radius = [];
        end
    end
end

function [theta, radius] = cart_to_eeglab_polar(xyz)
    xyz = double(xyz(:)');
    xyz = xyz ./ (norm(xyz) + eps);
    x = xyz(1);
    y = xyz(2);
    z = xyz(3);
    theta = atan2d(y, x);
    radius = 0.5 * acos(max(min(z, 1), -1)) / (pi / 2);
    radius = min(max(radius, 0), 0.5);
end

function [maps_out, chanlocs_out, keep] = restrict_maps_to_usable_channels(maps, chanlocs)
    keep = has_usable_topoplot_location(chanlocs);
    maps_out = maps(:, keep);
    chanlocs_out = chanlocs(keep);
    finite_ch = all(isfinite(maps_out), 1);
    maps_out = maps_out(:, finite_ch);
    chanlocs_out = chanlocs_out(finite_ch);
    tmp = find(keep);
    keep2 = false(size(keep));
    keep2(tmp(finite_ch)) = true;
    keep = keep2;
    if numel(chanlocs_out) < 4
        error('Only %d usable topoplot channels remain.', numel(chanlocs_out));
    end
end

function rows = build_plot_rows(H, ref_centers, chanlocs, keep_ch, cfg, K)
    rows = empty_row_struct();
    rows(1) = [];

    if cfg.include_global
        row = empty_row_struct();
        row.level = 'global';
        row.label = 'Global';
        row.group = '';
        row.condition = '';
        row.name = 'global';
        row.inherited = false;
        row.n_maps = get_numeric_field(H.global, 'n_maps', NaN);
        row.centers_raw = double(H.global.centers);
        row.centers_aligned = ref_centers;
        row.assignment = 1:K;
        row.signs = ones(1, K);
        rows(end+1) = row; %#ok<AGROW>
    end

    if cfg.include_groups && isfield(H, 'groups') && ~isempty(H.groups)
        for i = 1:numel(H.groups)
            node = H.groups(i);
            if ~node_passes_filters(node, cfg, false)
                continue;
            end
            rows(end+1) = node_to_row(node, 'group', sprintf('Group: %s', char(node.group)), ref_centers, chanlocs, keep_ch, K); %#ok<AGROW>
        end
    end

    if cfg.include_conditions
        switch cfg.condition_mode
            case 'group_condition'
                if isfield(H, 'group_conditions') && ~isempty(H.group_conditions)
                    for i = 1:numel(H.group_conditions)
                        node = H.group_conditions(i);
                        if ~node_passes_filters(node, cfg, true)
                            continue;
                        end
                        label = sprintf('%s | %s', char(node.group), char(node.condition));
                        rows(end+1) = node_to_row(node, 'group_condition', label, ref_centers, chanlocs, keep_ch, K); %#ok<AGROW>
                    end
                end
            case 'collapsed_condition'
                if ~isfield(H, 'group_conditions') || isempty(H.group_conditions)
                    warning('No group_conditions found. Cannot build collapsed condition rows.');
                else
                    rows = append_collapsed_condition_rows(rows, H.group_conditions, ref_centers, chanlocs, keep_ch, cfg, K);
                end
        end
    end
end

function tf = node_passes_filters(node, cfg, require_condition)
    tf = true;
    if ~isempty(cfg.group_filter)
        tf = tf && isfield(node, 'group') && any(strcmp(char(node.group), cfg.group_filter));
    end
    if require_condition && ~isempty(cfg.condition_filter)
        tf = tf && isfield(node, 'condition') && any(strcmp(char(node.condition), cfg.condition_filter));
    end
end

function row = node_to_row(node, level, label, ref_centers, chanlocs, keep_ch, K)
    if ~isfield(node, 'centers') || isempty(node.centers)
        error('Node %s has no centres to plot.', char(node.name));
    end
    centers = double(node.centers);
    if size(centers, 2) == numel(keep_ch)
        centers = centers(:, keep_ch);
    end
    if size(centers, 1) ~= K
        error('Node %s has %d centres, but K=%d.', char(node.name), size(centers, 1), K);
    end
    if size(centers, 2) ~= numel(chanlocs)
        error('Node %s has %d channels, but the plotting chanlocs have %d.', char(node.name), size(centers, 2), numel(chanlocs));
    end
    [aligned, assignment, signs] = align_centers_to_reference_for_plot(centers, ref_centers);
    row = empty_row_struct();
    row.level = level;
    row.label = char(label);
    row.group = char(get_char_field(node, 'group', ''));
    row.condition = char(get_char_field(node, 'condition', ''));
    row.name = char(get_char_field(node, 'name', label));
    row.inherited = logical(get_numeric_field(node, 'inherited', false));
    row.n_maps = get_numeric_field(node, 'n_maps', NaN);
    row.centers_raw = centers;
    row.centers_aligned = aligned;
    row.assignment = assignment;
    row.signs = signs;
end

function rows = append_collapsed_condition_rows(rows, nodes, ref_centers, chanlocs, keep_ch, cfg, K)
    conds = unique(cellfun(@(x) char(x), {nodes.condition}, 'UniformOutput', false), 'stable');
    for ci = 1:numel(conds)
        cond = conds{ci};
        if ~isempty(cfg.condition_filter) && ~any(strcmp(cond, cfg.condition_filter))
            continue;
        end
        idx = find(strcmp(cellfun(@(x) char(x), {nodes.condition}, 'UniformOutput', false), cond));
        kept = [];
        aligned_stack = [];
        n_maps_total = 0;
        for ii = idx(:)'
            node = nodes(ii);
            if ~node_passes_filters(node, cfg, false)
                continue;
            end
            centers = double(node.centers);
            if size(centers, 2) == numel(keep_ch)
                centers = centers(:, keep_ch);
            end
            if isempty(centers) || size(centers, 1) ~= K || size(centers, 2) ~= numel(chanlocs)
                continue;
            end
            aligned = align_centers_to_reference_for_plot(centers, ref_centers);
            aligned_stack(:, :, end+1) = aligned; %#ok<AGROW>
            kept(end+1) = ii; %#ok<AGROW>
            n_maps_total = n_maps_total + get_numeric_field(node, 'n_maps', 0);
        end
        if isempty(kept)
            continue;
        end
        avg = mean(aligned_stack, 3, 'omitnan');
        avg = normalise_maps_for_plot(avg);
        [avg, assignment, signs] = align_centers_to_reference_for_plot(avg, ref_centers);
        row = empty_row_struct();
        row.level = 'condition_collapsed';
        row.label = sprintf('Condition: %s', cond);
        row.group = '';
        row.condition = cond;
        row.name = sprintf('condition:%s', cond);
        row.inherited = false;
        row.n_maps = n_maps_total;
        row.centers_raw = avg;
        row.centers_aligned = avg;
        row.assignment = assignment;
        row.signs = signs;
        rows(end+1) = row; %#ok<AGROW>
    end
end

function plot_file = write_plot_files(rows, chanlocs, cfg, K, H)
    specs = build_output_specs(rows, cfg);
    plot_file = struct();
    for i = 1:numel(specs)
        plot_file.(specs(i).key) = write_output_group(specs(i), chanlocs, cfg, K, H);
    end
end

function specs = build_output_specs(rows, cfg)
    specs = struct('key', {}, 'title_text', {}, 'rows', {}, 'output_file', {});
    if strcmp(cfg.output_mode, 'combined')
        specs(1).key = 'combined';
        specs(1).title_text = 'Hierarchical microstate topographies';
        specs(1).rows = rows;
        specs(1).output_file = cfg.output_file;
        return;
    end

    levels = {rows.level};
    global_rows = rows(strcmp(levels, 'global'));
    if ~isempty(global_rows)
        specs(end+1) = make_output_spec('global', 'Global microstates', global_rows, cfg.output_file, 'global'); %#ok<AGROW>
    end

    group_rows = rows(strcmp(levels, 'group'));
    if ~isempty(group_rows)
        specs(end+1) = make_output_spec('groupwise', 'Groupwise microstates', group_rows, cfg.output_file, 'groupwise'); %#ok<AGROW>
    end

    cond_mask = ismember(levels, {'group_condition', 'condition_collapsed'});
    condition_rows = rows(cond_mask);
    if ~isempty(condition_rows)
        title_text = 'Group vs condition microstates';
        if strcmp(cfg.condition_mode, 'collapsed_condition')
            title_text = 'Condition microstates';
        end
        specs(end+1) = make_output_spec('group_vs_condition', title_text, condition_rows, cfg.output_file, 'group_vs_condition'); %#ok<AGROW>
    end
end

function spec = make_output_spec(key, title_text, rows, base_output_file, suffix)
    spec = struct();
    spec.key = key;
    spec.title_text = title_text;
    spec.rows = rows;
    spec.output_file = append_output_suffix(base_output_file, suffix);
end

function out_file = append_output_suffix(base_output_file, suffix)
    [out_dir, base_name, ext] = fileparts(base_output_file);
    out_file = fullfile(out_dir, sprintf('%s_%s%s', base_name, suffix, ext));
end

function files_out = write_output_group(spec, chanlocs, cfg, K, H)
    rows = spec.rows;
    n_rows_total = numel(rows);
    chunk_size = min(n_rows_total, max(1, floor(cfg.max_rows_per_figure)));
    n_chunks = ceil(n_rows_total / chunk_size);
    files_cell = cell(n_chunks, 1);

    for ch = 1:n_chunks
        lo = (ch - 1) * chunk_size + 1;
        hi = min(n_rows_total, ch * chunk_size);
        rows_this = rows(lo:hi);
        out_file = spec.output_file;
        if n_chunks > 1
            [d, b, e] = fileparts(spec.output_file);
            out_file = fullfile(d, sprintf('%s_part%02d%s', b, ch, e));
        end
        files_cell{ch} = out_file;
        make_one_figure(rows_this, chanlocs, cfg, K, H, out_file, spec.title_text, ch, n_chunks, lo, hi, n_rows_total);
    end

    if n_chunks == 1
        files_out = files_cell{1};
    else
        files_out = files_cell;
    end
end

function make_one_figure(rows, chanlocs, cfg, K, H, out_file, figure_title, chunk_idx, n_chunks, row_lo, row_hi, n_rows_total)
    n_rows = numel(rows);
    visible_state = ternary(cfg.visible, 'on', 'off');
    fig_w = max(900, 210 * K + 360);
    fig_h = max(280, 185 * n_rows + 140);
    fig = figure('Name', figure_title, ...
        'NumberTitle', 'off', 'Color', 'white', 'Visible', visible_state, ...
        'Position', [100, 100, fig_w, fig_h]);

    tl = tiledlayout(n_rows, K, 'TileSpacing', 'compact', 'Padding', 'compact');
    state_titles = infer_state_titles(H, K);

    maps_all = vertcat(rows.centers_display);
    limits_global = compute_maplimits(maps_all, cfg.maplimits, cfg.map_percentile);

    for r = 1:n_rows
        maps = rows(r).centers_display;
        maplimits = resolve_maplimits(maps, limits_global, cfg);

        for k = 1:K
            ax = nexttile(tl, (r - 1) * K + k);
            topoplot(maps(k, :), chanlocs, ...
                'electrodes', cfg.electrodes, ...
                'numcontour', cfg.numcontour, ...
                'maplimits', maplimits);
            colormap(ax, cfg.colormap_name);
            if r == 1
                title(state_titles{k}, 'FontWeight', 'bold', 'Interpreter', 'none');
            end
            if k == 1
                ylabel(row_ylabel(rows(r)), 'FontWeight', 'bold', 'Interpreter', 'none');
            end
            if k == K
                cb = colorbar;
                cb.FontSize = 7;
                cb.Label.String = display_scale_label(cfg.display_normalisation);
            end
        end
    end

    title_text = sprintf('%s | K=%d | scale=%s', figure_title, K, describe_scale(cfg));
    if isfield(H, 'created')
        title_text = sprintf('%s | fit created %s', title_text, char(H.created));
    end
    if n_chunks > 1
        title_text = sprintf('%s | rows %d-%d/%d | part %d/%d', title_text, row_lo, row_hi, n_rows_total, chunk_idx, n_chunks);
    end
    title(tl, title_text, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');

    drawnow;
    save_figure(fig, out_file, cfg);
    if ~cfg.visible
        close(fig);
    end
end

function txt = row_ylabel(row)
    txt = row.label;
    if isfinite(row.n_maps)
        txt = sprintf('%s\nn=%g', txt, row.n_maps);
    end
    if row.inherited
        txt = sprintf('%s\ninherited', txt);
    end
end

function save_figure(fig, out_file, cfg)
    [out_dir, ~, ext] = fileparts(out_file);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    ext = lower(strrep(ext, '.', ''));
    if isempty(ext)
        ext = cfg.format;
        out_file = sprintf('%s.%s', out_file, ext);
    end
    switch ext
        case {'png', 'tif', 'tiff', 'pdf'}
            exportgraphics(fig, out_file, 'Resolution', cfg.resolution);
        case 'fig'
            savefig(fig, out_file);
        otherwise
            error('Unsupported output extension: %s', ext);
    end
end

function maplimits = resolve_maplimits(maps, limits_global, cfg)
    if isnumeric(cfg.maplimits)
        maplimits = compute_maplimits(maps, cfg.maplimits, cfg.map_percentile);
        return;
    end

    mode = lower(char(cfg.maplimits));
    switch mode
        case {'row', 'row_percentile'}
            maplimits = compute_maplimits(maps, mode, cfg.map_percentile);
        case {'global', 'global_percentile', 'percentile', 'robust_symmetric'}
            maplimits = limits_global;
        otherwise
            error('Unsupported maplimits mode: %s', mode);
    end
end

function txt = describe_scale(cfg)
    if isnumeric(cfg.maplimits)
        txt = sprintf('[%.3g %.3g]', cfg.maplimits(1), cfg.maplimits(2));
        return;
    end

    mode = lower(char(cfg.maplimits));
    switch mode
        case {'global_percentile', 'row_percentile', 'percentile', 'robust_symmetric'}
            txt = sprintf('%s p%d', mode, round(cfg.map_percentile));
        otherwise
            txt = mode;
    end
end

function txt = display_scale_label(display_normalisation)
    switch lower(char(display_normalisation))
        case 'zscore'
            txt = 'Topography z-score';
        case 'l2'
            txt = 'Unit-norm value';
        otherwise
            txt = 'Map value';
    end
end

function save_aligned_outputs(output_file, PlotInfo)
    [out_dir, base, ~] = fileparts(output_file);
    if isempty(out_dir)
        out_dir = pwd;
    end
    mat_file = fullfile(out_dir, [base '_aligned_maps.mat']);
    save(mat_file, 'PlotInfo', '-v7.3');

    rows = PlotInfo.rows;
    n = numel(rows);
    level = cell(n, 1);
    label = cell(n, 1);
    name = cell(n, 1);
    group = cell(n, 1);
    condition = cell(n, 1);
    n_maps = nan(n, 1);
    inherited = false(n, 1);
    for i = 1:n
        level{i} = rows(i).level;
        label{i} = rows(i).label;
        name{i} = rows(i).name;
        group{i} = rows(i).group;
        condition{i} = rows(i).condition;
        n_maps(i) = rows(i).n_maps;
        inherited(i) = rows(i).inherited;
    end
    T = table(level, label, name, group, condition, n_maps, inherited);
    writetable(T, fullfile(out_dir, [base '_row_manifest.csv']));

    for i = 1:n
        safe = regexprep(rows(i).name, '[^A-Za-z0-9_\-]+', '_');
        csv_file = fullfile(out_dir, sprintf('%s_%03d_%s_aligned_templates.csv', base, i, safe));
        writematrix(rows(i).centers_aligned, csv_file);
    end
end

function [aligned, assignment, signs] = align_centers_to_reference_for_plot(centers, reference)
    centers = normalise_maps_for_plot(centers);
    reference = normalise_maps_for_plot(reference);
    K = size(reference, 1);
    aligned = zeros(size(reference));
    assignment = nan(1, K);
    signs = ones(1, K);
    S = abs(reference * centers');
    used = false(1, size(centers, 1));
    for k = 1:K
        vals = S(k, :);
        vals(used) = -Inf;
        [~, j] = max(vals);
        if isempty(j) || ~isfinite(vals(j))
            j = find(~used, 1, 'first');
        end
        if isempty(j)
            error('Could not align centres to reference.');
        end
        used(j) = true;
        c = centers(j, :);
        if c * reference(k, :)' < 0
            c = -c;
            signs(k) = -1;
        else
            signs(k) = 1;
        end
        aligned(k, :) = c;
        assignment(k) = j;
    end
    aligned = normalise_maps_for_plot(aligned);
end

function rows = add_display_maps(rows, display_normalisation)
    for i = 1:numel(rows)
        rows(i).centers_display = apply_display_normalisation(rows(i).centers_aligned, display_normalisation);
    end
end

function maps = apply_display_normalisation(maps, mode)
    mode = lower(char(mode));
    switch mode
        case 'zscore'
            maps = spatial_zscore_maps(maps);
        case 'l2'
            maps = normalise_maps_for_plot(maps);
        case 'none'
            maps = double(maps);
        otherwise
            error('Unsupported display normalisation: %s', mode);
    end
end

function maps = normalise_maps_for_plot(maps)
    maps = double(maps);
    for i = 1:size(maps, 1)
        x = maps(i, :);
        finite_mask = isfinite(x);
        if ~any(finite_mask)
            maps(i, :) = zeros(size(x));
            continue;
        end
        x = x - mean(x(finite_mask));
        denom = sqrt(sum(x(finite_mask) .^ 2)) + eps;
        maps(i, :) = x ./ denom;
    end
end

function maps = spatial_zscore_maps(maps)
    maps = double(maps);
    for i = 1:size(maps, 1)
        x = maps(i, :);
        finite_mask = isfinite(x);
        if ~any(finite_mask)
            maps(i, :) = zeros(size(x));
            continue;
        end
        x = x - mean(x(finite_mask));
        denom = std(x(finite_mask));
        if ~isfinite(denom) || denom <= eps
            denom = 1;
        end
        maps(i, :) = x ./ denom;
    end
end

function maplimits = compute_maplimits(maps, requested, percentile_value)
    if isnumeric(requested)
        if ~isequal(size(requested), [1 2]) && ~isequal(size(requested), [2 1])
            error('Numeric maplimits must be [lo hi].');
        end
        maplimits = double(requested(:)');
        return;
    end
    mode = lower(char(requested));
    if ~ismember(mode, {'global', 'row', 'global_percentile', 'row_percentile', 'percentile', 'robust_symmetric'})
        error(['maplimits must be ''global'', ''row'', ''global_percentile'', ', ...
               '''row_percentile'', or numeric [lo hi].']);
    end
    abs_vals = abs(double(maps(:)));
    abs_vals = abs_vals(isfinite(abs_vals));
    if isempty(abs_vals)
        clim = 1;
        maplimits = [-clim clim];
        return;
    end

    switch mode
        case {'global_percentile', 'row_percentile', 'percentile', 'robust_symmetric'}
            clim = local_percentile(abs_vals, percentile_value);
        otherwise
            clim = max(abs_vals);
    end
    if ~isfinite(clim) || clim <= eps
        clim = 1;
    end
    maplimits = [-clim clim];
end

function value = local_percentile(x, pct)
    x = sort(double(x(:)));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
        return;
    end
    if numel(x) == 1
        value = x;
        return;
    end
    pct = min(max(double(pct), 0), 100);
    pos = 1 + (numel(x) - 1) * (pct / 100);
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        value = x(lo);
    else
        w = pos - lo;
        value = (1 - w) * x(lo) + w * x(hi);
    end
end

function valid = has_usable_topoplot_location(chanlocs)
    valid = false(1, numel(chanlocs));
    for i = 1:numel(chanlocs)
        has_polar = isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && ...
            isfield(chanlocs(i), 'radius') && ~isempty(chanlocs(i).radius) && ...
            isnumeric(chanlocs(i).theta) && isnumeric(chanlocs(i).radius) && ...
            isfinite(double(chanlocs(i).theta)) && isfinite(double(chanlocs(i).radius)) && ...
            double(chanlocs(i).radius) >= 0 && double(chanlocs(i).radius) <= 0.5;
        has_xyz = isfield(chanlocs(i), 'X') && ~isempty(chanlocs(i).X) && ...
            isfield(chanlocs(i), 'Y') && ~isempty(chanlocs(i).Y) && ...
            isfield(chanlocs(i), 'Z') && ~isempty(chanlocs(i).Z) && ...
            all(isfinite(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]))) && ...
            norm(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z])) > eps;
        valid(i) = has_polar || has_xyz;
    end
end

function row = empty_row_struct()
    row = struct('level', '', 'label', '', 'group', '', 'condition', '', 'name', '', ...
        'inherited', false, 'n_maps', NaN, 'centers_raw', [], 'centers_aligned', [], ...
        'centers_display', [], 'assignment', [], 'signs', []);
end

function vals = normalise_filter_values(vals)
    if isempty(vals)
        vals = {};
    elseif ischar(vals) || isstring(vals)
        vals = cellstr(vals);
    elseif iscell(vals)
        vals = cellfun(@char, vals, 'UniformOutput', false);
    else
        error('Filter values must be empty, char, string, string array, or cellstr.');
    end
    vals = vals(:)';
end

function value = get_numeric_field(S, field_name, default_value)
    value = default_value;
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    end
end

function value = get_char_field(S, field_name, default_value)
    value = default_value;
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    end
end

function ta = get_global_template_alignment(H)
    ta = struct();
    if isstruct(H) && isfield(H, 'global') && isfield(H.global, 'template_alignment') && ~isempty(H.global.template_alignment)
        ta = H.global.template_alignment;
    elseif isstruct(H) && isfield(H, 'canonical_template_alignment') && ~isempty(H.canonical_template_alignment)
        ta = H.canonical_template_alignment;
    end
end

function titles = infer_state_titles(H, K)
    titles = arrayfun(@(k) sprintf('MS %d', k), 1:K, 'UniformOutput', false);
    ta = get_global_template_alignment(H);
    if ~isstruct(ta) || ~isfield(ta, 'labels') || numel(ta.labels) < K
        return;
    end

    has_corr = isfield(ta, 'correlations') && numel(ta.correlations) >= K;
    for k = 1:K
        label = char(ta.labels{k});
        if isempty(label)
            continue;
        end
        if has_corr && isfinite(ta.correlations(k))
            titles{k} = sprintf('MS %d (%s r=%.2f)', k, label, ta.correlations(k));
        else
            titles{k} = sprintf('MS %d (%s)', k, label);
        end
    end
end

function c = canonical_label(label)
    c = lower(regexprep(strtrim(char(label)), '[^a-zA-Z0-9]+', ''));
end

function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end
