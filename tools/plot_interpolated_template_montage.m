function [plot_file, PlotInfo] = plot_interpolated_template_montage(input_arg, varargin)
% PLOT_INTERPOLATED_TEMPLATE_MONTAGE
%
% Plot the montage-specific interpolated MetaMaps template that was saved by
% fit_microstate_hierarchical_dataset.

    util = microstate_utilities();
    repo_cfg = util.load_config();
    plot_defaults = repo_cfg.plotting;

    p = inputParser;
    addRequired(p, 'input_arg', @(x) ischar(x) || isstring(x) || isstruct(x));
    addParameter(p, 'K', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    addParameter(p, 'output_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'display_normalisation', char(plot_defaults.display_normalisation), @(x) ischar(x) || isstring(x));
    addParameter(p, 'maplimits', char(plot_defaults.maplimits));
    addParameter(p, 'map_percentile', double(plot_defaults.map_percentile), @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 100);
    addParameter(p, 'colormap_name', char(plot_defaults.colormap_name), @(x) ischar(x) || isstring(x));
    addParameter(p, 'electrodes', char(plot_defaults.electrodes), @(x) ischar(x) || isstring(x));
    addParameter(p, 'numcontour', 6, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'visible', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'resolution', double(plot_defaults.resolution), @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, input_arg, varargin{:});
    cfg = p.Results;

    if exist('topoplot', 'file') ~= 2
        error('EEGLAB topoplot is not on the MATLAB path. Start EEGLAB or add it to the path first.');
    end

    [centers, labels, chanlocs, source_label, default_output_file] = load_template_source(input_arg, cfg.K);
    if isempty(centers)
        error('No interpolated template maps were found in the requested input.');
    end

    [centers, chanlocs, keep_idx] = restrict_to_plot_channels(centers, chanlocs);
    if size(centers, 2) ~= numel(chanlocs)
        error('Template width does not match the number of plottable channels.');
    end

    display_maps = apply_display_normalisation_local(centers, cfg.display_normalisation);
    map_limits = choose_map_limits(display_maps, cfg.maplimits, cfg.map_percentile);

    if isempty(cfg.output_file)
        plot_file = default_output_file;
    else
        plot_file = char(cfg.output_file);
    end
    out_dir = fileparts(plot_file);
    if isempty(out_dir)
        out_dir = pwd;
        plot_file = fullfile(out_dir, plot_file);
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    K = size(display_maps, 1);
    fig = figure('Visible', onoff(cfg.visible), 'Color', 'w', ...
        'Position', [50, 50, max(320, 240 * K), 280]);
    tl = tiledlayout(fig, 1, K, 'Padding', 'compact', 'TileSpacing', 'compact');
    colormap(fig, cfg.colormap_name);
    for k = 1:K
        ax = nexttile(tl, k); %#ok<NASGU>
        topoplot(display_maps(k, :), chanlocs, ...
            'electrodes', char(cfg.electrodes), ...
            'maplimits', map_limits, ...
            'numcontour', cfg.numcontour);
        if numel(labels) >= k && ~isempty(labels{k})
            title(labels{k}, 'FontWeight', 'bold', 'Interpreter', 'none');
        else
            title(sprintf('State %d', k), 'FontWeight', 'bold', 'Interpreter', 'none');
        end
    end
    title(tl, sprintf('Interpolated Canonical Template | %s', source_label), ...
        'FontWeight', 'bold', 'Interpreter', 'none');

    exportgraphics(fig, plot_file, 'Resolution', cfg.resolution);
    if ~cfg.visible
        close(fig);
    end

    PlotInfo = struct();
    PlotInfo.K = K;
    PlotInfo.labels = labels;
    PlotInfo.keep_channel_indices = keep_idx;
    PlotInfo.source_label = source_label;
    PlotInfo.display_normalisation = cfg.display_normalisation;
    PlotInfo.maplimits = map_limits;
    PlotInfo.output_file = plot_file;
end

function [centers, labels, chanlocs, source_label, output_file] = load_template_source(input_arg, K_req)
    centers = [];
    labels = {};
    chanlocs = [];
    source_label = '';
    output_file = fullfile(pwd, 'interpolated_template_plot.png');

    if isstruct(input_arg)
        [centers, labels, chanlocs, source_label] = load_from_struct(input_arg, K_req);
        output_file = fullfile(pwd, sprintf('interpolated_template_K%02d.png', size(centers, 1)));
        return;
    end

    mat_file = char(input_arg);
    if ~isfile(mat_file)
        error('Input file not found: %s', mat_file);
    end
    S = load(mat_file);
    if isfield(S, 'HResults')
        [centers, labels, chanlocs, source_label] = load_from_struct(S.HResults, K_req);
        output_file = fullfile(fileparts(mat_file), sprintf('interpolated_template_K%02d_plot.png', size(centers, 1)));
        return;
    end
    if isfield(S, 'template_cache_entry')
        entry = S.template_cache_entry;
        centers = entry.centers;
        labels = entry.labels;
        chanlocs = entry.common_chanlocs;
        source_label = sprintf('cache | K=%d | %s', entry.K, entry.channel_match_mode);
        output_file = fullfile(fileparts(mat_file), sprintf('interpolated_template_K%02d_plot.png', entry.K));
        return;
    end
    if isfield(S, 'centers') && isfield(S, 'common_chanlocs')
        centers = S.centers;
        if isfield(S, 'labels')
            labels = S.labels;
        end
        chanlocs = S.common_chanlocs;
        source_label = 'saved interpolated template';
        output_file = fullfile(fileparts(mat_file), [strip_extension(mat_file) '_plot.png']);
        return;
    end
    error('Could not find an interpolated template structure in %s.', mat_file);
end

function [centers, labels, chanlocs, source_label] = load_from_struct(H, K_req)
    centers = [];
    labels = {};
    chanlocs = [];
    source_label = '';

    if ~isfield(H, 'interpolated_template_library') || isempty(H.interpolated_template_library)
        error('HResults does not contain an interpolated_template_library.');
    end
    if nargin < 2 || isempty(K_req)
        if isfield(H, 'selected_K') && ~isempty(H.selected_K)
            K_req = H.selected_K;
        else
            K_req = H.interpolated_template_library(1).K;
        end
    end
    idx = find([H.interpolated_template_library.K] == round(K_req), 1, 'first');
    if isempty(idx)
        error('No interpolated template entry was saved for K=%d.', round(K_req));
    end
    entry = H.interpolated_template_library(idx);
    centers = entry.centers;
    labels = entry.labels;
    chanlocs = H.common_chanlocs;
    source_label = sprintf('HResults | K=%d | %s', entry.K, entry.channel_match_mode);
end

function [maps_out, chanlocs_out, keep_idx] = restrict_to_plot_channels(maps_in, chanlocs_in)
    keep_idx = false(1, numel(chanlocs_in));
    for i = 1:numel(chanlocs_in)
        has_xyz = isfield(chanlocs_in(i), 'X') && ~isempty(chanlocs_in(i).X) && ...
            isfield(chanlocs_in(i), 'Y') && ~isempty(chanlocs_in(i).Y) && ...
            isfield(chanlocs_in(i), 'Z') && ~isempty(chanlocs_in(i).Z) && ...
            all(isfinite(double([chanlocs_in(i).X chanlocs_in(i).Y chanlocs_in(i).Z])));
        has_polar = isfield(chanlocs_in(i), 'theta') && ~isempty(chanlocs_in(i).theta) && ...
            isfield(chanlocs_in(i), 'radius') && ~isempty(chanlocs_in(i).radius) && ...
            isfinite(double(chanlocs_in(i).theta)) && isfinite(double(chanlocs_in(i).radius));
        keep_idx(i) = has_xyz || has_polar;
    end
    maps_out = maps_in(:, keep_idx);
    chanlocs_out = chanlocs_in(keep_idx);
    if size(maps_out, 2) < 4
        error('Only %d plottable channels remain.', size(maps_out, 2));
    end
end

function maps_out = apply_display_normalisation_local(maps_in, mode)
    mode = lower(char(mode));
    maps_out = double(maps_in);
    switch mode
        case 'zscore'
            for i = 1:size(maps_out, 1)
                mu = mean(maps_out(i, :), 'omitnan');
                sd = std(maps_out(i, :), 0, 2, 'omitnan');
                if sd < eps
                    sd = 1;
                end
                maps_out(i, :) = (maps_out(i, :) - mu) ./ sd;
            end
        case 'l2'
            for i = 1:size(maps_out, 1)
                maps_out(i, :) = maps_out(i, :) ./ (norm(maps_out(i, :)) + eps);
            end
        case 'none'
            return;
        otherwise
            error('Unsupported display_normalisation: %s', mode);
    end
end

function limits = choose_map_limits(maps_in, mode, pct)
    if isnumeric(mode) && numel(mode) == 2
        limits = double(mode(:)');
        return;
    end
    mode = lower(char(mode));
    switch mode
        case 'global'
            lim = max(abs(maps_in(:)));
        case 'global_percentile'
            lim = prctile(abs(maps_in(:)), pct);
        otherwise
            lim = max(abs(maps_in(:)));
    end
    limits = [-lim, lim];
end

function txt = onoff(tf)
    if tf
        txt = 'on';
    else
        txt = 'off';
    end
end

function base = strip_extension(pth)
    [folder, name, ~] = fileparts(char(pth));
    if isempty(folder)
        base = name;
    else
        base = fullfile(folder, name);
    end
end
