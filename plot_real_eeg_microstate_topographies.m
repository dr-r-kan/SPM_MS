function plot_file = plot_real_eeg_microstate_topographies(json_file, eeg_file, plot_file)
% PLOT_REAL_EEG_MICROSTATE_TOPOGRAPHIES Plot estimated microstates as scalp maps.
%
% Uses montage/channel-location information from the real EEG .set file,
% rather than treating microstate vectors as images.

    if nargin < 3 || isempty(plot_file)
        [json_dir, json_name, ~] = fileparts(json_file);
        plot_file = fullfile(json_dir, [json_name '_microstates.png']);
    end

    if ~exist('pop_loadset', 'file')
        error('EEGLAB pop_loadset is not on the MATLAB path.');
    end
    if ~exist('topoplot', 'file')
        error('EEGLAB topoplot is not on the MATLAB path.');
    end
    if ~exist(json_file, 'file')
        error('JSON file not found: %s', json_file);
    end
    if ~exist(eeg_file, 'file')
        error('EEG file not found: %s', eeg_file);
    end

    data = jsondecode(fileread(json_file));
    EEG = pop_loadset('filename', eeg_file);
    chanlocs_all = EEG.chanlocs;

    json_labels = cellstr(data.channel_info.labels);
    if isfield(data.channel_info, 'labels_sanitized')
        json_labels_sanitized = cellstr(data.channel_info.labels_sanitized);
    else
        json_labels_sanitized = sanitize_channel_labels_plot(json_labels);
    end

    [json_idx, set_idx] = match_json_channels_to_set(json_labels, chanlocs_all);
    valid = has_usable_topoplot_location(chanlocs_all(set_idx));
    json_idx = json_idx(valid);
    set_idx = set_idx(valid);

    if numel(json_idx) < 4
        error('Only %d channels could be matched to usable scalp locations.', numel(json_idx));
    end

    chanlocs = chanlocs_all(set_idx);
    est_states = fieldnames(data.estimated_microstates);
    K = numel(est_states);
    maps = zeros(K, numel(json_idx));

    for k = 1:K
        state_data = data.estimated_microstates.(est_states{k});
        for c = 1:numel(json_idx)
            ch_key = json_labels_sanitized{json_idx(c)};
            if isfield(state_data, ch_key)
                maps(k, c) = state_data.(ch_key);
            else
                maps(k, c) = NaN;
            end
        end
    end

    keep_ch = all(isfinite(maps), 1);
    maps = maps(:, keep_ch);
    chanlocs = chanlocs(keep_ch);
    if numel(chanlocs) < 4
        error('Fewer than four finite mapped channels remain after JSON extraction.');
    end

    clim_abs = max(abs(maps(:)));
    if clim_abs <= eps || ~isfinite(clim_abs)
        clim_abs = 1;
    end

    n_cols = min(4, K);
    n_rows = ceil(K / n_cols);
    fig = figure('Name', 'Microstate Topographies', ...
        'NumberTitle', 'off', ...
        'Color', 'white', ...
        'Visible', 'off', ...
        'Position', [100, 100, 280*n_cols, 260*n_rows + 80]);

    for k = 1:K
        subplot(n_rows, n_cols, k);
        topoplot(maps(k, :), chanlocs, ...
            'electrodes', 'off', ...
            'numcontour', 6, ...
            'maplimits', [-clim_abs clim_abs]);
        colormap(gca, 'jet');
        colorbar;
        title(state_title(data.estimated_microstates.(est_states{k}), k), ...
            'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    end

    title_text = json_title(data, json_file);
    sgtitle(title_text, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

    out_dir = fileparts(plot_file);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    exportgraphics(fig, plot_file, 'Resolution', 200);
    close(fig);
end

function [json_idx, set_idx] = match_json_channels_to_set(json_labels, chanlocs)
    set_labels = cell(numel(chanlocs), 1);
    for i = 1:numel(chanlocs)
        set_labels{i} = lower(strtrim(char(chanlocs(i).labels)));
    end
    json_idx = [];
    set_idx = [];
    for j = 1:numel(json_labels)
        label = lower(strtrim(char(json_labels{j})));
        hit = find(strcmp(label, set_labels), 1, 'first');
        if ~isempty(hit)
            json_idx(end+1) = j; %#ok<AGROW>
            set_idx(end+1) = hit; %#ok<AGROW>
        end
    end
end

function valid = has_usable_topoplot_location(chanlocs)
    valid = false(1, numel(chanlocs));
    for i = 1:numel(chanlocs)
        has_polar = isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && ...
            isfield(chanlocs(i), 'radius') && ~isempty(chanlocs(i).radius) && ...
            isfinite(chanlocs(i).theta) && isfinite(chanlocs(i).radius) && ...
            chanlocs(i).radius > 0 && chanlocs(i).radius <= 0.5;
        has_xyz = isfield(chanlocs(i), 'X') && ~isempty(chanlocs(i).X) && ...
            isfield(chanlocs(i), 'Y') && ~isempty(chanlocs(i).Y) && ...
            isfield(chanlocs(i), 'Z') && ~isempty(chanlocs(i).Z) && ...
            all(isfinite([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z])) && ...
            norm([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]) > eps;
        valid(i) = has_polar || (~isfield(chanlocs(i), 'radius') && has_xyz);
    end
end

function title_text = state_title(state_data, k)
    title_text = sprintf('State %d', k);
    if isfield(state_data, 'template_label')
        title_text = sprintf('%s (%s', title_text, state_data.template_label);
        if isfield(state_data, 'template_correlation')
            title_text = sprintf('%s r=%.2f', title_text, state_data.template_correlation);
        end
        title_text = sprintf('%s)', title_text);
    end
end

function title_text = json_title(data, json_file)
    [~, json_name, ~] = fileparts(json_file);
    title_text = json_name;
    if isfield(data, 'metadata')
        meta = data.metadata;
        parts = {json_name};
        if isfield(meta, 'method')
            parts{end+1} = sprintf('Method: %s', meta.method); %#ok<AGROW>
        end
        if isfield(meta, 'criterion')
            parts{end+1} = sprintf('Criterion: %s', meta.criterion); %#ok<AGROW>
        end
        if isfield(meta, 'K_estimated')
            parts{end+1} = sprintf('K: %d', meta.K_estimated); %#ok<AGROW>
        end
        title_text = strjoin(parts, ' | ');
    end
end

function sanitized = sanitize_channel_labels_plot(ch_labels)
    sanitized = cell(size(ch_labels));
    for i = 1:length(ch_labels)
        label = char(ch_labels{i});
        label = regexprep(label, '[-/\\\s\.\,\(\)\[\]\{\}]', '_');
        label = regexprep(label, '^_+|_+$', '');
        if isempty(label) || isempty(regexp(label(1), '[A-Za-z]', 'once'))
            label = ['Ch' label];
        end
        sanitized{i} = matlab.lang.makeValidName(label);
    end
end
