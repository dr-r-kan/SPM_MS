%% Plot Microstate Comparison: True vs Estimated
% This script loads the JSON file containing true and estimated microstate data,
% and plots them as topographic maps for visual comparison.
% Requires: EEGLAB with topoplot function and the microstates toolbox

clear all; close all; clc;

%% Load JSON data
json_file = 'fit_003_K4_SNR+10_kmeans_koenig_elbow.json';
json_file_folder = "E:\EEGs\SPM_MS_old\out_microstate_comparison\microstates_json\";
name_stem = erase(json_file, '.json');
json_file = append(json_file_folder, json_file)
data = jsondecode(fileread(json_file));

set_file = 'MetaMaps_2023_06.set';

% Extract metadata if present
if isfield(data, 'metadata')
    META = data.metadata;
else
    META = struct();
end

% Print metadata summary
fprintf('Loaded JSON metadata:\n');
disp(META);

%% Extract data from JSON
true_microstates = data.true_microstates;
estimated_microstates = data.estimated_microstates;
json_channel_labels = data.channel_info.labels;

if isfield(data.channel_info, 'labels_sanitized')
    json_channel_labels_sanitized = data.channel_info.labels_sanitized;
else
    % Fallback: sanitize labels ourselves
    json_channel_labels_sanitized = sanitize_channel_labels_plot(json_channel_labels);
end

% Number of channels and states
n_channels = data.channel_info.n_channels;
true_states = fieldnames(true_microstates);
est_states = fieldnames(estimated_microstates);
n_true_states = length(true_states);
n_est_states = length(est_states);

fprintf('Loaded JSON with %d channels, %d true states, %d estimated states\n', ...
    n_channels, n_true_states, n_est_states);

%% Load montage template from .set file
if ~exist(set_file, 'file')
    error('Set file not found: %s\nPlease update the set_file path in the script.', set_file);
end

EEG = pop_loadset(set_file);
fprintf('✓ Loaded .set file: %s\n', set_file);
fprintf('  Contains %d channels\n', EEG.nbchan);

%% Match JSON channels to .set file channels by name
% Create mapping from JSON channel index to .set file channel index
json_to_set_mapping = zeros(n_channels, 1);
matched_count = 0;
unmatched_channels = {};

for json_ch = 1:n_channels
    json_label = json_channel_labels{json_ch};
    
    % Try to find this channel in the .set file
    found = false;
    for set_ch = 1:EEG.nbchan
        set_label = EEG.chanlocs(set_ch).labels;
        if strcmp(json_label, set_label)
            json_to_set_mapping(json_ch) = set_ch;
            found = true;
            matched_count = matched_count + 1;
            break;
        end
    end
    
    if ~found
        unmatched_channels{end+1} = json_label;
    end
end

fprintf('\n✓ Channel matching results:\n');
fprintf('  Matched: %d / %d channels\n', matched_count, n_channels);

if ~isempty(unmatched_channels)
    fprintf('  Unmatched channels: %s\n', strjoin(unmatched_channels, ', '));
    warning('Not all channels were matched. Unmatched channels will be skipped.');
end

if matched_count < n_channels * 0.8
    error('Less than 80%% of channels matched. Check that the .set file contains the correct montage.');
end

%% ✅ FIXED: Extract channel locations from .set file for matched channels
% Initialize as a proper array of structs
chanlocs(1:n_channels) = struct('labels', '', 'X', [], 'Y', [], 'Z', [], ...
    'theta', [], 'radius', [], 'sph_theta', [], 'sph_phi', [], 'sph_radius', []);

for json_ch = 1:n_channels
    set_ch = json_to_set_mapping(json_ch);
    if set_ch > 0  % Only include matched channels
        % Copy all fields from the reference .set file
        chanlocs_template = EEG.chanlocs(set_ch);
        field_names = fieldnames(chanlocs_template);
        for f = 1:length(field_names)
            fn = field_names{f};
            chanlocs(json_ch).(fn) = chanlocs_template.(fn);
        end
    end
end

fprintf('✓ Created chanlocs structure with %d channels\n\n', n_channels);

%% Convert microstate data to matrices
true_data = zeros(n_channels, n_true_states);
est_data = zeros(n_channels, n_est_states);

% ✅ FIXED: Extract data using sanitized channel labels from JSON
for i = 1:n_true_states
    state_name = true_states{i};
    state_data = true_microstates.(state_name);
    for ch = 1:n_channels
        % ✅ Use sanitized label as JSON field name
        ch_name_sanitized = json_channel_labels_sanitized{ch};
        if isfield(state_data, ch_name_sanitized)
            true_data(ch, i) = state_data.(ch_name_sanitized);
        else
            % Fallback: try original label
            ch_name_orig = json_channel_labels{ch};
            if isfield(state_data, ch_name_orig)
                true_data(ch, i) = state_data.(ch_name_orig);
            else
                warning('Channel %s not found in true_microstates.%s', ch_name_sanitized, state_name);
            end
        end
    end
end

for i = 1:n_est_states
    state_name = est_states{i};
    state_data = estimated_microstates.(state_name);
    for ch = 1:n_channels
        % ✅ Use sanitized label as JSON field name
        ch_name_sanitized = json_channel_labels_sanitized{ch};
        if isfield(state_data, ch_name_sanitized)
            est_data(ch, i) = state_data.(ch_name_sanitized);
        else
            % Fallback: try original label
            ch_name_orig = json_channel_labels{ch};
            if isfield(state_data, ch_name_orig)
                est_data(ch, i) = state_data.(ch_name_orig);
            else
                warning('Channel %s not found in estimated_microstates.%s', ch_name_sanitized, state_name);
            end
        end
    end
end

fprintf('Data matrix sizes - True: %d x %d, Estimated: %d x %d\n', ...
    size(true_data, 1), size(true_data, 2), size(est_data, 1), size(est_data, 2));

%% Calculate global min/max for consistent color scaling
true_clim = [min(true_data(:)), max(true_data(:))];
est_clim = [min(est_data(:)), max(est_data(:))];
global_clim = [min(true_clim(1), est_clim(1)), max(true_clim(2), est_clim(2))];

fprintf('Color scale: [%.4f, %.4f]\n\n', global_clim(1), global_clim(2));

%% Set up publication-quality plot style
set(0, 'DefaultFigureColor', 'white');
set(0, 'DefaultAxesColor', 'white');
set(0, 'DefaultAxesXColor', 'black');
set(0, 'DefaultAxesYColor', 'black');
set(0, 'DefaultTextColor', 'black');
set(0, 'DefaultTextInterpreter', 'tex');

%% ✅ Figure 1: Create detailed comparison figure for ALL matched states
if isfield(data, 'matches') && ~isempty(data.matches)
    matches = data.matches;
    match_fields = fieldnames(matches);
    n_matches = length(match_fields);
    
    fprintf('\nCreating matched microstates figure with %d matched states...\n', n_matches);
    
    % ✅ FIXED: One row per matched pair, 3 columns (true, est, diff)
    n_rows = n_matches;
    n_cols = 3;
    
    % Use tiledlayout for overall layout and title, but plot into
    % independent axes placed at the tiles' positions. This avoids
    % TiledChartLayout warnings caused when third-party functions
    % (like topoplot) try to change axes Position.
    fig_matched = figure('Name', 'ALL Matched States Detail Comparison', ...
        'NumberTitle', 'off', 'Position', [100, 100, 1200, 150 + 300*n_matches], 'Color', 'white');

    t = tiledlayout(fig_matched, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'loose');

    % First pass: create temporary axes in each tile to capture positions
    n_tiles = n_rows * n_cols;
    tile_positions = cell(1, n_tiles);
    for ti = 1:n_tiles
        ax_temp = nexttile(t, ti);
        drawnow; % ensure layout has been computed
        tile_positions{ti} = get(ax_temp, 'Position');
        % remove the temporary tile-managed axes so we can place a free axes there
        delete(ax_temp);
    end

    % Second pass: create free axes at captured positions and plot into them
    for match_idx = 1:n_matches
        match = matches.(match_fields{match_idx});
        est_idx = match.estimated_state;
        true_idx = match.true_state;
        similarity = match.similarity;

        % Compute tile indices for the three columns in this row
        base_tile = (match_idx-1)*3;

        % Verify indices are within bounds
        if true_idx <= n_true_states && est_idx <= n_est_states
            % Plot true state (shrink axes slightly to leave room for title)
            pos1 = tile_positions{base_tile + 1};
            % shrink height a bit so title area remains clear
            pos1(4) = pos1(4) * 0.90;
            ax1 = axes('Parent', fig_matched, 'Position', pos1);
            axes(ax1);
            topoplot(true_data(:, true_idx), chanlocs, 'electrodes', 'off', 'style', 'map', 'maplimits', global_clim);
            title(sprintf('True State %d', true_idx), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
            caxis(global_clim);
            colorbar('FontSize', 8);

            % Plot estimated state
            pos2 = tile_positions{base_tile + 2};
            pos2(4) = pos2(4) * 0.90;
            ax2 = axes('Parent', fig_matched, 'Position', pos2);
            axes(ax2);
            topoplot(est_data(:, est_idx), chanlocs, 'electrodes', 'off', 'style', 'map', 'maplimits', global_clim);
            title(sprintf('Estimated State %d', est_idx), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
            caxis(global_clim);
            colorbar('FontSize', 8);

            % Plot difference
            pos3 = tile_positions{base_tile + 3};
            pos3(4) = pos3(4) * 0.90;
            ax3 = axes('Parent', fig_matched, 'Position', pos3);
            axes(ax3);
            difference = true_data(:, true_idx) - est_data(:, est_idx);
            diff_lim = max(abs(difference(:)));
            topoplot(difference, chanlocs, 'electrodes', 'off', 'style', 'map', 'maplimits', [-diff_lim, diff_lim]);
            title(sprintf('Difference\n(Sim: %.3f)', similarity), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
            colorbar('FontSize', 8);
        end
    end

    % Add metadata to the tiledlayout Title when available
    meta_str = format_meta_short(META);
    title_str = sprintf('ALL Matched Microstate Comparison (%d matches) with Difference Maps\n%s', n_matches, meta_str);
    t.Title.String = title_str;
    t.Title.FontSize = 12;
    t.Title.FontWeight = 'bold';
    t.Title.Interpreter = 'none';
    try
        t.Title.Color = [0 0 0];
    catch
        % ignore if property unsupported
    end
end

%% ✅ Figure 2: Create figure for unmatched (extra) estimated states
n_extra_est = n_est_states - n_true_states;

if n_extra_est > 0
    fprintf('\nCreating unmatched estimated microstates figure with %d extra states...\n', n_extra_est);
    
    % Find which estimated states were NOT matched
    matched_est_indices = [];
    if isfield(data, 'matches') && ~isempty(data.matches)
        matches = data.matches;
        match_fields = fieldnames(matches);
        for match_idx = 1:length(match_fields)
            match = matches.(match_fields{match_idx});
            matched_est_indices = [matched_est_indices, match.estimated_state];
        end
    end
    
    unmatched_est_indices = setdiff(1:n_est_states, matched_est_indices);
    
    % Create figure for unmatched states
    max_cols_extra = 5;
    n_cols_extra = min(length(unmatched_est_indices), max_cols_extra);
    n_rows_extra = ceil(length(unmatched_est_indices) / n_cols_extra);
    
    fig_extra = figure('Name', 'Unmatched Estimated Microstates', ...
        'NumberTitle', 'off', 'Position', [100, 100, 200 + 350*n_cols_extra, 250 + 350*n_rows_extra], 'Color', 'white');

    t_extra = tiledlayout(fig_extra, n_rows_extra, n_cols_extra, 'TileSpacing', 'compact', 'Padding', 'loose');

    % Capture tile positions by creating temporary axes
    n_tiles_extra = n_rows_extra * n_cols_extra;
    tile_pos_extra = cell(1, n_tiles_extra);
    for ti = 1:n_tiles_extra
        ax_temp = nexttile(t_extra, ti);
        drawnow;
        tile_pos_extra{ti} = get(ax_temp, 'Position');
        delete(ax_temp);
    end

    for extra_plot_idx = 1:length(unmatched_est_indices)
        state_idx = unmatched_est_indices(extra_plot_idx);
    pos = tile_pos_extra{extra_plot_idx};
    pos(4) = pos(4) * 0.88; % leave extra headroom for the title
    ax = axes('Parent', fig_extra, 'Position', pos);
        axes(ax);
        topoplot(est_data(:, state_idx), chanlocs, 'electrodes', 'off', 'style', 'map', 'maplimits', global_clim);
        % Compute best similarity to any true state (cosine similarity)
        best_sim = NaN;
        best_true_idx = NaN;
        try
            est_vec = est_data(:, state_idx);
            % normalize
            est_norm = norm(est_vec);
            if est_norm > eps && exist('true_data', 'var') && ~isempty(true_data)
                sims = zeros(1, size(true_data, 2));
                for ti = 1:size(true_data, 2)
                    tvec = true_data(:, ti);
                    sims(ti) = (tvec' * est_vec) / (norm(tvec) * est_norm + eps);
                end
                [best_sim, bi] = max(sims);
                best_true_idx = bi;
            end
        catch
            best_sim = NaN;
            best_true_idx = NaN;
        end

        if ~isnan(best_sim)
            title(sprintf('Estimated State %d (Unmatched)\nBest true: %d (sim=%.3f)', state_idx, best_true_idx, best_sim), 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
        else
            title(sprintf('Estimated State %d (Unmatched)', state_idx), 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
        end
        caxis(global_clim);
        h_cbar = colorbar;
        set(h_cbar, 'FontSize', 9);
    end
    meta_str = format_meta_short(META);
    title_str = sprintf('Unmatched/Extra Estimated Microstates (%d states)\n%s', n_extra_est, meta_str);
    t_extra.Title.String = title_str;
    t_extra.Title.FontSize = 14;
    t_extra.Title.FontWeight = 'bold';
    t_extra.Title.Interpreter = 'none';
    try
        t_extra.Title.Color = [0 0 0];
    catch
        % ignore
    end
end

%% Save figures
fprintf('\nSaving figures...\n');

if isfield(data, 'matches') && ~isempty(data.matches)
    save_name = sprintf('%s_matched_detail.png', name_stem);
    saveas(fig_matched, save_name);
    fprintf('✓ Saved matched plot\n');
end

if n_extra_est > 0
    save_name = sprintf('%s_unmatched_estimated.png', name_stem);
    saveas(fig_extra, save_name);
    fprintf('✓ Saved unmatched plot\n');
end

fprintf('\nPlotting completed successfully!\n');
fprintf('Channel locations matched from: %s\n', set_file);

% Helper to format metadata for titles
function s = format_meta_short(M)
    if isempty(M) || ~isstruct(M)
        s = '';
        return;
    end
    parts = {};
    if isfield(M, 'method'), parts{end+1} = sprintf('Method: %s', M.method); end
    if isfield(M, 'criterion'), parts{end+1} = sprintf('Criterion: %s', M.criterion); end
    if isfield(M, 'K_true'), parts{end+1} = sprintf('K_true: %d', M.K_true); end
    if isfield(M, 'K_estimated'), parts{end+1} = sprintf('K_est: %d', M.K_estimated); end
    if isfield(M, 'SNR_dB'), parts{end+1} = sprintf('SNR: %+.0fdB', M.SNR_dB); end
    if isfield(M, 'runtime_s'), parts{end+1} = sprintf('Runtime: %.1fs', M.runtime_s); end
    s = strjoin(parts, ' | ');
end

% ✅ HELPER FUNCTION
function sanitized = sanitize_channel_labels_plot(ch_labels)
% SANITIZE_CHANNEL_LABELS_PLOT: Convert channel labels to valid struct field names
% (Matches the sanitization in save_microstate_json.m)

    sanitized = cell(size(ch_labels));
    
    for i = 1:length(ch_labels)
        label = ch_labels{i};
        
        % Replace invalid characters with underscores
        label = regexprep(label, '[-/\s\.\,\(\)\[\]\{\}]', '_');
        
        % Remove leading/trailing underscores
        label = regexprep(label, '^_+|_+$', '');
        
        % If label becomes empty or starts with digit, add 'ch_' prefix
        if isempty(label) || ~isnan(str2double(label(1)))
            label = ['ch_' label];
        end
        
        sanitized{i} = label;
    end
end