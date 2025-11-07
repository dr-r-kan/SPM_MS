%% Plot Microstate Comparison: True vs Estimated
% This script loads the JSON file containing true and estimated microstate data,
% and plots them as topographic maps for visual comparison.
% Requires: EEGLAB with topoplot function and the microstates toolbox

clear all; close all; clc;

%% Load JSON data
json_file = 'fit_009_K4_SNR+10_spm_vb_elbow_sil_combined.json';
json_file_folder = "E:\EEGs\SPM_MS\out_microstate_comparison\microstates_json\";
json_file = append(json_file_folder, json_file)
data = jsondecode(fileread(json_file));

set_file = 'MetaMaps_2023_06.set';

%% Extract data from JSON
true_microstates = data.true_microstates;
estimated_microstates = data.estimated_microstates;
json_channel_labels = data.channel_info.labels;

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

%% Extract channel locations from .set file for matched channels
chanlocs = struct();
for json_ch = 1:n_channels
    set_ch = json_to_set_mapping(json_ch);
    if set_ch > 0  % Only include matched channels
        chanlocs(json_ch) = EEG.chanlocs(set_ch);
    end
end

fprintf('✓ Created chanlocs structure with %d channels\n\n', numel(chanlocs));

%% Convert microstate data to matrices
true_data = zeros(n_channels, n_true_states);
est_data = zeros(n_channels, n_est_states);

for i = 1:n_true_states
    state_name = true_states{i};
    state_data = true_microstates.(state_name);
    for ch = 1:n_channels
        ch_name = sprintf('Ch%03d', ch);
        true_data(ch, i) = state_data.(ch_name);
    end
end

for i = 1:n_est_states
    state_name = est_states{i};
    state_data = estimated_microstates.(state_name);
    for ch = 1:n_channels
        ch_name = sprintf('Ch%03d', ch);
        est_data(ch, i) = state_data.(ch_name);
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

%% Create comparison figure - Overview
fig = figure('Name', 'Microstate Comparison: True vs Estimated', ...
    'NumberTitle', 'off', 'Position', [100, 100, 1600, 1000], 'Color', 'white');

% Plot True Microstates
for state_idx = 1:n_true_states
    subplot(3, max(n_true_states, n_est_states), state_idx);
    topoplot(true_data(:, state_idx), chanlocs, 'electrodes', 'on', ...
        'style', 'map', 'maplimits', global_clim, 'emarker', {'o', 'k', 8, 1});
    title(sprintf('True State %d', state_idx), 'FontSize', 11, 'FontWeight', 'bold');
    caxis(global_clim);
    h_cbar = colorbar;
    set(h_cbar, 'FontSize', 9);
end

% Plot Estimated Microstates
for state_idx = 1:n_est_states
    subplot(3, max(n_true_states, n_est_states), ...
        max(n_true_states, n_est_states) + state_idx);
    topoplot(est_data(:, state_idx), chanlocs, 'electrodes', 'on', ...
        'style', 'map', 'maplimits', global_clim, 'emarker', {'o', 'k', 8, 1});
    title(sprintf('Estimated State %d', state_idx), 'FontSize', 11, 'FontWeight', 'bold');
    caxis(global_clim);
    h_cbar = colorbar;
    set(h_cbar, 'FontSize', 9);
end

%% Plot Recovery Statistics in bottom section
subplot(3, max(n_true_states, n_est_states), ...
    2*max(n_true_states, n_est_states) + 1);
hold on;
ax = gca;
ax.Visible = 'off';
axis off;

% Display recovery metrics
recovery = data.recovery;
metadata = data.metadata;

info_text = sprintf(['Recovery Metrics:\n' ...
    '─────────────────────\n' ...
    'Matched States: %d / %d\n' ...
    'Sensitivity: %.3f\n' ...
    'Precision: %.3f\n' ...
    'F1-Score: %.4f\n' ...
    'Mean Recovery (Matched): %.4f\n' ...
    'Mean Recovery (Padded): %.4f\n\n' ...
    'Analysis Parameters:\n' ...
    '─────────────────────\n' ...
    'Method: %s\n' ...
    'True K: %d | Estimated K: %d\n' ...
    'SNR: %d dB\n' ...
    'Duration: %.1f s\n' ...
    'Channels: %d\n'], ...
    recovery.n_matched, n_true_states, recovery.sensitivity, recovery.precision, ...
    recovery.f1_score, recovery.mean_recovery_matched, recovery.mean_recovery_padded, ...
    metadata.method, metadata.K_true, metadata.K_estimated, metadata.SNR_dB, ...
    metadata.duration_s, metadata.n_channels);

text(0.05, 0.95, info_text, 'FontSize', 10, 'VerticalAlignment', 'top', ...
    'FontFamily', 'monospaced', 'BackgroundColor', [0.95 0.95 0.95], ...
    'EdgeColor', 'black', 'Padding', 10, 'Color', 'black');

sgtitle(['Microstate Comparison: True (K=' num2str(metadata.K_true) ...
    ') vs Estimated (K=' num2str(metadata.K_estimated) ')'], ...
    'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');

%% Create detailed comparison figure for matched states
if isfield(data, 'matches') && ~isempty(data.matches)
    matches = data.matches;
    match_fields = fieldnames(matches);
    n_matches = length(match_fields);
    
    fig2 = figure('Name', 'Matched States Detail Comparison', ...
        'NumberTitle', 'off', 'Position', [100, 1050, 1400, 800], 'Color', 'white');
    
    for match_idx = 1:n_matches
        match = matches.(match_fields{match_idx});
        est_idx = match.estimated_state;
        true_idx = match.true_state;
        similarity = match.similarity;
        
        % Verify indices are within bounds
        if true_idx <= n_true_states && est_idx <= n_est_states
            % Plot true state
            subplot(n_matches, 3, (match_idx-1)*3 + 1);
            topoplot(true_data(:, true_idx), chanlocs, 'electrodes', 'on', ...
                'style', 'map', 'maplimits', global_clim, 'emarker', {'o', 'k', 8, 1});
            title(sprintf('True State %d', true_idx), 'FontSize', 10, 'FontWeight', 'bold');
            caxis(global_clim);
            colorbar('FontSize', 8);
            
            % Plot estimated state
            subplot(n_matches, 3, (match_idx-1)*3 + 2);
            topoplot(est_data(:, est_idx), chanlocs, 'electrodes', 'on', ...
                'style', 'map', 'maplimits', global_clim, 'emarker', {'o', 'k', 8, 1});
            title(sprintf('Estimated State %d', est_idx), 'FontSize', 10, 'FontWeight', 'bold');
            caxis(global_clim);
            colorbar('FontSize', 8);
            
            % Plot difference
            subplot(n_matches, 3, (match_idx-1)*3 + 3);
            difference = true_data(:, true_idx) - est_data(:, est_idx);
            diff_lim = max(abs(difference(:)));
            topoplot(difference, chanlocs, 'electrodes', 'on', ...
                'style', 'map', 'maplimits', [-diff_lim, diff_lim], ...
                'emarker', {'o', 'k', 8, 1});
            title(sprintf('Difference\n(Similarity: %.4f)', similarity), ...
                'FontSize', 10, 'FontWeight', 'bold');
            colorbar('FontSize', 8);
        end
    end
    
    sgtitle('Matched Microstate Comparison with Difference Maps', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
end

%% Save figures
fprintf('Saving figures...\n');
saveas(fig, 'microstate_comparison_overview.png');
fprintf('✓ Saved: microstate_comparison_overview.png\n');

if isfield(data, 'matches') && ~isempty(data.matches)
    saveas(fig2, 'microstate_matched_detail.png');
    fprintf('✓ Saved: microstate_matched_detail.png\n');
end

fprintf('\nPlotting completed successfully!\n');
fprintf('Channel locations matched from: %s\n', set_file);