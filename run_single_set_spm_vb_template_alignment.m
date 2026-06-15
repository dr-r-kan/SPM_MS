% RUN_SINGLE_SET_SPM_VB_TEMPLATE_ALIGNMENT
%
% Fit one EEGLAB .set file with SPM-VB microstates, select K using the
% free-energy/elbow/silhouette combined criterion, align the fitted states
% to the canonical MetaMaps template, and save a labelled scalp-topography
% plot ordered by template label.
%
% Usage:
%   1. Set eeg_file below to your .set path.
%   2. Run this script.
%
% Outputs are left in the workspace:
%   Results   - fitted microstate solution with template alignment
%   json_file - saved JSON summary
%   plot_file - saved labelled scalp-map plot

clearvars -except eeg_file

util = microstate_utilities();
cfg = util.load_config();


% ===== User settings =====
% Example:
eeg_file = '/home/rohan/EEG_Data/LEMON/EEG_Preprocessed/sub-010004_EC.set';
template_file = char(cfg.paths.template_file);
K_candidates = 4:6;
verbose = true;
% =========================

if isempty(eeg_file)
    error(['Set the variable eeg_file to the .set file you want to analyse, for example:' newline ...
        '  eeg_file = ''/absolute/path/to/recording.set'';']);
end

eeg_file = util.resolve_path(char(eeg_file), pwd);
template_file = util.resolve_path(template_file, util.project_root());

if ~isfile(eeg_file)
    error('EEG file not found: %s', eeg_file);
end
if ~endsWith(lower(eeg_file), '.set')
    error('This script is intended for EEGLAB .set files. Received: %s', eeg_file);
end
if ~isfile(template_file)
    error('Template file not found: %s', template_file);
end

[eeg_dir, eeg_name, ~] = fileparts(eeg_file);
output_dir = fullfile(eeg_dir, [eeg_name '_spm_vb_template_alignment']);
json_dir = fullfile(output_dir, 'json');
plot_dir = fullfile(output_dir, 'plots');

if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(json_dir, 'dir'), mkdir(json_dir); end
if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end

[Results, json_file] = analyze_single_eeg_file(eeg_file, ...
    'method', 'spm_vb', ...
    'criterion', 'elbow_sil_combined', ...
    'K_candidates', K_candidates, ...
    'align_template', true, ...
    'template_file', template_file, ...
    'save_json', true, ...
    'json_dir', json_dir, ...
    'plot_dir', plot_dir, ...
    'verbose', verbose);

if ~isfield(Results, 'template_alignment') || isempty(Results.template_alignment)
    error('Template alignment was not produced. Check the template file and channel labels.');
end

plot_file = '';
if ~isempty(json_file)
    [~, json_name, ~] = fileparts(json_file);
    plot_file = fullfile(plot_dir, [json_name '_microstates.png']);
end

fprintf('\nCompleted single-file SPM-VB template-aligned fit.\n');
fprintf('EEG file: %s\n', eeg_file);
fprintf('Selected K: %d\n', Results.K_estimated);
fprintf('Template labels: %s\n', strjoin(cellstr(string(Results.template_alignment.labels(:)')), ', '));
if ~isempty(json_file)
    fprintf('JSON: %s\n', json_file);
end
if ~isempty(plot_file) && isfile(plot_file)
    fprintf('Plot: %s\n', plot_file);
else
    fprintf('Plot directory: %s\n', plot_dir);
end
