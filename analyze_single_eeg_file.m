function Results = analyze_single_eeg_file(eeg_file, varargin)
% ANALYZE_SINGLE_EEG_FILE: Analyze a single real EEG file
%
% This function applies microstate analysis to a real EEG file without
% requiring ground truth maps. Recovery metrics are computed only if
% ground truth is provided.
%
% INPUTS:
%   eeg_file     - Path to EEG file (.set, .mat, or other supported format)
%   
% OPTIONAL NAME-VALUE PAIRS:
%   'method'       - Analysis method: 'kmeans_koenig' or 'spm_vb' (default: 'kmeans_koenig')
%   'criterion'    - Selection criterion (default: 'silhouette' for kmeans, 'elbow_sil_combined' for spm_vb)
%   'K_candidates' - Vector of K values to test (default: 2:10)
%   'true_maps'    - Optional ground truth maps for validation (default: [])
%   'sfreq'        - Sampling frequency (default: 250)
%   'verbose'      - Show progress messages (default: true)
%
% OUTPUTS:
%   Results      - Structure containing fitted microstates and metrics
%
% EXAMPLE:
%   % Analyze without ground truth
%   Results = analyze_single_eeg_file('my_eeg.set', 'method', 'kmeans_koenig');
%   
%   % Analyze with ground truth for validation
%   Results = analyze_single_eeg_file('my_eeg.set', 'method', 'spm_vb', ...
%                                     'true_maps', ground_truth_maps);

    % Parse inputs
    p = inputParser;
    addRequired(p, 'eeg_file', @ischar);
    addParameter(p, 'method', 'kmeans_koenig', @ischar);
    addParameter(p, 'criterion', '', @ischar);  % Auto-select based on method
    addParameter(p, 'K_candidates', 2:10, @isnumeric);
    addParameter(p, 'true_maps', [], @isnumeric);
    addParameter(p, 'sfreq', 250, @isnumeric);
    addParameter(p, 'verbose', true, @islogical);
    parse(p, eeg_file, varargin{:});
    
    CONFIG = p.Results;
    
    % Auto-select criterion if not specified
    if isempty(CONFIG.criterion)
        if strcmp(CONFIG.method, 'spm_vb')
            CONFIG.criterion = 'elbow_sil_combined';
        else
            CONFIG.criterion = 'silhouette';
        end
    end
    
    % Validate method and criterion combination
    if strcmp(CONFIG.method, 'spm_vb') && strcmp(CONFIG.criterion, 'gev')
        error('GEV criterion is not supported for spm_vb method. Use ''silhouette'', ''free_energy'', ''elbow'', or ''elbow_sil_combined'' instead.');
    end
    
    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('Single EEG File Analysis\n');
        fprintf('========================================\n');
        fprintf('File: %s\n', eeg_file);
        fprintf('Method: %s\n', CONFIG.method);
        fprintf('Criterion: %s\n', CONFIG.criterion);
        fprintf('K candidates: %s\n', mat2str(CONFIG.K_candidates));
        if ~isempty(CONFIG.true_maps)
            fprintf('Ground truth: Provided (%d maps)\n', size(CONFIG.true_maps, 1));
        else
            fprintf('Ground truth: Not provided (real data mode)\n');
        end
        fprintf('========================================\n\n');
    end
    
    % Load EEG data
    if CONFIG.verbose
        fprintf('1. Loading EEG data...\n');
    end
    
    if ~exist(eeg_file, 'file')
        error('EEG file not found: %s', eeg_file);
    end
    
    % Load based on file extension
    [~, ~, ext] = fileparts(eeg_file);
    
    if strcmp(ext, '.set')
        % EEGLAB .set file
        if ~exist('pop_loadset', 'file')
            error('EEGLAB not found. Please add EEGLAB to MATLAB path.');
        end
        EEG = pop_loadset(eeg_file);
        eeg_data = EEG.data;  % Channels × Time
        sfreq = EEG.srate;
        
    elseif strcmp(ext, '.mat')
        % MATLAB .mat file
        data = load(eeg_file);
        if isfield(data, 'eeg_data')
            eeg_data = data.eeg_data;
        elseif isfield(data, 'data')
            eeg_data = data.data;
        elseif isfield(data, 'EEG')
            eeg_data = data.EEG.data;
        else
            error('Could not find EEG data in .mat file. Expected fields: eeg_data, data, or EEG');
        end
        if isfield(data, 'sfreq')
            sfreq = data.sfreq;
        elseif isfield(data, 'srate')
            sfreq = data.srate;
        else
            sfreq = CONFIG.sfreq;
            if CONFIG.verbose
                fprintf('  ⚠ Sampling frequency not found, using default: %d Hz\n', sfreq);
            end
        end
        
    else
        error('Unsupported file format: %s. Use .set or .mat files.', ext);
    end
    
    [n_channels, n_samples] = size(eeg_data);
    duration_s = n_samples / sfreq;
    
    if CONFIG.verbose
        fprintf('  ✓ Loaded: %d channels, %d samples (%.1f s @ %d Hz)\n', ...
            n_channels, n_samples, duration_s, sfreq);
    end
    
    % Create simulation-like structure for compatibility with existing methods
    Sim = struct();
    Sim.eeg = eeg_data;
    Sim.sfreq = sfreq;
    Sim.duration_s = duration_s;
    Sim.n_channels = n_channels;
    Sim.n_samples = n_samples;
    
    % Add ground truth if provided
    if ~isempty(CONFIG.true_maps)
        Sim.maps_true = CONFIG.true_maps;
        Sim.K_true = size(CONFIG.true_maps, 1);
        Sim.SNR_dB = NaN;  % Unknown for real data
    else
        % No ground truth - will skip recovery metrics
        Sim.maps_true = [];
        Sim.K_true = NaN;
        Sim.SNR_dB = NaN;
    end
    
    % Run analysis
    if CONFIG.verbose
        fprintf('\n2. Running microstate analysis...\n');
    end
    
    try
        if strcmp(CONFIG.method, 'spm_vb')
            Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, CONFIG.criterion);
        elseif strcmp(CONFIG.method, 'kmeans_koenig')
            Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_candidates, CONFIG.criterion);
        else
            error('Unknown method: %s. Use ''kmeans_koenig'' or ''spm_vb''.', CONFIG.method);
        end
    catch ME
        fprintf('\n✗ Analysis failed: %s\n', ME.message);
        fprintf('  Stack trace:\n%s\n', ME.getReport());
        rethrow(ME);
    end
    
    % Display results
    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('ANALYSIS RESULTS\n');
        fprintf('========================================\n');
        fprintf('Method: %s\n', Results.method);
        fprintf('Criterion: %s\n', Results.criterion);
        fprintf('K estimated: %d\n', Results.K_estimated);
        if ~isnan(Sim.K_true)
            fprintf('K true: %d (correct: %s)\n', Sim.K_true, ...
                iif(Results.K_estimated == Sim.K_true, 'YES', 'NO'));
        end
        fprintf('Runtime: %.2f s\n', Results.runtime);
        
        if isfield(Results, 'recovery_metrics') && ~isempty(Results.recovery_metrics)
            fprintf('\nRecovery Metrics:\n');
            fprintf('  Matched maps: %d\n', Results.recovery_metrics.n_matched);
            fprintf('  F1 score: %.4f\n', Results.recovery_metrics.f1_score);
            fprintf('  Sensitivity: %.4f\n', Results.recovery_metrics.sensitivity);
            fprintf('  Precision: %.4f\n', Results.recovery_metrics.precision);
        end
        fprintf('========================================\n\n');
    end
end

function out = iif(condition, true_val, false_val)
    % Inline if function
    if condition
        out = true_val;
    else
        out = false_val;
    end
end