function [Segmentation, Stats] = backfit_microstates_json(json_file, eeg_file, varargin)
% BACKFIT_MICROSTATES_JSON: Backfit microstates from JSON to EEG with Bayesian weighting
%
% INPUTS:
%   json_file   - Path to the JSON file containing microstate templates
%   eeg_file    - Path to the EEG file (.set or .mat)
%   varargin    - Optional parameters:
%                 'use_weights' - true/false (default: true) to use confidence weights
%                 'smooth_window' - Window size for temporal smoothing (default: 0)
%                 'ignore_polarity' - true/false (default: true)
%
% OUTPUTS:
%   Segmentation - Structure containing:
%                  .labels     - Microstate label for each time point
%                  .certainty  - Confidence/Probability of assignment
%                  .gfp        - Global Field Power
%                  .timestamps - Time vector
%   Stats        - Structure containing coverage, occurrence, etc.

    % Parse inputs
    p = inputParser;
    addParameter(p, 'use_weights', true, @islogical);
    addParameter(p, 'smooth_window', 0, @isnumeric);
    addParameter(p, 'ignore_polarity', true, @islogical);
    parse(p, varargin{:});
    
    use_weights = p.Results.use_weights;
    smooth_window = p.Results.smooth_window;
    ignore_polarity = p.Results.ignore_polarity;

    % Check files
    if ~exist(json_file, 'file'), error('JSON file not found: %s', json_file); end
    if ~exist(eeg_file, 'file'), error('EEG file not found: %s', eeg_file); end

    % =========================================================================
    % 1. LOAD MICROSTATE TEMPLATES FROM JSON
    % =========================================================================
    fprintf('Loading microstates from: %s\n', json_file);
    txt = fileread(json_file);
    data = jsondecode(txt);
    
    if ~isfield(data, 'estimated_microstates')
        error('JSON does not contain "estimated_microstates" field');
    end
    
    % Extract maps and weights
    states = fieldnames(data.estimated_microstates);
    n_states = length(states);
    
    % Get channel labels from first state to initialize
    first_state = data.estimated_microstates.(states{1});
    fields = fieldnames(first_state);
    % Filter out 'confidence' field to get channel names
    ch_fields = fields(~strcmp(fields, 'confidence'));
    n_channels_ms = length(ch_fields);
    
    maps = zeros(n_states, n_channels_ms);
    weights = ones(1, n_states) / n_states; % Default uniform weights
    
    % Map channel names to indices for consistent ordering
    ms_channel_labels = ch_fields;
    
    for k = 1:n_states
        state_name = states{k};
        state_data = data.estimated_microstates.(state_name);
        
        % Extract map values
        for c = 1:n_channels_ms
            ch_name = ms_channel_labels{c};
            if isfield(state_data, ch_name)
                maps(k, c) = state_data.(ch_name);
            else
                warning('Channel %s missing in state %s', ch_name, state_name);
            end
        end
        
        % Extract confidence weight if available
        if isfield(state_data, 'confidence')
            weights(k) = state_data.confidence;
        end
    end
    
    % Normalize maps
    maps = maps - mean(maps, 2);
    maps = maps ./ sqrt(sum(maps.^2, 2));
    
    if use_weights
        fprintf('Using Bayesian confidence weights: %s\n', mat2str(weights, 3));
    else
        weights = ones(1, n_states); % Uniform weights (standard backfitting)
        fprintf('Using uniform weights (standard backfitting)\n');
    end

    % =========================================================================
    % 2. LOAD AND PREPROCESS EEG
    % =========================================================================
    fprintf('Loading EEG from: %s\n', eeg_file);
    
    % Try loading with EEGLAB if available, otherwise standard load
    try
        if endsWith(eeg_file, '.set')
            EEG = pop_loadset(eeg_file);
            X = EEG.data;
            srate = EEG.srate;
            eeg_labels = {EEG.chanlocs.labels};
        elseif endsWith(eeg_file, '.mat')
            tmp = load(eeg_file);
            if isfield(tmp, 'EEG')
                X = tmp.EEG.data;
                srate = tmp.EEG.srate;
                eeg_labels = {tmp.EEG.chanlocs.labels};
            elseif isfield(tmp, 'X_clean')
                X = tmp.X_clean;
                srate = 250; % Default if not found
                eeg_labels = {}; % Unknown
            else
                error('Could not parse .mat file');
            end
        else
            error('Unsupported file format');
        end
    catch ME
        error('Failed to load EEG: %s', ME.message);
    end
    
    % Handle channel matching
    if ~isempty(eeg_labels)
        % Match channels between EEG and Microstates
        [common_labels, idx_eeg, idx_ms] = intersect(lower(eeg_labels), lower(ms_channel_labels), 'stable');
        
        if length(common_labels) < n_channels_ms * 0.8
            warning('Low channel match: %d/%d channels found', length(common_labels), n_channels_ms);
        end
        
        X = X(idx_eeg, :);
        maps = maps(:, idx_ms);
        fprintf('Matched %d channels between EEG and Microstates\n', length(common_labels));
    else
        warning('No channel labels in EEG. Assuming identical channel order.');
        if size(X, 1) ~= n_channels_ms
            error('Channel count mismatch: EEG has %d, Microstates have %d', size(X, 1), n_channels_ms);
        end
    end
    
    % Bandpass filter (2-20 Hz)
    fprintf('Preprocessing EEG (2-20 Hz bandpass)...\n');
    X = double(X);
    X = X - mean(X, 1); % Average reference
    
    % Simple FFT filter
    T = size(X, 2);
    F = fft(X, [], 2);
    freqs = (0:T-1) * (srate / T);
    bp = [2 20];
    mask = (freqs >= bp(1) & freqs <= bp(2)) | (freqs >= srate - bp(2) & freqs <= srate - bp(1));
    F(:, ~mask) = 0;
    X = real(ifft(F, [], 2));
    
    % Calculate GFP
    gfp = std(X, 0, 1);
    
    % =========================================================================
    % 3. BACKFITTING (SEGMENTATION)
    % =========================================================================
    fprintf('Backfitting %d samples...\n', T);
    
    % Normalize EEG at each time point
    X_norm = X ./ (sqrt(sum(X.^2, 1)) + eps);
    
    % Calculate spatial correlation (Cosine Similarity)
    % Correlation matrix: (n_states x n_samples)
    Corr = maps * X_norm;
    
    if ignore_polarity
        Corr = abs(Corr);
    end
    
    % Apply Bayesian Weighting (Model Evidence)
    % Score = Correlation * Weight
    % This biases the selection towards states with higher model evidence
    WeightedCorr = Corr .* weights(:);
    
    % Winner-Takes-All Assignment
    [max_score, labels] = max(WeightedCorr, [], 1);
    
    % Calculate Certainty as a normalized posterior probability (0 to 1)
    % P(k|t) = (Corr(k,t) * Prior(k)) / Sum_j(Corr(j,t) * Prior(j))
    % This assumes the likelihood is proportional to the spatial correlation.
    sum_weighted_corr = sum(WeightedCorr, 1);
    certainty = max_score ./ (sum_weighted_corr + eps);
    
    % Optional: Temporal Smoothing
    if smooth_window > 0
        fprintf('Applying temporal smoothing (window: %d)...\n', smooth_window);
        labels = smooth_labels(labels, smooth_window);
        
        % Re-calculate certainty based on smoothed labels
        % (This is an approximation)
        idx = sub2ind(size(WeightedCorr), labels, 1:T);
        certainty = WeightedCorr(idx);
    end
    
    % =========================================================================
    % 4. CALCULATE STATISTICS
    % =========================================================================
    fprintf('Calculating statistics...\n');
    
    % Global Explained Variance (GEV)
    % GEV = sum( (GFP * Corr_winner)^2 ) / sum(GFP^2)
    % Note: Use original (unweighted) correlation for GEV calculation
    idx = sub2ind(size(Corr), labels, 1:T);
    corr_winner = Corr(idx);
    gev = sum((gfp .* corr_winner).^2) / sum(gfp.^2);
    
    % Coverage (fraction of time)
    coverage = zeros(1, n_states);
    for k = 1:n_states
        coverage(k) = sum(labels == k) / T;
    end
    
    % Duration and Occurrence
    durations = cell(1, n_states);
    occurrences = zeros(1, n_states);
    
    % Run-length encoding
    if T > 0
        diff_labels = [1, diff(labels) ~= 0];
        change_indices = find(diff_labels);
        run_lengths = diff([change_indices, T+1]);
        run_values = labels(change_indices);
        
        for i = 1:length(run_values)
            k = run_values(i);
            durations{k} = [durations{k}, run_lengths(i)];
        end
        
        for k = 1:n_states
            occurrences(k) = length(durations{k}) / (T / srate); % per second
            mean_duration = mean(durations{k}) / srate * 1000; % ms
            if isnan(mean_duration), mean_duration = 0; end
            Stats.mean_duration(k) = mean_duration;
        end
    end
    
    % Pack outputs
    Segmentation.labels = labels;
    Segmentation.certainty = certainty;
    Segmentation.gfp = gfp;
    Segmentation.timestamps = (0:T-1) / srate;
    
    Stats.gev = gev;
    Stats.coverage = coverage;
    Stats.occurrence = occurrences;
    Stats.weights_used = weights;
    
    fprintf('Done. GEV: %.2f%%\n', gev * 100);
end

function labels_smooth = smooth_labels(labels, window_size)
% Simple majority voting smoothing
    labels_smooth = labels;
    T = length(labels);
    half_win = floor(window_size/2);
    
    for t = 1:T
        start_idx = max(1, t - half_win);
        end_idx = min(T, t + half_win);
        window = labels(start_idx:end_idx);
        labels_smooth(t) = mode(window);
    end
end
