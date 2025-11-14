function [Sim, maps_true, pos] = generate_microstate_eeg(K_true, snr_db, duration_s, sfreq, seed)
% GENERATE_MICROSTATE_EEG: Generate synthetic EEG with REALISTIC amplitudes
%
% Microstate amplitudes: 10-50 µV (typical values from literature)
% Pink noise baseline: 5-20 µV 
% GFP range: 0.5-3 µV
%
% NOTE: Generates 71-channel EEG to match artifact template montage
% Uses channel positions from template file

    if nargin < 4, sfreq = 250; end
    if nargin < 5, seed = 42; end
    
    rng(seed, 'twister');
    
    % Load channel positions from template file
    fprintf('Loading channel positions from template...\n');
    [pos, chanlocs, channel_labels] = load_chanlocs_from_template();
    
    if isempty(pos)
        error('Failed to load channel positions from template file');
    end
    
    C = size(pos, 1);
    fprintf('✓ Loaded %d channels from template\n', C);
    fprintf('✓ Channel labels: %s\n', sprintf('%s ', channel_labels{:}));
    
    % Generate true microstate templates with REALISTIC AMPLITUDE
    % Real microstate maps: 20-50 µV peak amplitude (let's use 30 µV as default)
    microstate_amplitude_uv = 30;  % microvolts
    
    len_scale = 0.25;
    W = rbf_kernel(pos, len_scale);
    maps_true_normalized = generate_microstate_templates(K_true, pos, W);
    maps_true = maps_true_normalized * microstate_amplitude_uv;  % Scale to µV
    
    % Temporal parameters
    T = round(duration_s * sfreq);
    mean_dur_ms = 80;
    mean_dur_samples = max(1, round((mean_dur_ms/1000)*sfreq));
    
    % Generate clean EEG
    X_clean = zeros(C, T);
    z = zeros(1, T);
    
    p_switch = 1 / mean_dur_samples;
    cur_state = randi(K_true);
    t = 1;
    
    while t <= T
        dwell = min(T - t + 1, geornd(p_switch) + 1);
        
        % Add spatial jitter to maps (~15% of amplitude)
        map_jitter_sd = 0.15 * microstate_amplitude_uv;
        jitter = map_jitter_sd * (W * randn(C, 1));
        template = maps_true(cur_state, :)' + jitter;
        
        % GFP envelope: realistic range 0.3-0.8 (modulates 30-80% of peak)
        gfp_env = ar1_positive_realistic(dwell, 0.98, 0.2);  % 0.3-0.8 range
        
        seg_idx = t:(t + dwell - 1);
        z(seg_idx) = cur_state;
        X_clean(:, seg_idx) = template * gfp_env;
        
        % Pick next state
        if K_true > 1
            other_states = setdiff(1:K_true, cur_state);
            cur_state = other_states(randi(length(other_states)));
        end
        
        t = t + dwell;
    end
    
    % Generate PINK noise with REALISTIC AMPLITUDE
    % Background EEG: 5-20 µV RMS (let's use 10 µV as baseline)
    background_noise_rms_uv = 10;
    
    fprintf('Generating pink noise (%.1f µV RMS)...\n', background_noise_rms_uv);
    N = zeros(C, T);
    for c = 1:C
        N(c, :) = generate_pink_noise(T) * background_noise_rms_uv;
    end
    
    % Spatially correlate noise through RBF kernel
    for t = 1:T
        N(:, t) = W * (randn(C, 1) .* N(:, t) / (norm(N(:, t)) + eps));
    end
    
    % Normalize to target noise level
    N = N / (std(N(:)) + eps) * background_noise_rms_uv;
    
    % Scale by SNR
    Ps = mean(X_clean(:).^2);
    Pn = mean(N(:).^2) + eps;
    
    % SNR_dB = 10 * log10(Ps / Pn) => Scale noise appropriately
    scale = sqrt((10^(-snr_db/10) * Ps) / Pn);
    X_noisy = X_clean + scale * N;
    
    % Inject real artifacts from template file
    fprintf('Injecting real artifacts...\n');
    try
        X_noisy = inject_real_artifacts(X_noisy, sfreq);
    catch ME
        fprintf('⚠ Warning: Could not load artifacts (%s). Continuing without real artifacts.\n', ME.message);
    end
    
    % Return structure with amplitude info
    Sim = struct( ...
        'X_clean', X_clean, ...
        'X_noisy', X_noisy, ...
        'maps_true', maps_true, ...  % In µV
        'z_true', z, ...
        'sfreq', sfreq, ...
        'pos', pos, ...
        'chanlocs', chanlocs, ...  % Channel location structure
        'channel_labels', {channel_labels}, ...  % ✅ NEW: Channel labels as cell array
        'K_true', K_true, ...
        'SNR_dB', snr_db, ...
        'duration_s', duration_s, ...
        'microstate_amplitude_uv', microstate_amplitude_uv, ...
        'background_noise_rms_uv', background_noise_rms_uv);
end

% ======================== CHANNEL MONTAGE ========================

function [pos, chanlocs, channel_labels] = load_chanlocs_from_template()
% LOAD_CHANLOCS_FROM_TEMPLATE: Load channel positions from template SET file
%
% Uses the same template file as artifact injection to ensure consistency

    pos = [];
    chanlocs = [];
    channel_labels = {};  % ✅ NEW: Initialize labels
    
    try
        % Find template file
        template_file = find_template_file();
        
        if isempty(template_file)
            fprintf('Error: Template SET file not found\n');
            return;
        end
        
        fprintf('Loading channel positions from: %s\n', template_file);
        
        % Load template with EEGLAB
        if ~exist('pop_loadset', 'file')
            fprintf('Error: EEGLAB not found\n');
            return;
        end
        
        EEG_template = pop_loadset('filename', template_file);
        
        if isempty(EEG_template.chanlocs)
            fprintf('Error: Template has no channel locations\n');
            return;
        end
        
        chanlocs = EEG_template.chanlocs;
        n_channels = length(chanlocs);
        
        % ✅ NEW: Extract channel labels from chanlocs
        channel_labels = cell(n_channels, 1);
        for i = 1:n_channels
            if isfield(chanlocs(i), 'labels') && ~isempty(chanlocs(i).labels)
                channel_labels{i} = chanlocs(i).labels;
            else
                channel_labels{i} = sprintf('Ch%03d', i);  % Fallback
            end
        end
        
        fprintf('✓ Extracted %d channel labels\n', length(channel_labels));
        
        % Extract 3D positions
        pos = zeros(n_channels, 3);
        valid_coords = 0;
        
        for i = 1:n_channels
            if ~isempty(chanlocs(i).X) && ~isempty(chanlocs(i).Y) && ~isempty(chanlocs(i).Z)
                pos(i, :) = [chanlocs(i).X, chanlocs(i).Y, chanlocs(i).Z];
                valid_coords = valid_coords + 1;
            else
                % Fallback: convert from spherical coordinates
                if ~isempty(chanlocs(i).theta) && ~isempty(chanlocs(i).radius)
                    [x, y, z] = sph2cart(deg2rad(chanlocs(i).theta), ...
                                        deg2rad(90 - chanlocs(i).radius), 1);
                    pos(i, :) = [x, y, z];
                    valid_coords = valid_coords + 1;
                end
            end
        end
        
        % Normalize positions to unit sphere
        for i = 1:size(pos, 1)
            r = norm(pos(i, :));
            if r > 0
                pos(i, :) = pos(i, :) / r;
            end
        end
        
        fprintf('✓ Loaded %d channels with valid 3D coordinates\n', valid_coords);
        
    catch ME
        fprintf('Error loading template: %s\n', ME.message);
        pos = [];
        chanlocs = [];
        channel_labels = {};
    end
end

function template_file = find_template_file()
% FIND_TEMPLATE_FILE: Locate the template SET file

    search_paths = {
        'E:\EEGs\SPM_MS\MetaMaps_2023_06.set', ...
        fullfile(pwd, 'MetaMaps_2023_06.set'), ...
        fullfile(pwd, '..', 'MetaMaps_2023_06.set'), ...
        fullfile(pwd, 'data', 'MetaMaps_2023_06.set'), ...
        fullfile(pwd, '..', 'data', 'MetaMaps_2023_06.set')
    };
    
    template_file = '';
    for p = 1:length(search_paths)
        if isfile(search_paths{p})
            template_file = search_paths{p};
            return;
        end
    end
end

% ======================== ARTIFACT INJECTION ========================

function X_out = inject_real_artifacts(X, sfreq_target)
% INJECT_REAL_ARTIFACTS: Load and inject real EEG artifacts from EEGLAB SET file

    artifact_file = find_artifact_template();
    
    if isempty(artifact_file)
        error('artefact_template.set not found');
    end
    
    fprintf('  Loading artifact template from: %s\n', artifact_file);
    
    if ~exist('pop_loadset', 'file')
        error('EEGLAB not found. Please add EEGLAB to path.');
    end
    
    try
        EEG = pop_loadset('filename', artifact_file);
        
        if isempty(EEG.data)
            error('No data in SET file');
        end
        
        X_artifacts = EEG.data;  % Channels x Timepoints
        sfreq_artifacts = EEG.srate;
        n_channels_artifact = size(X_artifacts, 1);
        n_channels_target = size(X, 1);
        
        fprintf('  ✓ Loaded %d channels, %.1f seconds at %.0f Hz\n', ...
            n_channels_artifact, size(X_artifacts, 2)/sfreq_artifacts, sfreq_artifacts);
        
        % Verify channel match
        if n_channels_artifact ~= n_channels_target
            fprintf('  ⚠ Channel count mismatch: artifact has %d, target has %d\n', ...
                n_channels_artifact, n_channels_target);
        end
        
    catch ME
        error('Failed to load SET file: %s', ME.message);
    end
    
    % Resample artifacts if needed
    if sfreq_artifacts ~= sfreq_target
        fprintf('  Resampling artifacts from %.0f to %.0f Hz\n', sfreq_artifacts, sfreq_target);
        X_artifacts = resample_eeg_data(X_artifacts, sfreq_artifacts, sfreq_target);
    end
    
    % Inject random artifact snippets with REALISTIC AMPLITUDE
    X_out = X;
    n_artifacts = max(1, floor(size(X, 2) / (10 * sfreq_target)));
    snippet_duration_s = 0.5;
    snippet_len = round(snippet_duration_s * sfreq_target);
    
    fprintf('  Injecting %d artifact snippets (~%.1f s each)...\n', n_artifacts, snippet_duration_s);
    
    for art_idx = 1:n_artifacts
        max_start = max(1, size(X_artifacts, 2) - snippet_len);
        art_start = randi([1, max_start]);
        art_end = min(art_start + snippet_len - 1, size(X_artifacts, 2));
        
        max_inj = max(1, size(X_out, 2) - (art_end - art_start + 1));
        inj_start = randi([1, max_inj]);
        inj_end = inj_start + (art_end - art_start);
        
        artifact_snippet = X_artifacts(:, art_start:art_end);
        
        % Only interpolate if channel counts don't match
        if size(artifact_snippet, 1) ~= size(X_out, 1)
            fprintf('    Interpolating artifact from %d to %d channels\n', ...
                size(artifact_snippet, 1), size(X_out, 1));
            artifact_snippet = interpolate_channels(artifact_snippet, size(X_out, 1));
        end
        
        % Scale artifact to be 50-100% of signal amplitude (realistic contamination)
        artifact_rms = sqrt(mean(artifact_snippet(:).^2));
        output_std = std(X_out(:));
        artifact_scale = (0.5 + 0.5*rand()) * output_std / (artifact_rms + eps);
        artifact_snippet = artifact_scale * artifact_snippet;
        
        if inj_end <= size(X_out, 2)
            fade_len = min(10, floor((inj_end - inj_start) / 10));
            fade_in = linspace(0, 1, fade_len);
            fade_out = linspace(1, 0, fade_len);
            
            X_out(:, inj_start:inj_start+fade_len-1) = ...
                X_out(:, inj_start:inj_start+fade_len-1) + ...
                artifact_snippet(:, 1:fade_len) .* fade_in;
            
            mid_start = inj_start + fade_len;
            mid_end = inj_end - fade_len;
            if mid_start <= mid_end
                X_out(:, mid_start:mid_end) = ...
                    X_out(:, mid_start:mid_end) + ...
                    artifact_snippet(:, fade_len+1:end-fade_len);
            end
            
            X_out(:, inj_end-fade_len+1:inj_end) = ...
                X_out(:, inj_end-fade_len+1:inj_end) + ...
                artifact_snippet(:, end-fade_len+1:end) .* fade_out;
        end
    end
    
    fprintf('  ✓ Artifacts injected successfully\n');
end

function artifact_file = find_artifact_template()
    search_paths = {
        fullfile(pwd, 'artefact_template.set'), ...
        fullfile(pwd, '..', 'artefact_template.set'), ...
        fullfile(pwd, 'data', 'artefact_template.set'), ...
        fullfile(pwd, '..', 'data', 'artefact_template.set'), ...
        'E:\EEGs\SPM_MS\artefact_template.set', ...
        'D:\EEGData\artefact_template.set'
    };
    
    artifact_file = '';
    for p = 1:length(search_paths)
        if isfile(search_paths{p})
            artifact_file = search_paths{p};
            return;
        end
    end
end

function X_resampled = resample_eeg_data(X, sfreq_old, sfreq_new)
    [P, Q] = rat(sfreq_new / sfreq_old);
    X_resampled = resample(X', P, Q)';
end

function X_interp = interpolate_channels(X_source, n_target)
    [n_source, T] = size(X_source);
    
    if n_source == n_target
        X_interp = X_source;
        return;
    end
    
    if n_target > n_source
        X_interp = zeros(n_target, T);
        source_idx = linspace(1, n_source, n_target);
        
        for t = 1:T
            X_interp(:, t) = interp1(1:n_source, X_source(:, t), source_idx, 'linear', 'extrap');
        end
    else
        X_interp = zeros(n_target, T);
        bin_size = n_source / n_target;
        
        for i = 1:n_target
            start_idx = round((i-1) * bin_size + 1);
            end_idx = round(i * bin_size);
            X_interp(i, :) = mean(X_source(start_idx:end_idx, :), 1);
        end
    end
end

function pink_noise = generate_pink_noise(N)
    white = randn(1, N);
    fft_white = fft(white);
    
    freqs = (0:N-1)';
    freqs(1) = 1;
    spectrum_1d = 1 ./ sqrt(freqs);
    
    if mod(N, 2) == 0
        spectrum = [spectrum_1d(1:N/2+1); flipud(spectrum_1d(2:N/2))];
    else
        spectrum = [spectrum_1d(1:(N+1)/2); flipud(spectrum_1d(2:(N+1)/2))];
    end
    
    spectrum = spectrum(1:N);
    spectrum = reshape(spectrum, size(fft_white));
    
    fft_pink = fft_white .* spectrum;
    pink_noise = real(ifft(fft_pink));
    pink_noise = pink_noise / (std(pink_noise) + eps);
end

function templates = generate_microstate_templates(K, pos, W)
    C = size(pos, 1);
    templates = zeros(K, C);
    anchors = fibonacci_sphere(K);
    for k = 1:K
        v = pos * anchors(k, :)';
        m = W * v;
        m = m - mean(m);
        templates(k, :) = m / (norm(m) + eps);
    end
end

function g = ar1_positive_realistic(T, rho, eta)
    % Generate GFP envelope with REALISTIC range: 0.3-0.8
    if T < 2
        g = 0.5 * ones(1, T);
        return;
    end
    
    g = zeros(1, T);
    x = 0;
    for t = 1:T
        x = rho * x + eta * randn;
        g(t) = log1p(exp(x));
    end
    
    g_mean = mean(g);
    g_std = std(g);
    if g_std > eps
        g = (g - g_mean) / g_std;
    else
        g = (g - g_mean);
    end
    
    g_min = min(g);
    g_max = max(g);
    if g_max > g_min
        % Map to 0.3-0.8 range (more realistic GFP)
        g = 0.3 + 0.5 * ((g - g_min) / (g_max - g_min));
    else
        g = 0.55 * ones(1, T);
    end
end

function W = rbf_kernel(pos, len_scale)
    D = pdist2(pos, pos);
    W = exp(-(D.^2) / (2 * len_scale^2));
    W = W ./ (sum(W, 2) + eps);
end

function pos = fibonacci_sphere(C)
    ga = (sqrt(5) - 1) / 2;
    i = (0:C-1)' + 0.5;
    phi = 2 * pi * mod(i * ga, 1);
    z = 1 - 2 * i / C;
    r = sqrt(max(0, 1 - z.^2));
    pos = [r.*cos(phi), r.*sin(phi), z];
end