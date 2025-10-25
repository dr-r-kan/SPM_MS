function [Sim, maps_true, pos] = generate_microstate_eeg(K_true, snr_db, duration_s, sfreq, seed)
% GENERATE_MICROSTATE_EEG: Simulate clustery EEG data with microstate templates
%
% Generates synthetic multi-channel EEG data from microstate templates with
% configurable noise levels and microstate K values.
%
% INPUTS:
%   K_true      - Number of true microstates (clusters) to generate
%   snr_db      - Signal-to-noise ratio in dB
%   duration_s  - Duration of EEG data in seconds
%   sfreq       - Sampling frequency in Hz (default 250)
%   seed        - RNG seed for reproducibility (default 42)
%
% OUTPUTS:
%   Sim         - Structure containing:
%                 .X_clean      - Clean EEG (C x T)
%                 .X_noisy      - Noisy EEG (C x T)
%                 .maps_true    - True microstate templates (K x C)
%                 .z_true       - True state sequence (1 x T)
%                 .sfreq        - Sampling frequency
%                 .pos          - Channel positions (C x 3)
%   maps_true   - True microstate templates (K x C)
%   pos         - Channel positions (C x 3)

    if nargin < 4, sfreq = 250; end
    if nargin < 5, seed = 42; end
    
    rng(seed);
    
    % --- Channel Setup (Standard 10-20) ---
    C = 64;  % Standard EEG channel count
    pos = fibonacci_sphere(C);  % Quasi-uniform sphere positions
    
    % --- Generate True Microstate Templates ---
    % Create K_true spatially-distributed templates using RBF kernels
    len_scale = 0.25;
    W = rbf_kernel(pos, len_scale);
    maps_true = generate_microstate_templates(K_true, pos, W);
    
    % --- Temporal Parameters ---
    T = round(duration_s * sfreq);
    mean_dur_ms = 80;  % Mean microstate duration
    mean_dur_samples = max(1, round((mean_dur_ms/1000)*sfreq));
    
    % --- Generate Clean EEG (Winner-Take-All) ---
    X_clean = zeros(C, T);
    z = zeros(1, T);
    
    p_switch = 1 / mean_dur_samples;  % Transition probability
    cur_state = randi(K_true);
    t = 1;
    
    while t <= T
        % Dwell time in current state (geometric distribution)
        dwell = min(T - t + 1, geornd(p_switch) + 1);
        
        % Get template for current state with jitter
        map_jitter_sd = 0.15;
        jitter = map_jitter_sd * (W * randn(C, 1));
        template = maps_true(cur_state, :)' + jitter;
        template = (template - mean(template)) / (norm(template) + eps);
        
        % Apply temporal modulation (GFP envelope)
        gfp_env = ar1_positive(dwell, 0.98, 0.2);
        
        % Fill in this segment
        seg_idx = t:(t + dwell - 1);
        z(seg_idx) = cur_state;
        X_clean(:, seg_idx) = template * gfp_env;
        
        % Transition to next state
        cur_state = randi(K_true - 1);
        if cur_state >= cur_state, cur_state = cur_state + 1; end
        t = t + dwell;
    end
    
    % --- Generate Correlated Noise ---
    beta_noise = 1.0;  % 1/f noise exponent
    N = zeros(C, T);
    for c = 1:C
        N(c, :) = temporal_colored_noise(beta_noise, T, randn(1, T));
    end
    
    % Spatially correlate noise using RBF kernel
    for t = 1:T
        N(:, t) = W * (randn(C, 1) .* N(:, t) / (norm(N(:, t)) + eps));
    end
    
    % --- Scale Noise by SNR ---
    Ps = mean(X_clean(:).^2);
    Pn = mean(N(:).^2) + eps;
    scale = sqrt((10^(-snr_db/10) * Ps) / Pn);
    X_noisy = X_clean + scale * N;
    
    % --- Add Physiological Artifacts (Optional) ---
    % Add eye blinks
    n_blinks = max(1, floor(T / (5 * sfreq)));
    y_pos = pos(:, 2);
    eye_weight = normalise_vec(y_pos - min(y_pos));
    blink_template = hann_window(round(0.4 * sfreq))';
    
    for b = 1:n_blinks
        t0 = randi([1, T]);
        sl = max(1, t0 - numel(blink_template)/2):min(T, t0 + numel(blink_template)/2 - 1);
        X_noisy(:, sl) = X_noisy(:, sl) + 2.0 * eye_weight * blink_template(1:numel(sl));
    end
    
    % --- Add ECG artifact ---
    tvec = (0:T-1) / sfreq;
    ecg = sin(2*pi*1.2*tvec);
    z_pos = pos(:, 3);
    ecg_weight = normalise_vec(min(z_pos) - z_pos);
    X_noisy = X_noisy + 0.8 * ecg_weight * ecg;
    
    % --- Return Structure ---
    Sim = struct( ...
        'X_clean', X_clean, ...
        'X_noisy', X_noisy, ...
        'maps_true', maps_true, ...
        'z_true', z, ...
        'sfreq', sfreq, ...
        'pos', pos, ...
        'K_true', K_true, ...
        'SNR_dB', snr_db, ...
        'duration_s', duration_s);
end

% ======================== HELPER FUNCTIONS ========================

function templates = generate_microstate_templates(K, pos, W)
    % Generate K spatially-distributed microstate templates
    C = size(pos, 1);
    templates = zeros(K, C);
    
    % Create templates as spatial patterns with RBF kernels
    anchors = fibonacci_sphere(K);
    
    for k = 1:K
        % Project position onto anchor direction
        v = pos * anchors(k, :)';
        % Apply RBF smoothing
        m = W * v;
        % Normalize
        m = m - mean(m);
        templates(k, :) = m / (norm(m) + eps);
    end
end

function g = ar1_positive(T, rho, eta)
    % Generate AR(1) process with positive values
    g = zeros(1, T);
    x = 0;
    for t = 1:T
        x = rho * x + eta * randn;
        g(t) = log1p(exp(x));
    end
    g = (g - mean(g)) / (std(g) + eps);
    g = 0.5 + 0.5 * ((g - min(g)) / (max(g) - min(g) + eps));
end

function W = rbf_kernel(pos, len_scale)
    % Radial basis function kernel for spatial smoothing
    C = size(pos, 1);
    D = zeros(C, C);
    for i = 1:C
        for j = i:C
            d = norm(pos(i, :) - pos(j, :));
            D(i, j) = d;
            D(j, i) = d;
        end
    end
    W = exp(-(D.^2) / (2 * len_scale^2));
    W = W ./ (sum(W, 2) + eps);
end

function n = temporal_colored_noise(beta, T, seed_vec)
    % Generate temporally correlated 1/f noise
    freqs = (0:floor(T/2)) / T;
    mag = zeros(size(freqs));
    mag(2:end) = 1 ./ (freqs(2:end).^(beta/2));
    ph = rand(size(mag)) * 2 * pi;
    spec = mag .* exp(1i * ph);
    x = real(ifft([spec, conj(spec(end-1:-1:2))], 'symmetric'));
    x = x .* sign(seed_vec(:)');
    n = (x - mean(x)) ./ (std(x) + eps);
end

function pos = fibonacci_sphere(C)
    % Generate quasi-uniform points on sphere using golden angle
    ga = (sqrt(5) - 1) / 2;
    i = (0:C-1)' + 0.5;
    phi = 2 * pi * mod(i * ga, 1);
    z = 1 - 2 * i / C;
    r = sqrt(max(0, 1 - z.^2));
    pos = [r.*cos(phi), r.*sin(phi), z];
end

function w = hann_window(N)
    % Hann window
    if N <= 1
        w = 1;
        return;
    end
    n = (0:N-1)';
    w = 0.5 * (1 - cos(2*pi*n/(N-1)));
end

function y = normalise_vec(x)
    % Normalize vector to [0, 1]
    x = x(:);
    y = (x - min(x)) / (max(x) - min(x) + eps);
end