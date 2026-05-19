function report = validate_simulated_eeg_representativeness(Sim, varargin)
% VALIDATE_SIMULATED_EEG_REPRESENTATIVENESS Quantitative QC for synthetic EEG.
%
% The checks are deliberately broad sanity bounds, not proof of realism:
% amplitude/GFP scale, 1/f spectral slope, microstate dwell times, state
% coverage, map distinctness, and artifact/heavy-tail evidence.

    p = inputParser;
    addRequired(p, 'Sim', @isstruct);
    addParameter(p, 'output_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'verbose', true, @islogical);
    parse(p, Sim, varargin{:});

    if ~isfield(Sim, 'X_noisy') || isempty(Sim.X_noisy)
        error('Sim.X_noisy is required.');
    end
    if ~isfield(Sim, 'sfreq') || isempty(Sim.sfreq)
        error('Sim.sfreq is required.');
    end

    X = double(Sim.X_noisy);
    sfreq = double(Sim.sfreq);
    [n_channels, n_samples] = size(X);
    duration_s = n_samples / sfreq;

    report = struct();
    report.summary = struct( ...
        'n_channels', n_channels, ...
        'n_samples', n_samples, ...
        'sfreq', sfreq, ...
        'duration_s', duration_s, ...
        'has_nan_or_inf', any(~isfinite(X(:))));

    channel_rms = sqrt(mean((X - mean(X, 2)).^2, 2));
    gfp = sqrt(mean((X - mean(X, 1)).^2, 1));
    report.amplitude = struct( ...
        'global_rms_uv', std(X(:)), ...
        'median_channel_rms_uv', median(channel_rms), ...
        'p95_channel_rms_uv', prctile(channel_rms, 95), ...
        'peak_abs_uv', max(abs(X(:))), ...
        'gfp_median_uv', median(gfp), ...
        'gfp_p95_uv', prctile(gfp, 95), ...
        'gfp_max_uv', max(gfp), ...
        'flat_channel_fraction', mean(channel_rms < 1e-6));

    [freqs, mean_power] = mean_power_spectrum(X, sfreq);
    report.spectrum = spectrum_metrics(freqs, mean_power);

    report.microstate_dynamics = struct();
    if isfield(Sim, 'z_true') && ~isempty(Sim.z_true)
        report.microstate_dynamics = microstate_dynamics_metrics(Sim.z_true, sfreq);
    end

    report.maps = struct();
    if isfield(Sim, 'maps_true') && ~isempty(Sim.maps_true)
        util = microstate_utilities_SHARED();
        maps_norm = util.normalize_maps(Sim.maps_true);
        corr_matrix = abs(maps_norm * maps_norm');
        corr_matrix(eye(size(corr_matrix)) > 0) = NaN;
        report.maps = struct( ...
            'K_true', size(Sim.maps_true, 1), ...
            'max_abs_between_map_correlation', max(corr_matrix(:), [], 'omitnan'), ...
            'mean_abs_between_map_correlation', mean(corr_matrix(:), 'omitnan'), ...
            'max_abs_map_uv', max(abs(Sim.maps_true(:))));
    end

    report.artifacts = struct( ...
        'excess_kurtosis', kurtosis(X(:)) - 3, ...
        'fraction_samples_over_100uv', mean(any(abs(X) > 100, 1)), ...
        'fraction_samples_over_200uv', mean(any(abs(X) > 200, 1)));
    if isfield(Sim, 'n_overlap_events')
        report.artifacts.n_overlap_events = Sim.n_overlap_events;
    end

    checks = build_checks(report);
    report.checks = checks;
    n_fail = sum(strcmp({checks.status}, 'FAIL'));
    n_warn = sum(strcmp({checks.status}, 'WARN'));
    if n_fail > 0
        report.overall_status = 'FAIL';
    elseif n_warn > 0
        report.overall_status = 'WARN';
    else
        report.overall_status = 'PASS';
    end

    if p.Results.verbose
        print_report(report);
    end
    if ~isempty(p.Results.output_file)
        write_report_text(report, char(p.Results.output_file));
    end
end

function [freqs, mean_power] = mean_power_spectrum(X, sfreq)
    X = X - mean(X, 2);
    n = size(X, 2);
    F = fft(X, [], 2);
    n_half = floor(n / 2) + 1;
    power = abs(F(:, 1:n_half)).^2 / n;
    freqs = (0:(n_half - 1)) * (sfreq / n);
    mean_power = mean(power, 1);
end

function s = spectrum_metrics(freqs, power)
    idx_fit = freqs >= 2 & freqs <= min(40, max(freqs)) & power > 0;
    if sum(idx_fit) >= 3
        coeff = polyfit(log10(freqs(idx_fit)), log10(power(idx_fit)), 1);
        slope = coeff(1);
    else
        slope = NaN;
    end
    s = struct( ...
        'power_slope_2_40hz', slope, ...
        'delta_1_4', band_power(freqs, power, [1 4]), ...
        'theta_4_8', band_power(freqs, power, [4 8]), ...
        'alpha_8_13', band_power(freqs, power, [8 13]), ...
        'beta_13_30', band_power(freqs, power, [13 30]), ...
        'line_48_52', band_power(freqs, power, [48 52]));
end

function pwr = band_power(freqs, power, band)
    idx = freqs >= band(1) & freqs < band(2);
    if any(idx)
        pwr = trapz(freqs(idx), power(idx));
    else
        pwr = NaN;
    end
end

function d = microstate_dynamics_metrics(z, sfreq)
    z = z(:)';
    run_lengths = [];
    run_states = [];
    if isempty(z)
        d = struct();
        return;
    end
    start_idx = 1;
    for i = 2:(numel(z) + 1)
        if i > numel(z) || z(i) ~= z(start_idx)
            run_lengths(end+1) = i - start_idx; %#ok<AGROW>
            run_states(end+1) = z(start_idx); %#ok<AGROW>
            start_idx = i;
        end
    end
    dwell_ms = 1000 * run_lengths / sfreq;
    states = unique(z(~isnan(z)));
    coverage = zeros(size(states));
    for i = 1:numel(states)
        coverage(i) = mean(z == states(i));
    end
    d = struct( ...
        'mean_dwell_ms', mean(dwell_ms), ...
        'median_dwell_ms', median(dwell_ms), ...
        'p05_dwell_ms', prctile(dwell_ms, 5), ...
        'p95_dwell_ms', prctile(dwell_ms, 95), ...
        'n_segments', numel(run_lengths), ...
        'state_coverage', coverage, ...
        'max_state_coverage', max(coverage), ...
        'coverage_entropy', -sum(coverage .* log(coverage + eps)) / log(max(numel(coverage), 2)));
end

function checks = build_checks(report)
    checks = struct('name', {}, 'status', {}, 'value', {}, 'message', {});
    checks(end+1) = make_check('finite_signal', ~report.summary.has_nan_or_inf, 'FAIL', double(~report.summary.has_nan_or_inf), 'Signal contains NaN/Inf.');
    checks(end+1) = make_range_check('channel_count', report.summary.n_channels, [19 256], 'WARN', 'Unusual EEG channel count.');
    checks(end+1) = make_range_check('duration_s', report.summary.duration_s, [30 Inf], 'WARN', 'Very short simulations may not represent resting EEG dynamics.');
    checks(end+1) = make_range_check('median_channel_rms_uv', report.amplitude.median_channel_rms_uv, [1 150], 'WARN', 'Median channel RMS is outside a broad EEG-like range.');
    checks(end+1) = make_range_check('gfp_median_uv', report.amplitude.gfp_median_uv, [0.5 80], 'WARN', 'Median GFP is outside a broad EEG-like range.');
    checks(end+1) = make_range_check('peak_abs_uv', report.amplitude.peak_abs_uv, [5 1000], 'WARN', 'Peak amplitude is implausibly small or extremely large.');
    checks(end+1) = make_range_check('power_slope_2_40hz', report.spectrum.power_slope_2_40hz, [-3.5 -0.2], 'WARN', 'Power spectrum does not look broadly 1/f-like.');
    if isfield(report.microstate_dynamics, 'mean_dwell_ms')
        checks(end+1) = make_range_check('mean_dwell_ms', report.microstate_dynamics.mean_dwell_ms, [40 160], 'WARN', 'Mean dwell time is outside a typical microstate range.');
        checks(end+1) = make_range_check('max_state_coverage', report.microstate_dynamics.max_state_coverage, [0 0.65], 'WARN', 'One state dominates the simulation.');
    end
    if isfield(report.maps, 'max_abs_between_map_correlation')
        checks(end+1) = make_range_check('max_between_map_correlation', report.maps.max_abs_between_map_correlation, [0 0.98], 'WARN', 'At least two true maps are nearly redundant.');
    end
end

function c = make_check(name, pass, fail_status, value, message)
    if pass
        status = 'PASS';
        msg = '';
    else
        status = fail_status;
        msg = message;
    end
    c = struct('name', name, 'status', status, 'value', value, 'message', msg);
end

function c = make_range_check(name, value, bounds, fail_status, message)
    pass = isfinite(value) && value >= bounds(1) && value <= bounds(2);
    c = make_check(name, pass, fail_status, value, message);
end

function print_report(report)
    fprintf('\nSimulated EEG representativeness QC: %s\n', report.overall_status);
    fprintf('  Channels=%d, duration=%.1fs, RMS=%.2f uV, GFP median=%.2f uV\n', ...
        report.summary.n_channels, report.summary.duration_s, ...
        report.amplitude.median_channel_rms_uv, report.amplitude.gfp_median_uv);
    fprintf('  PSD slope 2-40 Hz=%.2f\n', report.spectrum.power_slope_2_40hz);
    if isfield(report.microstate_dynamics, 'mean_dwell_ms')
        fprintf('  Mean dwell=%.1f ms, max coverage=%.2f\n', ...
            report.microstate_dynamics.mean_dwell_ms, report.microstate_dynamics.max_state_coverage);
    end
    for i = 1:numel(report.checks)
        if ~strcmp(report.checks(i).status, 'PASS')
            fprintf('  %s: %s (value=%.4g) %s\n', report.checks(i).status, ...
                report.checks(i).name, report.checks(i).value, report.checks(i).message);
        end
    end
end

function write_report_text(report, output_file)
    fid = fopen(output_file, 'w');
    if fid < 0
        error('Could not open QC output file: %s', output_file);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'Simulated EEG representativeness QC: %s\n', report.overall_status);
    fprintf(fid, 'Channels: %d\nDuration_s: %.3f\nSampling_Hz: %.3f\n', ...
        report.summary.n_channels, report.summary.duration_s, report.summary.sfreq);
    fprintf(fid, 'Median_channel_RMS_uV: %.6g\nGFP_median_uV: %.6g\nPSD_slope_2_40Hz: %.6g\n', ...
        report.amplitude.median_channel_rms_uv, report.amplitude.gfp_median_uv, ...
        report.spectrum.power_slope_2_40hz);
    if isfield(report.microstate_dynamics, 'mean_dwell_ms')
        fprintf(fid, 'Mean_dwell_ms: %.6g\nMax_state_coverage: %.6g\n', ...
            report.microstate_dynamics.mean_dwell_ms, report.microstate_dynamics.max_state_coverage);
    end
    fprintf(fid, '\nChecks:\n');
    for i = 1:numel(report.checks)
        fprintf(fid, '%s\t%s\t%.6g\t%s\n', report.checks(i).status, ...
            report.checks(i).name, report.checks(i).value, report.checks(i).message);
    end
end
