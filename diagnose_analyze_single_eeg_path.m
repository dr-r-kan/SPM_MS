function report = diagnose_analyze_single_eeg_path(eeg_file)
% DIAGNOSE_ANALYZE_SINGLE_EEG_PATH Trace NaNs through analyze_single_eeg_file preprocessing.

    if nargin < 1
        eeg_file = 'E:\EEGs\TEST_EEG\sub_001_j_raw.set';
    end
    addpath('E:\EEGs\SPM_MS');

    EEG = pop_loadset('filename', eeg_file);
    Sim = struct();
    Sim.X_noisy = double(EEG.data);
    Sim.X_clean = Sim.X_noisy;
    Sim.sfreq = EEG.srate;
    Sim.duration_s = EEG.pnts / EEG.srate;
    Sim.n_channels = EEG.nbchan;
    Sim.n_samples = EEG.pnts;
    Sim.maps_true = [];
    Sim.K_true = NaN;
    Sim.SNR_dB = NaN;

    util = microstate_utilities_SHARED();
    X = Sim.X_noisy;
    X_bp = util.bandpass_filter(X, Sim.sfreq, [2 20]);
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original] = util.preprocess_maps(Sim);

    report = struct();
    report.raw = finite_summary(X);
    report.bandpassed = finite_summary(X_bp);
    report.gfp = finite_summary(gfp_vec);
    report.maps_original = finite_summary(maps_original);
    report.maps_norm = finite_summary(maps_norm);
    report.n_peaks = numel(idx_peaks);
    report.n_maps = n_maps;
    report.C_dims = C_dims;
    report.zero_or_tiny_gfp_samples = sum(gfp_vec < eps);
    report.channels_zero_sd_raw = sum(std(X, 0, 2) < eps);
    report.channels_zero_sd_bp = sum(std(X_bp, 0, 2) < eps);
    report.channels_nan_after_bp = find(any(~isfinite(X_bp), 2));
    report.maps_nan_rows = find(any(~isfinite(maps_norm), 2));
    report.first_peak_indices = idx_peaks(1:min(20, numel(idx_peaks)));

    fprintf('\nAnalyze path diagnostics for %s\n', eeg_file);
    print_summary('raw', report.raw);
    print_summary('bandpassed', report.bandpassed);
    print_summary('gfp', report.gfp);
    print_summary('maps_original', report.maps_original);
    print_summary('maps_norm', report.maps_norm);
    fprintf('  n_peaks=%d, n_maps=%d, C_dims=%d\n', report.n_peaks, report.n_maps, report.C_dims);
    fprintf('  zero/tiny GFP samples=%d\n', report.zero_or_tiny_gfp_samples);
    fprintf('  zero-SD channels raw=%d, bandpassed=%d\n', report.channels_zero_sd_raw, report.channels_zero_sd_bp);
    if ~isempty(report.maps_nan_rows)
        fprintf('  first NaN map rows: %s\n', mat2str(report.maps_nan_rows(1:min(20,end))));
    end
end

function s = finite_summary(X)
    s = struct( ...
        'size', size(X), ...
        'nan', sum(isnan(X(:))), ...
        'inf', sum(isinf(X(:))), ...
        'finite_fraction', mean(isfinite(X(:))), ...
        'min', min(X(:), [], 'omitnan'), ...
        'max', max(X(:), [], 'omitnan'), ...
        'mean', mean(X(:), 'omitnan'), ...
        'std', std(X(:), 'omitnan'));
end

function print_summary(name, s)
    fprintf('  %-14s size=%s NaN=%d Inf=%d finite=%.6f min=%.4g max=%.4g std=%.4g\n', ...
        name, mat2str(s.size), s.nan, s.inf, s.finite_fraction, s.min, s.max, s.std);
end
