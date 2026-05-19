function report = diagnose_real_eeg_nans(eeg_file)
% DIAGNOSE_REAL_EEG_NANS Inspect NaN/Inf distribution in an EEGLAB real EEG file.

    if nargin < 1
        eeg_file = 'E:\EEGs\TEST_EEG\sub_001_j_raw.set';
    end
    addpath('E:\EEGs\SPM_MS');

    if ~exist('pop_loadset', 'file')
        error('EEGLAB pop_loadset is not on the MATLAB path.');
    end
    EEG = pop_loadset('filename', eeg_file);
    X = double(EEG.data);

    nan_mask = isnan(X);
    inf_mask = isinf(X);
    bad_mask = ~isfinite(X);
    bad_by_channel = sum(bad_mask, 2);
    bad_by_sample = sum(bad_mask, 1);
    channel_has_bad = bad_by_channel > 0;
    sample_has_bad = bad_by_sample > 0;

    labels = channel_labels_from_eeg(EEG);
    bad_channel_idx = find(channel_has_bad);
    bad_sample_idx = find(sample_has_bad);
    bad_runs = contiguous_runs(bad_sample_idx);

    report = struct();
    report.file = eeg_file;
    report.nbchan = EEG.nbchan;
    report.pnts = EEG.pnts;
    report.srate = EEG.srate;
    report.duration_s = EEG.pnts / EEG.srate;
    report.total_nan = sum(nan_mask(:));
    report.total_inf = sum(inf_mask(:));
    report.total_bad = sum(bad_mask(:));
    report.finite_fraction = mean(isfinite(X(:)));
    report.n_bad_channels = numel(bad_channel_idx);
    report.n_bad_samples = numel(bad_sample_idx);
    report.bad_channel_idx = bad_channel_idx;
    report.bad_channel_labels = labels(bad_channel_idx);
    report.bad_count_by_bad_channel = bad_by_channel(bad_channel_idx);
    report.bad_runs = bad_runs;

    fprintf('\nNaN/Inf diagnostics for %s\n', eeg_file);
    fprintf('  %d channels, %d samples, %.3f Hz, %.1f seconds\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate, report.duration_s);
    fprintf('  NaN=%d, Inf=%d, finite fraction=%.6f\n', ...
        report.total_nan, report.total_inf, report.finite_fraction);
    fprintf('  Bad channels=%d/%d, bad samples=%d/%d\n', ...
        report.n_bad_channels, EEG.nbchan, report.n_bad_samples, EEG.pnts);

    if ~isempty(bad_channel_idx)
        fprintf('\n  Channels containing NaN/Inf:\n');
        n_show = min(30, numel(bad_channel_idx));
        for i = 1:n_show
            idx = bad_channel_idx(i);
            fprintf('    %3d %-12s bad=%d\n', idx, labels{idx}, bad_by_channel(idx));
        end
        if numel(bad_channel_idx) > n_show
            fprintf('    ... %d more channels\n', numel(bad_channel_idx) - n_show);
        end
    end

    if ~isempty(bad_runs)
        fprintf('\n  Bad sample runs:\n');
        n_show = min(20, size(bad_runs, 1));
        for i = 1:n_show
            fprintf('    samples %d-%d (%.3f-%.3f s), len=%d\n', ...
                bad_runs(i, 1), bad_runs(i, 2), ...
                (bad_runs(i, 1)-1)/EEG.srate, (bad_runs(i, 2)-1)/EEG.srate, ...
                bad_runs(i, 3));
        end
        if size(bad_runs, 1) > n_show
            fprintf('    ... %d more runs\n', size(bad_runs, 1) - n_show);
        end
    end

    if isfield(EEG, 'event') && ~isempty(EEG.event)
        fprintf('\n  Boundary-like events near bad runs:\n');
        print_nearby_events(EEG, bad_runs);
    end
end

function labels = channel_labels_from_eeg(EEG)
    labels = cell(EEG.nbchan, 1);
    for i = 1:EEG.nbchan
        if isfield(EEG.chanlocs, 'labels') && numel(EEG.chanlocs) >= i && ~isempty(EEG.chanlocs(i).labels)
            labels{i} = char(EEG.chanlocs(i).labels);
        else
            labels{i} = sprintf('Ch%03d', i);
        end
    end
end

function runs = contiguous_runs(idx)
    if isempty(idx)
        runs = [];
        return;
    end
    breaks = [1, find(diff(idx) > 1) + 1, numel(idx) + 1];
    runs = zeros(numel(breaks) - 1, 3);
    for r = 1:(numel(breaks) - 1)
        start_i = idx(breaks(r));
        end_i = idx(breaks(r + 1) - 1);
        runs(r, :) = [start_i, end_i, end_i - start_i + 1];
    end
end

function print_nearby_events(EEG, bad_runs)
    if isempty(bad_runs)
        fprintf('    no bad runs\n');
        return;
    end
    latencies = nan(numel(EEG.event), 1);
    types = cell(numel(EEG.event), 1);
    for e = 1:numel(EEG.event)
        if isfield(EEG.event, 'latency') && ~isempty(EEG.event(e).latency)
            latencies(e) = double(EEG.event(e).latency);
        end
        if isfield(EEG.event, 'type')
            types{e} = char(string(EEG.event(e).type));
        else
            types{e} = '';
        end
    end
    n_printed = 0;
    for r = 1:min(10, size(bad_runs, 1))
        lo = bad_runs(r, 1) - EEG.srate;
        hi = bad_runs(r, 2) + EEG.srate;
        nearby = find(latencies >= lo & latencies <= hi);
        for ii = nearby(:)'
            n_printed = n_printed + 1;
            fprintf('    event %-16s latency %.1f (%.3f s), near bad run %d-%d\n', ...
                types{ii}, latencies(ii), (latencies(ii)-1)/EEG.srate, bad_runs(r,1), bad_runs(r,2));
            if n_printed >= 30
                return;
            end
        end
    end
    if n_printed == 0
        fprintf('    none within +/- 1 second of the first bad runs\n');
    end
end
