function T = benchmark_ecological_backfit()
%BENCHMARK_ECOLOGICAL_BACKFIT Compare classic and all-sample backfit on ecological simulations.

    root = fileparts(mfilename('fullpath'));
    addpath(root);
    addpath(fullfile(root, 'Koenig_code'));
    util = microstate_utilities();
    cfg = util.load_config();
    if isfield(cfg.paths, 'eeglab_dir') && exist(char(cfg.paths.eeglab_dir), 'dir')
        addpath(char(cfg.paths.eeglab_dir));
    end
    if exist('eeglab', 'file') == 2
        eeglab nogui;
    end

    K = 4;
    seeds = 21:25;
    snrs = [20 6];
    rows = table();
    for snr_db = snrs
        for seed = seeds
            opts = struct( ...
                'ecological_profile', true, ...
                'inject_artifacts', false, ...
                'prob', 0, ...
                'map_jitter_fraction', 0.02, ...
                'template_pool_K', 7);
            Sim = generate_microstate_eeg(K, snr_db, 20, 250, seed, opts);
            Results = fit_microstate_kmeans_koenig(Sim, K, 'silhouette');

            classic_sim = rmfield_if_present(Sim, {'z_true', 'maps_true', 'state_weights_true'});
            classic = backfit_microstate_timecourse(classic_sim, Results);
            modern = backfit_microstate_timecourse(Sim, Results);

            est_to_true = estimated_to_true_state_map_local(Sim.maps_true, Results.centers, util);
            true_w = double(Sim.state_weights_true');
            classic_w = project_weights_local(classic.hard.weights, est_to_true, K);
            modern_hard_w = project_weights_local(modern.hard.weights, est_to_true, K);
            modern_mix_w = project_weights_local(modern.mixture.weights, est_to_true, K);
            peak_mask = false(size(true_w, 1), 1);
            peak_mask(Results.idx_peaks(:)) = true;

            rows = [rows; table( ... %#ok<AGROW>
                seed, snr_db, size(Results.maps_nc, 1), ...
                Results.recovery_metrics.mean_recovery_matched, ...
                Results.recovery_metrics.f1_score, ...
                score_top1(classic_w, true_w), mean(abs(classic_w - true_w), 'all'), ...
                score_top1(classic_w(peak_mask, :), true_w(peak_mask, :)), ...
                score_top1(classic_w(~peak_mask, :), true_w(~peak_mask, :)), ...
                score_top1(modern_hard_w, true_w), mean(abs(modern_hard_w - true_w), 'all'), ...
                score_top1(modern_hard_w(peak_mask, :), true_w(peak_mask, :)), ...
                score_top1(modern_hard_w(~peak_mask, :), true_w(~peak_mask, :)), ...
                score_top1(modern_mix_w, true_w), mean(abs(modern_mix_w - true_w), 'all'), ...
                string(classic.hard.mode), string(modern.hard.mode), string(modern.mixture.mode), ...
                'VariableNames', {'seed', 'snr_db', 'n_gfp_peaks', ...
                'map_mean_r', 'map_f1', ...
                'classic_interp_acc', 'classic_interp_mae', ...
                'classic_interp_peak_acc', 'classic_interp_nonpeak_acc', ...
                'modern_hard_acc', 'modern_hard_mae', ...
                'modern_hard_peak_acc', 'modern_hard_nonpeak_acc', ...
                'modern_mix_acc', 'modern_mix_mae', ...
                'classic_mode', 'modern_hard_mode', 'modern_mix_mode'})];
        end
    end

    T = rows;
    disp(rows);
    fprintf('\nMeans by SNR:\n');
    disp(groupsummary(rows, 'snr_db', 'mean', ...
        {'map_mean_r', 'map_f1', 'classic_interp_acc', 'classic_interp_mae', ...
        'classic_interp_peak_acc', 'classic_interp_nonpeak_acc', ...
        'modern_hard_acc', 'modern_hard_mae', ...
        'modern_hard_peak_acc', 'modern_hard_nonpeak_acc', ...
        'modern_mix_acc', 'modern_mix_mae'}));
end

function s = rmfield_if_present(s, fields)
    for i = 1:numel(fields)
        if isfield(s, fields{i})
            s = rmfield(s, fields{i});
        end
    end
end

function est_to_true = estimated_to_true_state_map_local(true_maps, estimated_maps, util)
    sim = abs(util.normalize_maps(double(estimated_maps)) * util.normalize_maps(double(true_maps))');
    [~, est_to_true] = max(sim, [], 2);
end

function weights_true = project_weights_local(weights_est, est_to_true, K_true)
    weights_true = zeros(size(weights_est, 1), K_true);
    for k = 1:numel(est_to_true)
        weights_true(:, est_to_true(k)) = weights_true(:, est_to_true(k)) + weights_est(:, k);
    end
    weights_true = weights_true ./ max(sum(weights_true, 2), eps);
end

function acc = score_top1(pred_w, true_w)
    [~, pred] = max(pred_w, [], 2);
    [~, truth] = max(true_w, [], 2);
    acc = mean(pred == truth);
end
