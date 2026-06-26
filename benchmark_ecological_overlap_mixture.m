function T = benchmark_ecological_overlap_mixture()
%BENCHMARK_ECOLOGICAL_OVERLAP_MIXTURE Check mixture recovery on ecological overlap data.

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
    seeds = 31:35;
    snrs = [20 6];
    rows = table();
    for snr_db = snrs
        for seed = seeds
            opts = struct( ...
                'ecological_profile', true, ...
                'inject_artifacts', false, ...
                'prob', 0.5, ...
                'ms_range', [10 40], ...
                'strength', 0.5, ...
                'map_jitter_fraction', 0.02, ...
                'template_pool_K', 7);
            Sim = generate_microstate_eeg(K, snr_db, 20, 250, seed, opts);
            Results = fit_microstate_kmeans_koenig(Sim, K, 'silhouette');
            backfit = backfit_microstate_timecourse(Sim, Results);

            est_to_true = estimated_to_true_state_map_local(Sim.maps_true, Results.centers, util);
            true_w = double(Sim.state_weights_true');
            hard_w = project_weights_local(backfit.hard.weights, est_to_true, K);
            mix_w = project_weights_local(backfit.mixture.weights, est_to_true, K);
            overlap_mask = sum(true_w > 0.05, 2) > 1;
            peak_mix = 0;
            if isfield(backfit.mixture, 'peak_mixture_samples')
                peak_mix = numel(backfit.mixture.peak_mixture_samples);
            end

            rows = [rows; table( ... %#ok<AGROW>
                seed, snr_db, Results.recovery_metrics.mean_recovery_matched, ...
                score_top1(hard_w, true_w), score_top1(mix_w, true_w), mean(abs(mix_w - true_w), 'all'), ...
                score_top1(mix_w(overlap_mask, :), true_w(overlap_mask, :)), mean(abs(mix_w(overlap_mask, :) - true_w(overlap_mask, :)), 'all'), ...
                pair_score(mix_w(overlap_mask, :), true_w(overlap_mask, :)), peak_mix, string(backfit.mixture.mode), ...
                'VariableNames', {'seed', 'snr_db', 'map_mean_r', 'hard_acc', 'mix_acc', 'mix_mae', ...
                'mix_overlap_acc', 'mix_overlap_mae', 'mix_overlap_pair_acc', 'peak_mixture_samples', 'mix_mode'})];
        end
    end

    T = rows;
    disp(rows);
    fprintf('\nMeans by SNR:\n');
    disp(groupsummary(rows, 'snr_db', 'mean', ...
        {'map_mean_r', 'hard_acc', 'mix_acc', 'mix_mae', 'mix_overlap_acc', 'mix_overlap_mae', 'mix_overlap_pair_acc', 'peak_mixture_samples'}));
    if ~exist('outputs', 'dir')
        mkdir('outputs');
    end
    writetable(rows, fullfile('outputs', 'ecological_overlap_mixture_benchmark.csv'));
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

function acc = pair_score(pred_w, true_w)
    if isempty(pred_w)
        acc = NaN;
        return;
    end
    [~, pred] = sort(pred_w, 2, 'descend');
    [~, truth] = sort(true_w, 2, 'descend');
    acc = mean(all(sort(pred(:, 1:2), 2) == sort(truth(:, 1:2), 2), 2));
end
