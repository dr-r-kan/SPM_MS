function output_dir = run_part1_simulated_cluster(varargin)
%RUN_PART1_SIMULATED_CLUSTER Cluster entry point for the extensive simulation study.

    util = microstate_utilities();
    default_output = fullfile(util.project_root(), 'outputs', 'cluster_runs', 'part1_simulated_extensive');

    p = inputParser;
    addParameter(p, 'output_dir', default_output, @(x) ischar(x) || isstring(x));
    addParameter(p, 'reps', 24, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'K_true_vals', 3:8, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'SNR_dbs', [-12 -9 -6 -3 0 3 6], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'K_candidates', 2:10, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'duration_s', 300, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'sfreq', 250, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'montages', {'full', '10-20-20', '10-20-12'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'overlap_probs', [0 0.25 0.5 0.75 1.0], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'overlap_ms_range', [10 40], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'overlap_strength', 0.5, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'n_workers', detect_worker_count(), @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'save_json', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'run_summary_analysis', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'summary_boot', 2000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'summary_boot_lmm', 1000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'summary_folds', 10, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    parse(p, varargin{:});
    cfg = p.Results;

    output_dir = char(cfg.output_dir);
    util.ensure_dir(output_dir);
    set_thread_cap(cfg.n_workers);

    fprintf('\n========================================\n');
    fprintf('Part 1: extensive simulated benchmark\n');
    fprintf('========================================\n');
    fprintf('Output dir: %s\n', output_dir);
    fprintf('Repetitions: %d\n', cfg.reps);
    fprintf('K true: %s\n', mat2str(cfg.K_true_vals));
    fprintf('SNRs: %s\n', mat2str(cfg.SNR_dbs));
    fprintf('Montages: %s\n', strjoin(cellstr(string(cfg.montages)), ', '));
    fprintf('Overlap probs: %s\n', mat2str(cfg.overlap_probs));
    fprintf('Workers/threads target: %d\n', cfg.n_workers);
    fprintf('Save JSON: %s\n', tf_local(cfg.save_json));
    fprintf('========================================\n\n');

    simulated_ms_retrieval_experiment( ...
        'out_dir', output_dir, ...
        'reps', cfg.reps, ...
        'K_true_vals', cfg.K_true_vals, ...
        'SNR_dbs', cfg.SNR_dbs, ...
        'K_candidates', cfg.K_candidates, ...
        'duration_s', cfg.duration_s, ...
        'sfreq', cfg.sfreq, ...
        'montages', cellstr(string(cfg.montages)), ...
        'overlap_probs', cfg.overlap_probs, ...
        'overlap_ms_range', cfg.overlap_ms_range, ...
        'overlap_strength', cfg.overlap_strength, ...
        'n_workers', cfg.n_workers, ...
        'save_json', cfg.save_json, ...
        'verbose', true);

    results_dir = fullfile(output_dir, 'results');
    if cfg.run_summary_analysis
        analyze_comparison_results(results_dir, ...
            'n_boot', cfg.summary_boot, ...
            'n_boot_lmm', cfg.summary_boot_lmm, ...
            'n_folds', cfg.summary_folds);
    end
end

function n = detect_worker_count()
    txt = getenv('NSLOTS');
    n = str2double(txt);
    if ~isfinite(n) || isnan(n) || n < 1
        n = 4;
    end
end

function set_thread_cap(n)
    try
        maxNumCompThreads(max(1, floor(n)));
    catch
    end
end

function s = tf_local(x)
    if x
        s = 'true';
    else
        s = 'false';
    end
end
