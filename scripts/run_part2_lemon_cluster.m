function output_dir = run_part2_lemon_cluster(varargin)
%RUN_PART2_LEMON_CLUSTER Cluster entry point for the LEMON dataset pipeline.

    util = microstate_utilities();
    default_manifest = fullfile(util.project_root(), 'data', 'lemon', 'lemon_manifest.csv');
    default_output = fullfile(util.project_root(), 'outputs', 'cluster_runs', 'part2_lemon');

    p = inputParser;
    addParameter(p, 'manifest_csv', default_manifest, @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', default_output, @(x) ischar(x) || isstring(x));
    addParameter(p, 'method', 'spm_vb', @(x) ischar(x) || isstring(x));
    addParameter(p, 'criterion', 'elbow_sil_combined', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', 2:10, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'meta_K_candidates', 2:10, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'pooled_K_candidates', 2:10, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'n_initialisations', 100, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'max_iter', 500, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'tol', 1e-7, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'max_maps_per_file', 2000, @(x) isnumeric(x) && isscalar(x) && x >= 100);
    addParameter(p, 'max_global_maps', 60000, @(x) isnumeric(x) && isscalar(x) && x >= 1000);
    addParameter(p, 'gfp_peak_quantile_schedule', [0.50 0.60 0.70 0.80 0.90 0.95], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'save_json', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'run_tanova', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'tanova_permutations', 5000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'n_threads', detect_thread_count(), @(x) isnumeric(x) && isscalar(x) && x >= 1);
    parse(p, varargin{:});
    cfg = p.Results;

    manifest_csv = char(cfg.manifest_csv);
    output_dir = char(cfg.output_dir);
    util.ensure_dir(output_dir);
    log_dir = fullfile(output_dir, 'logs');
    util.ensure_dir(log_dir);
    success_marker = fullfile(output_dir, 'PART2_CLUSTER_RUN_PASSED.txt');
    failure_marker = fullfile(output_dir, 'PART2_CLUSTER_RUN_FAILED.txt');
    if isfile(success_marker), delete(success_marker); end
    if isfile(failure_marker), delete(failure_marker); end
    set_thread_cap(cfg.n_threads);

    diary_file = fullfile(log_dir, ['part2_matlab_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
    diary(diary_file);
    cleanup_diary = onCleanup(@() diary('off')); %#ok<NASGU>

    fprintf('\n========================================\n');
    fprintf('Part 2: LEMON meta-microstate pipeline\n');
    fprintf('========================================\n');
    fprintf('Manifest: %s\n', manifest_csv);
    fprintf('Output dir: %s\n', output_dir);
    fprintf('MATLAB diary: %s\n', diary_file);
    fprintf('K candidates: %s\n', mat2str(cfg.K_candidates));
    fprintf('Meta K candidates: %s\n', mat2str(cfg.meta_K_candidates));
    fprintf('Pooled K candidates: %s\n', mat2str(cfg.pooled_K_candidates));
    fprintf('n_initialisations: %d\n', cfg.n_initialisations);
    fprintf('Workers/threads target: %d\n', cfg.n_threads);
    fprintf('Run TANOVA: %s\n', tf_local(cfg.run_tanova));
    fprintf('========================================\n\n');

    try
        metamicrostate_dataset_pipeline(manifest_csv, ...
            'output_dir', output_dir, ...
            'method', cfg.method, ...
            'criterion', cfg.criterion, ...
            'K_candidates', cfg.K_candidates, ...
            'meta_K_candidates', cfg.meta_K_candidates, ...
            'pooled_K_candidates', cfg.pooled_K_candidates, ...
            'n_initialisations', cfg.n_initialisations, ...
            'max_iter', cfg.max_iter, ...
            'tol', cfg.tol, ...
            'max_maps_per_file', cfg.max_maps_per_file, ...
            'max_global_maps', cfg.max_global_maps, ...
            'gfp_peak_quantile_schedule', cfg.gfp_peak_quantile_schedule, ...
            'save_json', cfg.save_json, ...
            'verbose', true);

        results_mat = fullfile(output_dir, 'hierarchical_microstate_results.mat');
        assert(isfile(results_mat), 'Part 2 run did not produce %s', results_mat);

        if cfg.run_tanova
            tanova_dir = fullfile(output_dir, 'tanova');
            run_microstate_hierarchical_tanova(results_mat, ...
                'output_dir', tanova_dir, ...
                'n_permutations', cfg.tanova_permutations, ...
                'save_plots', true, ...
                'verbose', true);
            tanova_csv = fullfile(tanova_dir, 'tanova_results.csv');
            assert(isfile(tanova_csv), 'Part 2 TANOVA did not produce %s', tanova_csv);
        end

        fid = fopen(success_marker, 'w');
        if fid >= 0
            fprintf(fid, 'Part 2 cluster run passed at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf(fid, 'MATLAB diary: %s\n', diary_file);
            fprintf(fid, 'Results MAT: %s\n', results_mat);
            if cfg.run_tanova
                fprintf(fid, 'TANOVA CSV: %s\n', fullfile(output_dir, 'tanova', 'tanova_results.csv'));
            end
            fclose(fid);
        end
        fprintf('Wrote success marker: %s\n', success_marker);
    catch ME
        fid = fopen(failure_marker, 'w');
        if fid >= 0
            fprintf(fid, 'Part 2 cluster run failed at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf(fid, 'Message: %s\n', ME.message);
            fprintf(fid, 'Identifier: %s\n', ME.identifier);
            fprintf(fid, 'MATLAB diary: %s\n', diary_file);
            fclose(fid);
        end
        fprintf(2, 'Part 2 cluster run failed. Wrote failure marker: %s\n', failure_marker);
        rethrow(ME);
    end
end

function n = detect_thread_count()
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
