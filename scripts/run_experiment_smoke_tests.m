function run_experiment_smoke_tests(varargin)
%RUN_EXPERIMENT_SMOKE_TESTS Quick environment test for both requested parts.

    util = microstate_utilities();
    default_manifest = fullfile(util.project_root(), 'data', 'lemon', 'lemon_manifest.csv');
    default_root = fullfile(util.project_root(), 'outputs', 'smoke_test');

    p = inputParser;
    addParameter(p, 'manifest_csv', default_manifest, @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_root', default_root, @(x) ischar(x) || isstring(x));
    addParameter(p, 'max_lemon_files', 2, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    parse(p, varargin{:});
    cfg = p.Results;

    output_root = char(cfg.output_root);
    util.ensure_dir(output_root);
    log_dir = fullfile(output_root, 'logs');
    util.ensure_dir(log_dir);
    success_marker = fullfile(output_root, 'SMOKE_TEST_PASSED.txt');
    failure_marker = fullfile(output_root, 'SMOKE_TEST_FAILED.txt');
    if isfile(success_marker), delete(success_marker); end
    if isfile(failure_marker), delete(failure_marker); end
    try
        maxNumCompThreads(1);
    catch
    end

    diary_file = fullfile(log_dir, ['smoke_test_matlab_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
    diary(diary_file);
    cleanup_diary = onCleanup(@() diary('off')); %#ok<NASGU>

    fprintf('\n========================================\n');
    fprintf('Smoke tests for both experiment parts\n');
    fprintf('========================================\n');
    fprintf('Output root: %s\n', output_root);
    fprintf('MATLAB diary: %s\n', diary_file);
    fprintf('Manifest CSV: %s\n', char(cfg.manifest_csv));
    fprintf('========================================\n');

    try
        run_smoke_tests_inner(cfg, output_root);
        fid = fopen(success_marker, 'w');
        if fid >= 0
            fprintf(fid, 'Smoke test passed at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf(fid, 'MATLAB diary: %s\n', diary_file);
            fclose(fid);
        end
        fprintf('Wrote success marker: %s\n', success_marker);
    catch ME
        fid = fopen(failure_marker, 'w');
        if fid >= 0
            fprintf(fid, 'Smoke test failed at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf(fid, 'Message: %s\n', ME.message);
            fprintf(fid, 'Identifier: %s\n', ME.identifier);
            fprintf(fid, 'MATLAB diary: %s\n', diary_file);
            fclose(fid);
        end
        fprintf(2, 'Smoke test failed. Wrote failure marker: %s\n', failure_marker);
        rethrow(ME);
    end
end

function run_smoke_tests_inner(cfg, output_root)
    util = microstate_utilities();
    part1_dir = fullfile(output_root, 'part1_simulated');
    run_part1_simulated_cluster( ...
        'output_dir', part1_dir, ...
        'reps', 1, ...
        'K_true_vals', 4, ...
        'SNR_dbs', 0, ...
        'K_candidates', 4:5, ...
        'duration_s', 20, ...
        'montages', {'full'}, ...
        'overlap_probs', 0, ...
        'n_workers', 1, ...
        'save_json', false, ...
        'run_summary_analysis', false);

    part1_csv = fullfile(part1_dir, 'results', 'comparison_results.csv');
    assert(isfile(part1_csv), 'Part 1 smoke test did not produce %s', part1_csv);
    fprintf('Part 1 smoke test passed: %s\n', part1_csv);

    manifest_csv = char(cfg.manifest_csv);
    if ~isfile(manifest_csv)
        error('Part 2 smoke test requires a manifest CSV. Missing: %s', manifest_csv);
    end
    T = readtable(manifest_csv, 'TextType', 'string');
    assert(height(T) >= cfg.max_lemon_files, 'Manifest needs at least %d rows for the smoke test.', cfg.max_lemon_files);
    T = T(1:cfg.max_lemon_files, :);
    part2_dir = fullfile(output_root, 'part2_lemon');
    util.ensure_dir(part2_dir);
    smoke_manifest = fullfile(part2_dir, 'lemon_smoke_manifest.csv');
    writetable(T, smoke_manifest);

    metamicrostate_dataset_pipeline(smoke_manifest, ...
        'output_dir', part2_dir, ...
        'K_candidates', 4:5, ...
        'meta_K_candidates', 4:5, ...
        'pooled_K_candidates', 4:5, ...
        'n_initialisations', 3, ...
        'max_iter', 50, ...
        'max_maps_per_file', 100, ...
        'max_global_maps', 500, ...
        'gfp_peak_quantile_schedule', [0.70 0.85], ...
        'save_json', false, ...
        'verbose', true);

    part2_mat = fullfile(part2_dir, 'hierarchical_microstate_results.mat');
    assert(isfile(part2_mat), 'Part 2 smoke test did not produce %s', part2_mat);
    fprintf('Part 2 smoke test passed: %s\n', part2_mat);

    fprintf('All smoke tests passed.\n');
end
