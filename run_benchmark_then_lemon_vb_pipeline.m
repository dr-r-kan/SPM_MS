function PipelineResults = run_benchmark_then_lemon_vb_pipeline(varargin)
%RUN_BENCHMARK_THEN_LEMON_VB_PIPELINE Public-facing benchmark + LEMON runner.
%
% This runner performs the shareable three-stage workflow:
%   1. Run the simulated EEG benchmark.
%   2. Analyse the simulation outputs to compare traditional K-means
%      against SPM-VB, and to compare criteria within SPM-VB.
%   3. Run the LEMON dataset pipeline using the SPM-VB approach.
%
% Example:
%   R = run_benchmark_then_lemon_vb_pipeline( ...
%       'manifest_csv', 'conditioned_lemon_sets.csv', ...
%       'output_root', 'outputs/public_release', ...
%       'simulation_args', {'reps', 10});
%
% Key optional overrides:
%   'manifest_csv'              - LEMON manifest CSV with file_path column
%   'output_root'               - Root output folder for all stages
%   'simulation_args'           - Extra name/value args for the simulation
%   'simulation_analysis_args'  - Extra args for analyze_comparison_results
%   'lemon_args'                - Extra args for metamicrostate_dataset_pipeline
%   'run_simulation'            - Enable stage 1
%   'run_simulation_analysis'   - Enable stage 2
%   'run_lemon_vb'              - Enable stage 3
%   'verbose'                   - Print progress

    util = microstate_utilities();
    cfg = util.load_config();
    sim_cfg = util.get_field(cfg, 'simulation', struct());
    hier_cfg = util.get_field(cfg, 'hierarchical', struct());

    p = inputParser;
    addParameter(p, 'manifest_csv', 'conditioned_lemon_sets.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_root', fullfile(char(cfg.paths.simulation_output_dir), 'public_release'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'simulation_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'lemon_output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', char(cfg.paths.template_file), @(x) ischar(x) || isstring(x));
    addParameter(p, 'simulation_args', {}, @iscell);
    addParameter(p, 'simulation_analysis_args', {}, @iscell);
    addParameter(p, 'lemon_args', {}, @iscell);
    addParameter(p, 'run_simulation', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'run_simulation_analysis', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'run_lemon_vb', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'verbose', true, @(x) islogical(x) && isscalar(x));
    parse(p, varargin{:});
    opts = p.Results;

    opts.output_root = util.resolve_path(char(opts.output_root), util.project_root());
    opts.manifest_csv = util.resolve_path(char(opts.manifest_csv), util.project_root());
    opts.template_file = util.resolve_path(char(opts.template_file), util.project_root());
    if isempty(opts.simulation_dir)
        opts.simulation_dir = fullfile(opts.output_root, 'simulation_benchmark');
    end
    if isempty(opts.lemon_output_dir)
        opts.lemon_output_dir = fullfile(opts.output_root, 'lemon_spm_vb');
    end
    opts.simulation_dir = util.resolve_path(char(opts.simulation_dir), util.project_root());
    opts.lemon_output_dir = util.resolve_path(char(opts.lemon_output_dir), util.project_root());

    util.ensure_dir(opts.output_root);
    util.ensure_dir(opts.simulation_dir);
    if opts.run_lemon_vb
        util.ensure_dir(opts.lemon_output_dir);
    end

    if opts.run_lemon_vb && ~isfile(opts.manifest_csv)
        error('Manifest CSV not found: %s', opts.manifest_csv);
    end
    if ~isfile(opts.template_file)
        error('Template file not found: %s', opts.template_file);
    end

    sim_args = opts.simulation_args;
    sim_args = local_set_default_arg(sim_args, 'out_dir', opts.simulation_dir);
    sim_args = local_set_default_arg(sim_args, 'methods', {'spm_vb', 'kmeans_koenig'});
    sim_args = local_set_default_arg(sim_args, 'criteria', { ...
        'silhouette', 'free_energy', 'free_energy_elbow', 'gev', ...
        'calinski_harabasz_score', 'covariance', 'covariance_elbow', ...
        'elbow_sil_combined', 'free_energy_covariance'});
    sim_args = local_set_default_arg(sim_args, 'save_k_candidate_metrics', true);
    sim_args = local_set_default_arg(sim_args, 'verbose', opts.verbose);
    if isfield(sim_cfg, 'clean_sanity_profile')
        sim_args = local_set_default_arg(sim_args, 'clean_sanity_profile', logical(sim_cfg.clean_sanity_profile));
    end
    if isfield(sim_cfg, 'clean_sanity_snr_db_threshold')
        sim_args = local_set_default_arg(sim_args, 'clean_sanity_snr_db_threshold', double(sim_cfg.clean_sanity_snr_db_threshold));
    end

    analyze_args = opts.simulation_analysis_args;

    lemon_args = opts.lemon_args;
    lemon_args = local_set_default_arg(lemon_args, 'output_dir', opts.lemon_output_dir);
    lemon_args = local_set_default_arg(lemon_args, 'method', 'spm_vb');
    lemon_args = local_set_default_arg(lemon_args, 'criterion', local_get_field(hier_cfg, 'criterion', 'elbow_sil_combined'));
    lemon_args = local_set_default_arg(lemon_args, 'K_candidates', local_get_field(hier_cfg, 'K_candidates', 4:7));
    lemon_args = local_set_default_arg(lemon_args, 'template_file', opts.template_file);
    lemon_args = local_set_default_arg(lemon_args, 'verbose', opts.verbose);

    settings_file = fullfile(opts.output_root, 'pipeline_settings.txt');
    local_write_settings_file(settings_file, opts, sim_args, analyze_args, lemon_args);

    PipelineResults = struct();
    PipelineResults.output_root = opts.output_root;
    PipelineResults.simulation_dir = opts.simulation_dir;
    PipelineResults.lemon_output_dir = opts.lemon_output_dir;
    PipelineResults.manifest_csv = opts.manifest_csv;
    PipelineResults.template_file = opts.template_file;
    PipelineResults.settings_file = settings_file;

    step_defs = [ ...
        opts.run_simulation, ...
        opts.run_simulation_analysis, ...
        opts.run_lemon_vb];
    step_total = sum(step_defs);
    step_idx = 0;

    if opts.verbose
        fprintf('\n========================================\n');
        fprintf('Benchmark + LEMON VB pipeline\n');
        fprintf('========================================\n');
        fprintf('Output root:  %s\n', opts.output_root);
        fprintf('Simulation:   %s\n', opts.simulation_dir);
        fprintf('LEMON VB:     %s\n', opts.lemon_output_dir);
        fprintf('Manifest:     %s\n', opts.manifest_csv);
        fprintf('Template:     %s\n', opts.template_file);
        fprintf('========================================\n\n');
    end

    if opts.run_simulation
        step_idx = step_idx + 1;
        fprintf('[%d/%d] Running simulated_ms_retrieval_experiment ...\n', step_idx, step_total);
        t0 = tic;
        SimulationResults = simulated_ms_retrieval_experiment(sim_args{:});
        elapsed = toc(t0);
        PipelineResults.simulation = SimulationResults;
        PipelineResults.simulation_results_dir = fullfile(opts.simulation_dir, 'results');
        PipelineResults.simulation_results_csv = fullfile(PipelineResults.simulation_results_dir, 'comparison_results.csv');
        PipelineResults.simulation_k_candidate_csv = fullfile(PipelineResults.simulation_results_dir, 'k_candidate_metrics.csv');
        fprintf('[%d/%d] Complete in %s\n\n', step_idx, step_total, local_duration_string(elapsed));
    else
        PipelineResults.simulation_results_dir = fullfile(opts.simulation_dir, 'results');
        PipelineResults.simulation_results_csv = fullfile(PipelineResults.simulation_results_dir, 'comparison_results.csv');
        PipelineResults.simulation_k_candidate_csv = fullfile(PipelineResults.simulation_results_dir, 'k_candidate_metrics.csv');
    end

    if opts.run_simulation_analysis
        step_idx = step_idx + 1;
        fprintf('[%d/%d] Running simulation analysis ...\n', step_idx, step_total);
        t0 = tic;
        PipelineResults.simulation_first_line_summary_csv = summarize_first_line_spm_vb_metrics( ...
            PipelineResults.simulation_results_dir, 'verbose', opts.verbose);
        analyze_comparison_results(PipelineResults.simulation_results_dir, analyze_args{:});
        elapsed = toc(t0);
        PipelineResults.simulation_analysis_dir = fullfile(opts.simulation_dir, 'analysis_plots');
        fprintf('[%d/%d] Complete in %s\n\n', step_idx, step_total, local_duration_string(elapsed));
    end

    if opts.run_lemon_vb
        step_idx = step_idx + 1;
        fprintf('[%d/%d] Running metamicrostate_dataset_pipeline (SPM-VB) ...\n', step_idx, step_total);
        t0 = tic;
        [LemonResults, lemon_manifest_out] = metamicrostate_dataset_pipeline(opts.manifest_csv, lemon_args{:});
        elapsed = toc(t0);
        PipelineResults.lemon = LemonResults;
        PipelineResults.lemon_output_manifest_csv = lemon_manifest_out;
        fprintf('[%d/%d] Complete in %s\n\n', step_idx, step_total, local_duration_string(elapsed));
    end

    PipelineResults.pipeline_results_mat = fullfile(opts.output_root, 'pipeline_results.mat');
    save(PipelineResults.pipeline_results_mat, 'PipelineResults', '-v7.3');

    fprintf('Finished benchmark + LEMON VB pipeline.\n');
    fprintf('Settings: %s\n', PipelineResults.settings_file);
    if isfield(PipelineResults, 'simulation_results_csv')
        fprintf('Simulation results: %s\n', PipelineResults.simulation_results_csv);
    end
    if isfield(PipelineResults, 'simulation_analysis_dir')
        fprintf('Simulation analysis: %s\n', PipelineResults.simulation_analysis_dir);
    end
    if isfield(PipelineResults, 'lemon_output_manifest_csv')
        fprintf('LEMON manifest output: %s\n', PipelineResults.lemon_output_manifest_csv);
    end
end

function args = local_set_default_arg(args, name, value)
    if ~iscell(args)
        error('Expected a cell array of name/value arguments.');
    end
    if mod(numel(args), 2) ~= 0
        error('Name/value argument cells must contain an even number of elements.');
    end
    for i = 1:2:numel(args)
        key = args{i};
        if (ischar(key) || isstring(key)) && strcmpi(char(key), name)
            return;
        end
    end
    args(end + 1:end + 2) = {name, value};
end

function value = local_get_field(s, field_name, default_value)
    value = default_value;
    if isstruct(s) && isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    end
end

function txt = local_duration_string(seconds_elapsed)
    seconds_elapsed = max(0, double(seconds_elapsed));
    hours = floor(seconds_elapsed / 3600);
    minutes = floor(mod(seconds_elapsed, 3600) / 60);
    seconds_only = mod(seconds_elapsed, 60);
    if hours > 0
        txt = sprintf('%dh %dm %.1fs', hours, minutes, seconds_only);
    elseif minutes > 0
        txt = sprintf('%dm %.1fs', minutes, seconds_only);
    else
        txt = sprintf('%.1fs', seconds_only);
    end
end

function local_write_settings_file(settings_file, opts, sim_args, analyze_args, lemon_args)
    fid = fopen(settings_file, 'w');
    if fid < 0
        error('Could not open settings file for writing: %s', settings_file);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'Benchmark + LEMON VB pipeline settings\n');
    fprintf(fid, 'Generated: %s\n\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fprintf(fid, 'output_root=%s\n', opts.output_root);
    fprintf(fid, 'simulation_dir=%s\n', opts.simulation_dir);
    fprintf(fid, 'lemon_output_dir=%s\n', opts.lemon_output_dir);
    fprintf(fid, 'manifest_csv=%s\n', opts.manifest_csv);
    fprintf(fid, 'template_file=%s\n', opts.template_file);
    fprintf(fid, 'run_simulation=%d\n', opts.run_simulation);
    fprintf(fid, 'run_simulation_analysis=%d\n', opts.run_simulation_analysis);
    fprintf(fid, 'run_lemon_vb=%d\n', opts.run_lemon_vb);
    fprintf(fid, '\nSimulation args:\n%s\n', local_args_to_text(sim_args));
    fprintf(fid, '\nSimulation analysis args:\n%s\n', local_args_to_text(analyze_args));
    fprintf(fid, '\nLEMON args:\n%s\n', local_args_to_text(lemon_args));
end

function txt = local_args_to_text(args)
    if isempty(args)
        txt = '  <none>';
        return;
    end
    if mod(numel(args), 2) ~= 0
        txt = '  <invalid uneven name/value list>';
        return;
    end
    lines = cell(numel(args) / 2, 1);
    line_idx = 0;
    for i = 1:2:numel(args)
        line_idx = line_idx + 1;
        lines{line_idx} = sprintf('  %s = %s', char(string(args{i})), local_value_to_string(args{i + 1}));
    end
    txt = strjoin(lines, newline);
end

function txt = local_value_to_string(value)
    if ischar(value)
        txt = value;
    elseif isstring(value)
        txt = char(strjoin(value(:)', ", "));
    elseif isnumeric(value) || islogical(value)
        txt = mat2str(value);
    elseif iscell(value)
        try
            txt = ['{' strjoin(cellfun(@local_value_to_string, value, 'UniformOutput', false), ', ') '}'];
        catch
            txt = '<cell>';
        end
    else
        txt = class(value);
    end
end
