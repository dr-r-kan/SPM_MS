% RUN_SIMULATED_THEN_METAMICROSTATE_PIPELINE
%
% Sequential runner for the current simulation + meta-selector + dataset plan:
%   1. simulated_ms_retrieval_experiment (training batch)
%   2. analyze_comparison_results (training batch)
%   3. train_spm_vb_meta_selector
%   4. simulated_ms_retrieval_experiment (held-out test batch)
%   5. analyze_comparison_results (held-out test batch)
%   6. apply_spm_vb_meta_selector
%   7. metamicrostate_dataset_pipeline
%
% Backward compatibility:
%   - If simulation_out_dir exists in the workspace, it is used as the
%     training simulation output directory.
%   - If simulation_args exists in the workspace, it is used as the base
%     argument list for both training and test simulation runs unless the
%     more specific *_args variables are already defined.
%
% Usage:
%   1. Edit the user settings below if needed.
%   2. Run this script from MATLAB.
%
% Optional workspace overrides:
%   manifest_csv
%   simulation_out_dir
%   simulation_train_out_dir, simulation_test_out_dir
%   metamicrostate_out_dir
%   simulation_args, simulation_train_args, simulation_test_args
%   analyze_train_args, analyze_test_args
%   meta_selector_model_file, meta_selector_train_args, meta_selector_apply_args
%   metamicrostate_args
%   run_training_analysis, run_meta_selector_training, run_test_analysis,
%   run_meta_selector_application, run_metamicrostate
%   verbose

clearvars -except manifest_csv simulation_out_dir simulation_train_out_dir simulation_test_out_dir ...
    metamicrostate_out_dir simulation_args simulation_train_args simulation_test_args ...
    analyze_train_args analyze_test_args meta_selector_model_file ...
    meta_selector_train_args meta_selector_apply_args metamicrostate_args ...
    run_training_analysis run_meta_selector_training run_test_analysis ...
    run_meta_selector_application run_metamicrostate verbose

util = microstate_utilities();
cfg = util.load_config();
sim_cfg = util.get_field(cfg, 'simulation', struct());

% ===== User settings =====
if ~exist('manifest_csv', 'var') || isempty(manifest_csv)
    manifest_csv = 'conditioned_lemon_sets.csv';
end

if ~exist('simulation_train_out_dir', 'var') || isempty(simulation_train_out_dir)
    if exist('simulation_out_dir', 'var') && ~isempty(simulation_out_dir)
        simulation_train_out_dir = simulation_out_dir;
    else
        simulation_train_out_dir = fullfile(char(cfg.paths.simulation_output_dir), 'meta_selector_train');
    end
end

if ~exist('simulation_test_out_dir', 'var') || isempty(simulation_test_out_dir)
    simulation_test_out_dir = fullfile(char(cfg.paths.simulation_output_dir), 'meta_selector_test');
end

if ~exist('metamicrostate_out_dir', 'var') || isempty(metamicrostate_out_dir)
    metamicrostate_out_dir = char(cfg.paths.hierarchical_output_dir);
end

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = true;
end

if ~exist('simulation_train_args', 'var') || isempty(simulation_train_args)
    if exist('simulation_args', 'var') && ~isempty(simulation_args)
        simulation_train_args = simulation_args;
    else
        simulation_train_args = {};
    end
end

if ~exist('simulation_test_args', 'var') || isempty(simulation_test_args)
    if exist('simulation_args', 'var') && ~isempty(simulation_args)
        simulation_test_args = simulation_args;
    else
        simulation_test_args = simulation_train_args;
    end
end

if ~exist('analyze_train_args', 'var') || isempty(analyze_train_args)
    analyze_train_args = {};
end

if ~exist('analyze_test_args', 'var') || isempty(analyze_test_args)
    analyze_test_args = analyze_train_args;
end

if ~exist('meta_selector_train_args', 'var') || isempty(meta_selector_train_args)
    meta_selector_train_args = {};
end

if ~exist('meta_selector_apply_args', 'var') || isempty(meta_selector_apply_args)
    meta_selector_apply_args = {};
end

if ~exist('metamicrostate_args', 'var') || isempty(metamicrostate_args)
    metamicrostate_args = {};
end

if ~exist('run_training_analysis', 'var') || isempty(run_training_analysis)
    run_training_analysis = true;
end

if ~exist('run_meta_selector_training', 'var') || isempty(run_meta_selector_training)
    run_meta_selector_training = true;
end

if ~exist('run_test_analysis', 'var') || isempty(run_test_analysis)
    run_test_analysis = true;
end

if ~exist('run_meta_selector_application', 'var') || isempty(run_meta_selector_application)
    run_meta_selector_application = true;
end

if ~exist('run_metamicrostate', 'var') || isempty(run_metamicrostate)
    run_metamicrostate = true;
end
% =========================

manifest_csv = util.resolve_path(char(manifest_csv), pwd);
simulation_train_out_dir = util.resolve_path(char(simulation_train_out_dir), util.project_root());
simulation_test_out_dir = util.resolve_path(char(simulation_test_out_dir), util.project_root());
metamicrostate_out_dir = util.resolve_path(char(metamicrostate_out_dir), util.project_root());

if ~isfile(manifest_csv)
    error('Manifest CSV not found: %s', manifest_csv);
end

util.ensure_dir(simulation_train_out_dir);
util.ensure_dir(simulation_test_out_dir);
util.ensure_dir(metamicrostate_out_dir);

hier_cfg = struct();
if isfield(cfg, 'hierarchical') && isstruct(cfg.hierarchical)
    hier_cfg = cfg.hierarchical;
end
single_cfg = struct();
if isfield(cfg, 'single_file') && isstruct(cfg.single_file)
    single_cfg = cfg.single_file;
end

simulation_train_args = local_set_default_arg(simulation_train_args, 'out_dir', simulation_train_out_dir);
simulation_train_args = local_set_default_arg(simulation_train_args, 'verbose', verbose);
simulation_train_args = local_set_default_arg(simulation_train_args, 'save_k_candidate_metrics', true);

simulation_test_args = local_set_default_arg(simulation_test_args, 'out_dir', simulation_test_out_dir);
simulation_test_args = local_set_default_arg(simulation_test_args, 'verbose', verbose);
simulation_test_args = local_set_default_arg(simulation_test_args, 'save_k_candidate_metrics', true);

default_sim_reps = double(util.get_field(sim_cfg, 'reps', 8));
train_rep_info = local_resolve_rep_block(simulation_train_args, default_sim_reps, 1);
simulation_train_args = train_rep_info.args_out;
if local_has_arg(simulation_test_args, 'rep_vals')
    test_rep_info = local_resolve_rep_block(simulation_test_args, default_sim_reps, train_rep_info.next_start);
    simulation_test_args = test_rep_info.args_out;
else
    test_rep_info = local_resolve_rep_block(simulation_test_args, train_rep_info.n_reps, train_rep_info.next_start);
    simulation_test_args = test_rep_info.args_out;
end

train_results_dir = fullfile(simulation_train_out_dir, 'results');
test_results_dir = fullfile(simulation_test_out_dir, 'results');
train_candidate_csv = fullfile(train_results_dir, 'k_candidate_metrics.csv');
test_candidate_csv = fullfile(test_results_dir, 'k_candidate_metrics.csv');

if ~exist('meta_selector_model_file', 'var') || isempty(meta_selector_model_file)
    meta_selector_model_file = fullfile(train_results_dir, 'spm_vb_meta_selector.mat');
end
meta_selector_model_file = util.resolve_path(char(meta_selector_model_file), util.project_root());

meta_selector_train_args = local_set_default_arg(meta_selector_train_args, 'output_model_file', meta_selector_model_file);
meta_selector_train_args = local_set_default_arg(meta_selector_train_args, 'verbose', verbose);

meta_selector_apply_prefix = fullfile(test_results_dir, 'spm_vb_meta_selector');
meta_selector_apply_args = local_set_default_arg(meta_selector_apply_args, 'output_prefix', meta_selector_apply_prefix);
meta_selector_apply_args = local_set_default_arg(meta_selector_apply_args, 'verbose', verbose);

metamicrostate_args = local_set_default_arg(metamicrostate_args, 'output_dir', metamicrostate_out_dir);
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'method', util.get_field(single_cfg, 'method', 'spm_vb'));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'criterion', util.get_field(hier_cfg, 'criterion', 'elbow_sil_combined'));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'K_candidates', util.get_field(hier_cfg, 'K_candidates', 4:7));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'template_file', char(cfg.paths.template_file));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'verbose', verbose);

step_defs = { ...
    struct('enabled', true, 'label', 'Training simulation'), ...
    struct('enabled', run_training_analysis, 'label', 'Training analysis'), ...
    struct('enabled', run_meta_selector_training, 'label', 'Meta-selector training'), ...
    struct('enabled', true, 'label', 'Held-out simulation'), ...
    struct('enabled', run_test_analysis, 'label', 'Held-out analysis'), ...
    struct('enabled', run_meta_selector_application, 'label', 'Meta-selector application'), ...
    struct('enabled', run_metamicrostate, 'label', 'Hierarchical dataset pipeline')};
step_total = sum(cellfun(@(s) logical(s.enabled), step_defs));
step_idx = 0;

fprintf('\n========================================\n');
fprintf('Sequential microstate runner\n');
fprintf('========================================\n');
fprintf('Manifest:          %s\n', manifest_csv);
fprintf('Train simulation:  %s\n', simulation_train_out_dir);
fprintf('Test simulation:   %s\n', simulation_test_out_dir);
fprintf('Meta model:        %s\n', meta_selector_model_file);
fprintf('Meta output:       %s\n', metamicrostate_out_dir);
fprintf('Train reps:        %s\n', mat2str(train_rep_info.rep_vals));
fprintf('Test reps:         %s\n', mat2str(test_rep_info.rep_vals));
fprintf('========================================\n\n');

RunnerResults = struct();
RunnerResults.manifest_csv = manifest_csv;
RunnerResults.simulation_train_out_dir = simulation_train_out_dir;
RunnerResults.simulation_test_out_dir = simulation_test_out_dir;
RunnerResults.metamicrostate_out_dir = metamicrostate_out_dir;
RunnerResults.meta_selector_model_file = meta_selector_model_file;

% Step 1: training simulation
step_idx = step_idx + 1;
fprintf('[%d/%d] Running simulated_ms_retrieval_experiment (training batch) ...\n', step_idx, step_total);
t0 = tic;
SimulationTrainResults = simulated_ms_retrieval_experiment(simulation_train_args{:});
train_sim_elapsed = toc(t0);
RunnerResults.simulation_train = SimulationTrainResults;
RunnerResults.simulation_train_results_csv = fullfile(train_results_dir, 'comparison_results.csv');
RunnerResults.simulation_train_k_candidate_csv = train_candidate_csv;
fprintf('[%d/%d] Complete in %s\n', step_idx, step_total, local_duration_string(train_sim_elapsed));
fprintf('          Results CSV: %s\n', RunnerResults.simulation_train_results_csv);
fprintf('          Per-K CSV:   %s\n\n', RunnerResults.simulation_train_k_candidate_csv);

% Step 2: training analysis
if run_training_analysis
    step_idx = step_idx + 1;
    fprintf('[%d/%d] Running analyze_comparison_results (training batch) ...\n', step_idx, step_total);
    t0 = tic;
    analyze_comparison_results(train_results_dir, analyze_train_args{:});
    train_analysis_elapsed = toc(t0);
    RunnerResults.training_analysis_dir = fullfile(simulation_train_out_dir, 'analysis_plots');
    fprintf('[%d/%d] Complete in %s\n\n', step_idx, step_total, local_duration_string(train_analysis_elapsed));
end

% Step 3: meta-selector training
if run_meta_selector_training
    step_idx = step_idx + 1;
    fprintf('[%d/%d] Running train_spm_vb_meta_selector ...\n', step_idx, step_total);
    t0 = tic;
    MetaSelectorModel = train_spm_vb_meta_selector(train_candidate_csv, meta_selector_train_args{:});
    meta_train_elapsed = toc(t0);
    RunnerResults.meta_selector_model = MetaSelectorModel;
    fprintf('[%d/%d] Complete in %s\n', step_idx, step_total, local_duration_string(meta_train_elapsed));
    fprintf('          Model file: %s\n\n', meta_selector_model_file);
end

% Step 4: held-out simulation
step_idx = step_idx + 1;
fprintf('[%d/%d] Running simulated_ms_retrieval_experiment (held-out batch) ...\n', step_idx, step_total);
t0 = tic;
SimulationTestResults = simulated_ms_retrieval_experiment(simulation_test_args{:});
test_sim_elapsed = toc(t0);
RunnerResults.simulation_test = SimulationTestResults;
RunnerResults.simulation_test_results_csv = fullfile(test_results_dir, 'comparison_results.csv');
RunnerResults.simulation_test_k_candidate_csv = test_candidate_csv;
fprintf('[%d/%d] Complete in %s\n', step_idx, step_total, local_duration_string(test_sim_elapsed));
fprintf('          Results CSV: %s\n', RunnerResults.simulation_test_results_csv);
fprintf('          Per-K CSV:   %s\n\n', RunnerResults.simulation_test_k_candidate_csv);

% Step 5: held-out analysis
if run_test_analysis
    step_idx = step_idx + 1;
    fprintf('[%d/%d] Running analyze_comparison_results (held-out batch) ...\n', step_idx, step_total);
    t0 = tic;
    analyze_comparison_results(test_results_dir, analyze_test_args{:});
    test_analysis_elapsed = toc(t0);
    RunnerResults.test_analysis_dir = fullfile(simulation_test_out_dir, 'analysis_plots');
    fprintf('[%d/%d] Complete in %s\n\n', step_idx, step_total, local_duration_string(test_analysis_elapsed));
end

% Step 6: meta-selector application
if run_meta_selector_application
    step_idx = step_idx + 1;
    fprintf('[%d/%d] Running apply_spm_vb_meta_selector ...\n', step_idx, step_total);
    t0 = tic;
    [MetaSelectorSummary, MetaSelectorSelected, MetaSelectorCandidates] = ...
        apply_spm_vb_meta_selector(test_candidate_csv, meta_selector_model_file, meta_selector_apply_args{:});
    meta_apply_elapsed = toc(t0);
    RunnerResults.meta_selector_summary = MetaSelectorSummary;
    RunnerResults.meta_selector_selected = MetaSelectorSelected;
    RunnerResults.meta_selector_candidates = MetaSelectorCandidates;
    fprintf('[%d/%d] Complete in %s\n', step_idx, step_total, local_duration_string(meta_apply_elapsed));
    fprintf('          Run-level accuracy: %.3f\n\n', MetaSelectorSummary.run_accuracy);
end

% Step 7: hierarchical dataset pipeline
if run_metamicrostate
    step_idx = step_idx + 1;
    fprintf('[%d/%d] Running metamicrostate_dataset_pipeline ...\n', step_idx, step_total);
    t0 = tic;
    [MetaResults, output_csv] = metamicrostate_dataset_pipeline(manifest_csv, metamicrostate_args{:});
    meta_elapsed = toc(t0);
    RunnerResults.metamicrostate = MetaResults;
    RunnerResults.metamicrostate_manifest_csv = output_csv;
    fprintf('[%d/%d] Complete in %s\n', step_idx, step_total, local_duration_string(meta_elapsed));
    fprintf('          Output manifest: %s\n\n', output_csv);
end

fprintf('Finished sequential run.\n');
fprintf('Training simulation: %s\n', local_duration_string(train_sim_elapsed));
fprintf('Held-out simulation: %s\n', local_duration_string(test_sim_elapsed));
if run_metamicrostate
    fprintf('Dataset pipeline:    %s\n', local_duration_string(meta_elapsed));
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

function tf = local_has_arg(args, name)
    tf = false;
    if ~iscell(args) || mod(numel(args), 2) ~= 0
        return;
    end
    for i = 1:2:numel(args)
        key = args{i};
        if (ischar(key) || isstring(key)) && strcmpi(char(key), name)
            tf = true;
            return;
        end
    end
end

function value = local_get_arg(args, name, default_value)
    value = default_value;
    if ~iscell(args) || mod(numel(args), 2) ~= 0
        return;
    end
    for i = 1:2:numel(args)
        key = args{i};
        if (ischar(key) || isstring(key)) && strcmpi(char(key), name)
            value = args{i + 1};
            return;
        end
    end
end

function info = local_resolve_rep_block(args_in, default_reps, start_rep)
    info = struct();
    args_out = args_in;
    rep_vals = local_get_arg(args_in, 'rep_vals', []);
    if isempty(rep_vals)
        reps = double(local_get_arg(args_in, 'reps', default_reps));
        reps = max(1, round(reps));
        rep_vals = start_rep:(start_rep + reps - 1);
        args_out = local_set_default_arg(args_out, 'rep_vals', rep_vals);
        args_out = local_set_default_arg(args_out, 'reps', reps);
    else
        rep_vals = double(rep_vals(:)');
        reps = numel(rep_vals);
        args_out = local_set_default_arg(args_out, 'reps', reps);
    end
    info.args_out = args_out;
    info.rep_vals = rep_vals;
    info.n_reps = reps;
    info.next_start = max(rep_vals) + 1;
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
