% RUN_SIMULATED_THEN_METAMICROSTATE_PIPELINE
%
% Sequential runner for:
%   1. simulated_ms_retrieval_experiment
%   2. metamicrostate_dataset_pipeline
%
% Usage:
%   1. Edit the user settings below if needed.
%   2. Run this script from MATLAB.
%
% Optional workspace overrides:
%   manifest_csv, simulation_out_dir, metamicrostate_out_dir,
%   simulation_args, metamicrostate_args, verbose

clearvars -except manifest_csv simulation_out_dir metamicrostate_out_dir simulation_args metamicrostate_args verbose

util = microstate_utilities();
cfg = util.load_config();

% ===== User settings =====
if ~exist('manifest_csv', 'var') || isempty(manifest_csv)
    manifest_csv = 'conditioned_lemon_sets.csv';
end

if ~exist('simulation_out_dir', 'var') || isempty(simulation_out_dir)
    simulation_out_dir = char(cfg.paths.simulation_output_dir);
end

if ~exist('metamicrostate_out_dir', 'var') || isempty(metamicrostate_out_dir)
    metamicrostate_out_dir = char(cfg.paths.hierarchical_output_dir);
end

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = true;
end

if ~exist('simulation_args', 'var') || isempty(simulation_args)
    simulation_args = {};
end

if ~exist('metamicrostate_args', 'var') || isempty(metamicrostate_args)
    metamicrostate_args = {};
end
% =========================

manifest_csv = util.resolve_path(char(manifest_csv), pwd);
simulation_out_dir = util.resolve_path(char(simulation_out_dir), util.project_root());
metamicrostate_out_dir = util.resolve_path(char(metamicrostate_out_dir), util.project_root());

if ~isfile(manifest_csv)
    error('Manifest CSV not found: %s', manifest_csv);
end

util.ensure_dir(simulation_out_dir);
util.ensure_dir(metamicrostate_out_dir);

hier_cfg = struct();
if isfield(cfg, 'hierarchical') && isstruct(cfg.hierarchical)
    hier_cfg = cfg.hierarchical;
end
single_cfg = struct();
if isfield(cfg, 'single_file') && isstruct(cfg.single_file)
    single_cfg = cfg.single_file;
end

simulation_args = local_set_default_arg(simulation_args, 'out_dir', simulation_out_dir);
simulation_args = local_set_default_arg(simulation_args, 'verbose', verbose);

metamicrostate_args = local_set_default_arg(metamicrostate_args, 'output_dir', metamicrostate_out_dir);
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'method', util.get_field(single_cfg, 'method', 'spm_vb'));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'criterion', util.get_field(hier_cfg, 'criterion', 'elbow_sil_combined'));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'K_candidates', util.get_field(hier_cfg, 'K_candidates', 4:7));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'template_file', char(cfg.paths.template_file));
metamicrostate_args = local_set_default_arg(metamicrostate_args, 'verbose', verbose);

fprintf('\n========================================\n');
fprintf('Sequential microstate runner\n');
fprintf('========================================\n');
fprintf('Manifest:    %s\n', manifest_csv);
fprintf('Simulation:  %s\n', simulation_out_dir);
fprintf('Meta output: %s\n', metamicrostate_out_dir);
fprintf('========================================\n\n');

sim_t0 = tic;
fprintf('[1/2] Running simulated_ms_retrieval_experiment ...\n');
SimulationResults = simulated_ms_retrieval_experiment(simulation_args{:});
sim_elapsed = toc(sim_t0);

simulation_results_csv = fullfile(simulation_out_dir, 'results', 'comparison_results.csv');
fprintf('[1/2] Complete in %s\n', local_duration_string(sim_elapsed));
if isfile(simulation_results_csv)
    fprintf('      Results CSV: %s\n\n', simulation_results_csv);
else
    fprintf('      Results directory: %s\n\n', fullfile(simulation_out_dir, 'results'));
end

meta_t0 = tic;
fprintf('[2/2] Running metamicrostate_dataset_pipeline ...\n');
[MetaResults, output_csv] = metamicrostate_dataset_pipeline(manifest_csv, metamicrostate_args{:});
meta_elapsed = toc(meta_t0);

RunnerResults = struct();
RunnerResults.simulation = SimulationResults; %#ok<NASGU>
RunnerResults.metamicrostate = MetaResults; %#ok<NASGU>
RunnerResults.manifest_csv = manifest_csv; %#ok<NASGU>
RunnerResults.simulation_out_dir = simulation_out_dir; %#ok<NASGU>
RunnerResults.metamicrostate_out_dir = metamicrostate_out_dir; %#ok<NASGU>
RunnerResults.simulation_results_csv = simulation_results_csv; %#ok<NASGU>
RunnerResults.metamicrostate_manifest_csv = output_csv; %#ok<NASGU>

fprintf('[2/2] Complete in %s\n', local_duration_string(meta_elapsed));
fprintf('      Output manifest: %s\n\n', output_csv);

fprintf('Finished sequential run.\n');
fprintf('Simulation time: %s\n', local_duration_string(sim_elapsed));
fprintf('Dataset time:    %s\n', local_duration_string(meta_elapsed));

function file_path = local_pick_existing_file(candidates)
    util_local = microstate_utilities();
    root_dir = util_local.project_root();
    for i = 1:numel(candidates)
        candidate = util_local.resolve_path(candidates{i}, root_dir);
        if isfile(candidate)
            file_path = candidate;
            return;
        end
    end
    error('No default manifest CSV found. Set manifest_csv before running this script.');
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
