function [HResults, mat_file] = fit_microstate_hierarchical_dataset(manifest_csv, varargin)
% FIT_MICROSTATE_HIERARCHICAL_DATASET
%
% Dataset-level hierarchical microstate fitting for EEG files listed in a CSV.
%
% This function is designed to sit alongside the existing single-file codebase:
%   - analyze_single_eeg_file.m
%   - fit_microstate_spm_vb.m
%   - fit_microstate_spm_kmeans.m
%   - fit_microstate_kmeans_koenig.m
%   - microstate_utilities.m
%
% It does not alter the single-file workflow.  It reuses the shared
% preprocessing utilities where possible, then performs a top-down empirical
% Bayes / MAP hierarchy:
%
%   global pooled maps        -> select K
%   group-level maps          -> fixed K, prior = global templates
%   group x condition maps    -> fixed K, prior = group templates
%   participant x group x condition maps -> fixed K, prior = condition templates
%   file-level maps           -> fixed K, prior = participant-condition templates
%
% The lower-level fits keep the globally selected K, so they are much faster
% than reselecting K at every node.  Parent templates enter the lower-level
% fit as a topographic axial prior.  Operationally this is a prior-weighted
% leading-eigenvector update of each microstate map, plus optional SPM
% variational-mixture initialisation in a polarity-invariant PCA feature
% space.  This is a defensible empirical-Bayes approximation, not a full
% hidden Markov/semi-Markov state-sequence model.
%
% REQUIRED CSV COLUMNS
%   file_path      path to .set or .mat EEG file
%   group          group label
%   condition      condition/state label
%
% OPTIONAL CSV COLUMNS
%   participant    participant/subject identifier. If absent, basename is used.
%
% Accepted aliases:
%   file_path:   file, filepath, path, eeg_file, eeg_path, filename
%   participant: participant, subject, sub, subj, id, participant_id, subject_id
%   group:       group, grp, diagnosis, cohort
%   condition:   condition, cond, state, task, block
%
% NAME-VALUE OPTIONS
%   'K_candidates'          candidate K values for the global fit (default: 4:7)
%   'criterion'             'elbow_sil_combined', 'silhouette', 'free_energy',
%                           'gev', or 'elbow' (default: 'elbow_sil_combined')
%   'output_dir'            output directory (default: config path)
%   'template_file'         MetaMaps .set file used for canonical per-K
%                           initialisation/alignment when available
%                           (default: config path)
%   'use_template_initialisation' seed the global K search and global
%                           refit from the canonical MetaMaps solution for
%                           the requested K when available (default: true)
%   'canonical_reporting_template_K' MetaMaps K used for final reporting
%                           alignment. Empty uses selected K (default: [])
%   'canonical_prior_weight_global' axial MetaMaps prior weight used during
%                           global fitting when templates are available
%                           (default: 100)
%   'apply_average_reference' apply common average reference before GFP
%                           extraction in each file (default: true)
%   'spatial_filter'       optional spatial filter before GFP extraction:
%                          'none', 'smoothing', or 'laplacian' (default: 'none')
%   'spatial_filter_matrix' optional custom n_channels x n_channels spatial
%                          filter matrix (default: [])
%   'spatial_filter_neighbours' neighbour count for laplacian filter
%                          (default: 6)
%   'spatial_filter_strength' blend strength for smoothing/laplacian
%                          (default: 1)
%   'filter_band'          temporal bandpass before GFP extraction
%                          (default: [2 20])
%   'reject_gfp_peak_outliers_individual' reject within-file GFP peak
%                          outliers before any hierarchy fit (default: true)
%   'gfp_outlier_mad_multiplier_individual' MAD multiplier for within-file
%                          GFP outlier rejection (default: 6)
%   'reject_gfp_peak_outliers_population' reject pooled GFP peak outliers
%                          after all files are preprocessed (default: true)
%   'gfp_outlier_mad_multiplier_population' MAD multiplier for pooled GFP
%                          outlier rejection (default: 6)
%   'reject_template_misaligned_peaks' remove GFP peaks whose topography has
%                          weak MetaMaps similarity before fitting (default: true)
%   'template_peak_min_abs_corr' minimum max |r| to MetaMaps templates for a
%                          GFP-peak map to be retained (default: 0.65)
%   'spm_path'              optional SPM mixture-toolbox path. If empty,
%                          common local paths are tried (default: '')
%   'require_spm_initialisation' error if spm_mix is unavailable or fails
%                          during SPM initialisation (default: true)
%   'sfreq'                 fallback sampling rate for .mat files (default: 250)
%   'use_scalp_channels'    drop obvious non-scalp channels if chanlocs exist (default: true)
%   'channel_policy'        'intersect' or 'strict' (default: 'intersect')
%   'max_maps_per_file'     cap GFP-peak maps per file for fitting (default: 1500)
%   'max_global_maps'       cap maps used in global K search (default: 30000)
%   'min_maps_per_node'     minimum maps to refit a node. Smaller nodes inherit
%                           parent templates and are only assigned (default: [])
%   'prior_weight_group'    parent prior weight for group nodes (default: 25)
%   'prior_weight_condition' parent prior weight for group x condition nodes (default: 35)
%   'prior_weight_participant' parent prior weight for participant-condition nodes (default: 50)
%   'prior_weight_file'     parent prior weight for file-level nodes (default: 75)
%   'spm_prior_pseudocount' pseudo-observations appended to SPM feature fit
%                           for each parent map (default: 8)
%   'use_spm_initialisation' use spm_mix where available (default: true)
%   'n_refine_iter'         topographic MAP refinement iterations (default: 50)
%   'random_seed'           deterministic seed (default: 1)
%   'verbose'               print progress (default: true)
%   'save_mat'              save full .mat results (default: true)
%   'save_csv'              save node and file summaries (default: true)
%
% OUTPUT
%   HResults   structured results, including selected K, templates, labels,
%              and summaries at every hierarchical level.
%   mat_file   path to the saved .mat file, or '' if save_mat=false.
%
% EXAMPLE
%   [H, f] = fit_microstate_hierarchical_dataset('eeg_manifest.csv', ...
%       'K_candidates', 2:8, ...
%       'criterion', 'elbow_sil_combined', ...
%       'output_dir', 'microstate_hierarchical_out');

    util = microstate_utilities();
    repo_cfg = util.load_config();
    h_defaults = repo_cfg.hierarchical;
    pre_defaults = repo_cfg.preprocessing;
    path_defaults = repo_cfg.paths;

    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', double(h_defaults.K_candidates(:)'), @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'criterion', char(h_defaults.criterion), @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', char(path_defaults.hierarchical_output_dir), @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', char(path_defaults.template_file), @(x) ischar(x) || isstring(x));
    addParameter(p, 'use_template_initialisation', logical(h_defaults.use_template_initialisation), @(x) islogical(x) && isscalar(x));
    addParameter(p, 'canonical_reporting_template_K', h_defaults.canonical_reporting_template_K, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    addParameter(p, 'canonical_prior_weight_global', double(h_defaults.canonical_prior_weight_global), @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'apply_average_reference', logical(pre_defaults.apply_average_reference), @(x) islogical(x) && isscalar(x));
    addParameter(p, 'spatial_filter', char(pre_defaults.spatial_filter), @(x) ischar(x) || isstring(x));
    addParameter(p, 'spatial_filter_matrix', [], @isnumeric);
    addParameter(p, 'spatial_filter_neighbours', double(pre_defaults.spatial_filter_neighbours), @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'spatial_filter_strength', double(pre_defaults.spatial_filter_strength), @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'filter_band', double(pre_defaults.filter_band(:)'), @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    addParameter(p, 'reject_gfp_peak_outliers_individual', logical(pre_defaults.reject_gfp_peak_outliers), @(x) islogical(x) && isscalar(x));
    addParameter(p, 'gfp_outlier_mad_multiplier_individual', double(pre_defaults.gfp_outlier_mad_multiplier), @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'reject_gfp_peak_outliers_population', logical(pre_defaults.reject_gfp_peak_outliers), @(x) islogical(x) && isscalar(x));
    addParameter(p, 'gfp_outlier_mad_multiplier_population', double(pre_defaults.gfp_outlier_mad_multiplier), @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'reject_template_misaligned_peaks', logical(h_defaults.reject_template_misaligned_peaks), @(x) islogical(x) && isscalar(x));
    addParameter(p, 'template_peak_min_abs_corr', double(h_defaults.template_peak_min_abs_corr), @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    addParameter(p, 'template_peak_template_K', double(h_defaults.template_peak_template_K), @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'min_peak_count_after_template_rejection', double(h_defaults.min_peak_count_after_template_rejection), @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'sfreq', 250, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'use_scalp_channels', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'channel_policy', 'intersect', @(x) ischar(x) || isstring(x));
    addParameter(p, 'max_maps_per_file', double(h_defaults.max_maps_per_file), @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'max_global_maps', double(h_defaults.max_global_maps), @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'min_maps_per_node', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 0));
    addParameter(p, 'prior_weight_group', 25, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_condition', 35, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_participant', 50, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'prior_weight_file', 75, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'spm_prior_pseudocount', 8, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'use_spm_initialisation', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'spm_path', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'require_spm_initialisation', logical(h_defaults.require_spm_initialisation), @(x) islogical(x) && isscalar(x));
    addParameter(p, 'n_refine_iter', 50, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'random_seed', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'verbose', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'save_mat', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'save_csv', true, @(x) islogical(x) && isscalar(x));
    parse(p, manifest_csv, varargin{:});
    cfg = p.Results;

    cfg.manifest_csv = char(cfg.manifest_csv);
    cfg.output_dir = char(util.resolve_path(cfg.output_dir, util.project_root()));
    cfg.template_file = char(util.resolve_path(cfg.template_file, util.project_root()));
    cfg.spm_path = char(cfg.spm_path);
    cfg.config_spm_mixture_paths = path_defaults.spm_mixture_paths;
    cfg.spatial_filter = char(cfg.spatial_filter);
    cfg.criterion = char(cfg.criterion);
    cfg.channel_policy = lower(char(cfg.channel_policy));
    cfg.K_candidates = unique(round(cfg.K_candidates(:)'));
    cfg.K_candidates = cfg.K_candidates(cfg.K_candidates >= 2);
    if isempty(cfg.K_candidates)
        error('K_candidates must contain at least one integer >= 2.');
    end
    if isempty(cfg.min_maps_per_node)
        cfg.min_maps_per_node = max(50, 8 * max(cfg.K_candidates));
    end
    if ~ismember(cfg.channel_policy, {'intersect', 'strict'})
        error('channel_policy must be ''intersect'' or ''strict''.');
    end
    if ~isfile(cfg.manifest_csv)
        error('Manifest CSV not found: %s', cfg.manifest_csv);
    end
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end

    rng(cfg.random_seed, 'twister');
    cfg = ensure_spm_mix_available(cfg);

    if cfg.verbose
        fprintf('\n========================================\n');
        fprintf('Hierarchical EEG microstate fitting\n');
        fprintf('========================================\n');
        fprintf('Manifest: %s\n', cfg.manifest_csv);
        fprintf('Output:   %s\n', cfg.output_dir);
        if cfg.use_template_initialisation
            fprintf('Canonical template init: %s\n', cfg.template_file);
        else
            fprintf('Canonical template init: disabled\n');
        end
        fprintf('Average reference: %s\n', ternary_str(cfg.apply_average_reference));
        fprintf('Spatial filter: %s\n', cfg.spatial_filter);
        if cfg.use_spm_initialisation
            fprintf('SPM initialisation: %s\n', ternary_str(exist('spm_mix', 'file') == 2));
        end
        fprintf('Individual GFP outlier rejection: %s (MAD x %.2f)\n', ...
            ternary_str(cfg.reject_gfp_peak_outliers_individual), cfg.gfp_outlier_mad_multiplier_individual);
        fprintf('Population GFP outlier rejection: %s (MAD x %.2f)\n', ...
            ternary_str(cfg.reject_gfp_peak_outliers_population), cfg.gfp_outlier_mad_multiplier_population);
        fprintf('Template peak filter: %s (min max |r| %.2f)\n', ...
            ternary_str(cfg.reject_template_misaligned_peaks), cfg.template_peak_min_abs_corr);
        fprintf('K candidates: %s\n', mat2str(cfg.K_candidates));
        fprintf('Global criterion: %s\n', cfg.criterion);
        fprintf('Hierarchy: global -> group -> group-condition -> participant-condition -> file\n');
        fprintf('========================================\n\n');
    end

    manifest = read_and_standardise_manifest(cfg.manifest_csv);
    n_files = height(manifest);
    if n_files == 0
        error('Manifest contains no rows.');
    end

    if cfg.verbose
        fprintf('1. Checking files and channel compatibility...\n');
    end
    file_meta = inspect_channel_sets(manifest, cfg);
    [common_labels, common_index_by_file, common_chanlocs, common_pos] = define_common_channels(file_meta, cfg);

    if cfg.verbose
        fprintf('   Files: %d\n', n_files);
        fprintf('   Common analysis channels: %d\n', numel(common_labels));
        fprintf('   Groups: %d | Conditions: %d | Participants: %d\n', ...
            numel(unique(manifest.group)), numel(unique(manifest.condition)), numel(unique(manifest.participant)));
    end
    if numel(common_labels) < 8
        error('Only %d common channels remain. Check the manifest or channel labels.', numel(common_labels));
    end

    [cfg.template_initial_centers_by_K, cfg.template_initialisation_info] = ...
        load_template_initial_centers_by_K(cfg, common_labels, common_pos);
    cfg.template_peak_reference = load_template_peak_reference(cfg, common_labels, common_pos);
    if cfg.verbose
        if ~isempty(cfg.template_initialisation_info)
            available = [cfg.template_initialisation_info.K];
            fprintf('   Canonical template seeds loaded for K: %s\n', mat2str(available));
        elseif cfg.use_template_initialisation
            fprintf('   No canonical template seeds were available for the requested K candidates.\n');
        end
    end

    if cfg.verbose
        fprintf('\n2. Extracting GFP-peak maps from each file...\n');
    end
    records = repmat(empty_record_struct(), n_files, 1);
    all_maps = cell(n_files, 1);
    all_maps_full = cell(n_files, 1);
    map_file_index = cell(n_files, 1);

    for i = 1:n_files
        if cfg.verbose
            fprintf('   [%d/%d] %s\n', i, n_files, manifest.file_path{i});
        end
        rec = extract_file_maps(manifest.file_path{i}, common_index_by_file{i}, common_labels, common_chanlocs, common_pos, cfg, util);
        rec = apply_template_peak_filter(rec, cfg);
        rec.file_index = i;
        rec.group = manifest.group{i};
        rec.condition = manifest.condition{i};
        rec.participant = manifest.participant{i};
        records(i) = rec;
        all_maps{i} = rec.maps_fit;
        all_maps_full{i} = rec.maps_norm;
        map_file_index{i} = repmat(i, size(rec.maps_fit, 1), 1);
    end

    [records, population_gfp_summary] = apply_population_gfp_outlier_filter(records, cfg);
    if cfg.verbose && cfg.reject_gfp_peak_outliers_population
        fprintf('   Population GFP threshold: %.4f | removed %d peaks\n', ...
            population_gfp_summary.threshold, population_gfp_summary.n_removed);
    end

    for i = 1:n_files
        records(i) = finalize_record_fit_maps(records(i), cfg.max_maps_per_file);
        all_maps{i} = records(i).maps_fit;
        all_maps_full{i} = records(i).maps_norm;
        map_file_index{i} = repmat(i, size(records(i).maps_fit, 1), 1);
    end

    pooled_maps = vertcat(all_maps{:});
    pooled_file_index = vertcat(map_file_index{:});
    if isempty(pooled_maps)
        error('No GFP-peak maps were extracted.');
    end
    pooled_maps = util.normalize_maps(pooled_maps);

    if cfg.verbose
        fprintf('   Pooled fitting maps: %d x %d\n', size(pooled_maps, 1), size(pooled_maps, 2));
    end

    global_maps = pooled_maps;
    if ~isempty(cfg.max_global_maps) && size(global_maps, 1) > cfg.max_global_maps
        idx = deterministic_subsample(size(global_maps, 1), cfg.max_global_maps);
        global_maps = global_maps(idx, :);
        if cfg.verbose
            fprintf('   Global K search capped at %d maps.\n', size(global_maps, 1));
        end
    end

    if cfg.verbose
        fprintf('\n3. Global K selection and pooled template fitting...\n');
    end
    global_fit = fit_microstate_maps_hierarchical(global_maps, cfg.K_candidates, cfg.criterion, [], 0, cfg, 'global');
    K = global_fit.K_estimated;

    if cfg.verbose
        fprintf('\nSelected global K = %d\n', K);
    end

    % Refit the selected global K on all capped per-file maps, not only on the
    % possible K-search subset.  This gives a single canonical parent template.
    global_selected = fit_microstate_maps_fixedK(pooled_maps, K, [], 0, cfg, 'global_selected');
    global_fit.selected = global_selected;
    global_fit.centers = global_selected.centers;
    global_fit.labels = global_selected.labels;
    global_fit.metrics = global_selected.metrics;
    [global_fit, global_selected] = attach_canonical_reporting_alignment(global_fit, global_selected, cfg, common_labels, common_pos);
    global_fit.selected = global_selected;
    global_fit.centers = global_selected.centers;
    global_fit.labels = global_selected.labels;
    global_fit.metrics = global_selected.metrics;

    if cfg.verbose
        fprintf('\n4. Fitting group-level nodes with fixed K=%d...\n', K);
    end
    group_labels = unique(manifest.group, 'stable');
    groups = repmat(empty_node_struct(), numel(group_labels), 1);
    for gi = 1:numel(group_labels)
        g = group_labels{gi};
        file_idx = find(strcmp(manifest.group, g));
        maps = vertcat(all_maps{file_idx});
        node = fit_node_or_inherit(sprintf('group:%s', g), 'group', g, '', '', file_idx, maps, ...
            K, global_fit.centers, cfg.prior_weight_group, cfg, global_fit.centers);
        groups(gi) = node;
    end

    if cfg.verbose
        fprintf('\n5. Fitting group x condition nodes with fixed K=%d...\n', K);
    end
    condition_nodes = empty_node_struct();
    condition_nodes(1) = [];
    cn = 0;
    for gi = 1:numel(groups)
        g = groups(gi).group;
        conds = unique(manifest.condition(strcmp(manifest.group, g)), 'stable');
        for ci = 1:numel(conds)
            c = conds{ci};
            file_idx = find(strcmp(manifest.group, g) & strcmp(manifest.condition, c));
            maps = vertcat(all_maps{file_idx});
            cn = cn + 1;
            condition_nodes(cn) = fit_node_or_inherit(sprintf('group_condition:%s:%s', g, c), ...
                'group_condition', g, c, '', file_idx, maps, K, groups(gi).centers, ...
                cfg.prior_weight_condition, cfg, global_fit.centers);
        end
    end

    if cfg.verbose
        fprintf('\n6. Fitting participant x group x condition nodes with fixed K=%d...\n', K);
    end
    participant_nodes = empty_node_struct();
    participant_nodes(1) = [];
    pn = 0;
    triplets = unique(manifest(:, {'participant', 'group', 'condition'}), 'rows', 'stable');
    for ti = 1:height(triplets)
        sub = triplets.participant{ti};
        g = triplets.group{ti};
        c = triplets.condition{ti};
        file_idx = find(strcmp(manifest.participant, sub) & strcmp(manifest.group, g) & strcmp(manifest.condition, c));
        maps = vertcat(all_maps{file_idx});
        parent_idx = find(strcmp({condition_nodes.group}, g) & strcmp({condition_nodes.condition}, c), 1, 'first');
        if isempty(parent_idx)
            parent_centers = global_fit.centers;
        else
            parent_centers = condition_nodes(parent_idx).centers;
        end
        pn = pn + 1;
        participant_nodes(pn) = fit_node_or_inherit(sprintf('participant_condition:%s:%s:%s', sub, g, c), ...
            'participant_condition', g, c, sub, file_idx, maps, K, parent_centers, ...
            cfg.prior_weight_participant, cfg, global_fit.centers);
    end

    if cfg.verbose
        fprintf('\n7. Assigning/refitting file-level nodes with fixed K=%d...\n', K);
    end
    file_nodes = repmat(empty_node_struct(), n_files, 1);
    for i = 1:n_files
        sub = manifest.participant{i};
        g = manifest.group{i};
        c = manifest.condition{i};
        parent_idx = find(strcmp({participant_nodes.participant}, sub) & ...
                          strcmp({participant_nodes.group}, g) & ...
                          strcmp({participant_nodes.condition}, c), 1, 'first');
        if isempty(parent_idx)
            parent_centers = global_fit.centers;
        else
            parent_centers = participant_nodes(parent_idx).centers;
        end
        maps = records(i).maps_norm;
        file_nodes(i) = fit_node_or_inherit(sprintf('file:%03d:%s', i, base_name(manifest.file_path{i})), ...
            'file', g, c, sub, i, maps, K, parent_centers, cfg.prior_weight_file, cfg, global_fit.centers);
        file_nodes(i).file_path = manifest.file_path{i};
    end

    if cfg.verbose
        fprintf('\n8. Building summaries and saving outputs...\n');
    end
    node_summary = make_node_summary(global_fit, groups, condition_nodes, participant_nodes, file_nodes, K);
    file_summary = make_file_summary(manifest, records, file_nodes, K);
    global_model_comparison = make_global_model_comparison(global_fit);

    HResults = struct();
    HResults.CONFIG = cfg;
    HResults.manifest = manifest;
    HResults.selected_K = K;
    HResults.common_channel_labels = common_labels;
    HResults.common_chanlocs = common_chanlocs;
    HResults.common_pos = common_pos;
    HResults.records = records;
    HResults.population_gfp_outlier_summary = population_gfp_summary;
    HResults.global = global_fit;
    HResults.groups = groups;
    HResults.group_conditions = condition_nodes;
    HResults.participant_conditions = participant_nodes;
    HResults.files = file_nodes;
    HResults.node_summary = node_summary;
    HResults.file_summary = file_summary;
    HResults.global_model_comparison = global_model_comparison;
    HResults.template_initialisation = cfg.template_initialisation_info;
    if isfield(global_fit, 'template_alignment')
        HResults.canonical_template_alignment = global_fit.template_alignment;
    else
        HResults.canonical_template_alignment = struct();
    end
    HResults.created = datestr(now, 30);

    mat_file = '';
    if cfg.save_csv
        writetable(node_summary, fullfile(cfg.output_dir, 'hierarchical_node_summary.csv'));
        writetable(file_summary, fullfile(cfg.output_dir, 'hierarchical_file_summary.csv'));
        writetable(global_model_comparison, fullfile(cfg.output_dir, 'global_K_model_comparison.csv'));
        write_manifest_copy(manifest, fullfile(cfg.output_dir, 'manifest_resolved.csv'));
    end
    if cfg.save_mat
        mat_file = fullfile(cfg.output_dir, 'hierarchical_microstate_results.mat');
        save(mat_file, 'HResults', '-v7.3');
    end

    save_template_matrices(HResults, cfg.output_dir);

    if cfg.verbose
        fprintf('\nDone.\n');
        if cfg.save_mat
            fprintf('Full results: %s\n', mat_file);
        end
        if cfg.save_csv
            fprintf('Node summary: %s\n', fullfile(cfg.output_dir, 'hierarchical_node_summary.csv'));
            fprintf('File summary: %s\n', fullfile(cfg.output_dir, 'hierarchical_file_summary.csv'));
            fprintf('Global K comparison: %s\n', fullfile(cfg.output_dir, 'global_K_model_comparison.csv'));
        end
    end
end

% =======================================================================
% Manifest and loading
% =======================================================================

function manifest = read_and_standardise_manifest(manifest_csv)
    T = readtable(manifest_csv, 'TextType', 'string', 'Delimiter', ',');
    if isempty(T)
        manifest = table();
        return;
    end
    names = matlab.lang.makeValidName(lower(strtrim(T.Properties.VariableNames)));
    T.Properties.VariableNames = names;

    file_col = find_first_col(names, {'file_path', 'filepath', 'file', 'path', 'eeg_file', 'eeg_path', 'filename'});
    group_col = find_first_col(names, {'group', 'grp', 'diagnosis', 'cohort'});
    cond_col = find_first_col(names, {'condition', 'cond', 'state', 'task', 'block'});
    part_col = find_first_col(names, {'participant', 'subject', 'sub', 'subj', 'id', 'participant_id', 'subject_id'});

    if isempty(file_col) || isempty(group_col) || isempty(cond_col)
        error(['Manifest must contain file_path, group, and condition columns. ', ...
               'Accepted file aliases: file, filepath, path, eeg_file, eeg_path, filename.']);
    end

    [csv_dir, ~, ~] = fileparts(which_or_absolute(manifest_csv));
    file_path = cellstr(string(T.(names{file_col})));
    for i = 1:numel(file_path)
        file_path{i} = resolve_manifest_path(file_path{i}, csv_dir);
    end

    group = clean_label_cell(T.(names{group_col}), 'group');
    condition = clean_label_cell(T.(names{cond_col}), 'condition');
    if isempty(part_col)
        participant = cell(numel(file_path), 1);
        for i = 1:numel(file_path)
            participant{i} = base_name(file_path{i});
        end
    else
        participant = clean_label_cell(T.(names{part_col}), 'participant');
    end

    manifest = table(file_path(:), participant(:), group(:), condition(:), ...
        'VariableNames', {'file_path', 'participant', 'group', 'condition'});

    for i = 1:height(manifest)
        if ~isfile(manifest.file_path{i})
            error('EEG file in manifest row %d was not found: %s', i, manifest.file_path{i});
        end
    end
end

function col = find_first_col(names, aliases)
    col = [];
    for a = 1:numel(aliases)
        idx = find(strcmp(names, matlab.lang.makeValidName(lower(aliases{a}))), 1, 'first');
        if ~isempty(idx)
            col = idx;
            return;
        end
    end
end

function labels = clean_label_cell(x, fallback_prefix)
    labels = cellstr(string(x));
    for i = 1:numel(labels)
        labels{i} = strtrim(labels{i});
        if isempty(labels{i}) || strcmpi(labels{i}, '<missing>') || strcmpi(labels{i}, 'missing')
            labels{i} = sprintf('%s_%03d', fallback_prefix, i);
        end
    end
end

function pth = resolve_manifest_path(pth, csv_dir)
    pth = char(strtrim(string(pth)));
    if isempty(pth)
        error('Empty file path in manifest.');
    end
    if is_absolute_path(pth)
        return;
    end
    candidate = fullfile(csv_dir, pth);
    if isfile(candidate)
        pth = candidate;
    else
        pth = char(java.io.File(pth).getAbsolutePath());
    end
end

function tf = is_absolute_path(pth)
    tf = startsWith(pth, filesep) || ...
         (~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'))) || ...
         startsWith(pth, '\\');
end

function out = which_or_absolute(pth)
    pth = char(pth);
    if isfile(pth)
        out = pth;
    else
        w = which(pth);
        if isempty(w)
            out = pth;
        else
            out = w;
        end
    end
end

function nm = base_name(pth)
    [~, nm, ~] = fileparts(pth);
end

function write_manifest_copy(manifest, out_file)
    writetable(manifest, out_file);
end

% =======================================================================
% Channel inspection and EEG loading
% =======================================================================

function file_meta = inspect_channel_sets(manifest, cfg)
    n_files = height(manifest);
    file_meta = repmat(struct('file_path', '', 'channel_labels', {{}}, 'chanlocs', [], 'pos', [], 'sfreq', NaN), n_files, 1);
    for i = 1:n_files
        [X, sfreq, chanlocs, labels, pos] = load_eeg_matrix(manifest.file_path{i}, cfg.sfreq);
        if cfg.use_scalp_channels && ~isempty(chanlocs)
            mask = scalp_channel_mask(chanlocs, size(X, 1));
            if any(mask)
                labels = labels(mask);
                chanlocs = chanlocs(mask);
                if ~isempty(pos), pos = pos(mask, :); end
            end
        end
        file_meta(i).file_path = manifest.file_path{i};
        file_meta(i).channel_labels = labels(:)';
        file_meta(i).chanlocs = chanlocs;
        file_meta(i).pos = pos;
        file_meta(i).sfreq = sfreq;
        clear X;
    end
end

function [common_labels, common_index_by_file, common_chanlocs, common_pos] = define_common_channels(file_meta, cfg)
    n_files = numel(file_meta);
    ref_labels = file_meta(1).channel_labels;
    ref_can = canonical_channel_labels(ref_labels);

    switch cfg.channel_policy
        case 'strict'
            common_labels = ref_labels;
            common_index_by_file = cell(n_files, 1);
            common_index_by_file{1} = 1:numel(ref_labels);
            for i = 2:n_files
                if numel(file_meta(i).channel_labels) ~= numel(ref_labels) || ...
                        ~all(strcmp(canonical_channel_labels(file_meta(i).channel_labels), ref_can))
                    error('Strict channel policy failed at file %d: %s', i, file_meta(i).file_path);
                end
                common_index_by_file{i} = 1:numel(ref_labels);
            end
        case 'intersect'
            common_can = ref_can;
            for i = 2:n_files
                this_can = canonical_channel_labels(file_meta(i).channel_labels);
                common_can = intersect(common_can, this_can, 'stable');
            end
            if isempty(common_can)
                error('No common channel labels across files.');
            end
            common_labels = cell(size(common_can));
            for j = 1:numel(common_can)
                ref_idx = find(strcmp(ref_can, common_can{j}), 1, 'first');
                common_labels{j} = ref_labels{ref_idx};
            end
            common_index_by_file = cell(n_files, 1);
            for i = 1:n_files
                this_can = canonical_channel_labels(file_meta(i).channel_labels);
                idx = nan(1, numel(common_can));
                for j = 1:numel(common_can)
                    idx(j) = find(strcmp(this_can, common_can{j}), 1, 'first');
                end
                common_index_by_file{i} = idx;
            end
        otherwise
            error('Unknown channel_policy: %s', cfg.channel_policy);
    end

    common_chanlocs = [];
    common_pos = [];
    if ~isempty(file_meta(1).chanlocs)
        ref_can = canonical_channel_labels(file_meta(1).channel_labels);
        idx = nan(1, numel(common_labels));
        common_can = canonical_channel_labels(common_labels);
        for j = 1:numel(common_can)
            idx(j) = find(strcmp(ref_can, common_can{j}), 1, 'first');
        end
        common_chanlocs = file_meta(1).chanlocs(idx);
    end
    if ~isempty(file_meta(1).pos)
        ref_can = canonical_channel_labels(file_meta(1).channel_labels);
        idx = nan(1, numel(common_labels));
        common_can = canonical_channel_labels(common_labels);
        for j = 1:numel(common_can)
            idx(j) = find(strcmp(ref_can, common_can{j}), 1, 'first');
        end
        common_pos = file_meta(1).pos(idx, :);
    elseif ~isempty(common_chanlocs)
        common_pos = pos_from_chanlocs(common_chanlocs, numel(common_chanlocs));
    end

    if cfg.use_scalp_channels
        [common_labels, common_index_by_file, common_chanlocs, common_pos] = ...
            prune_common_channels_by_geometry(common_labels, common_index_by_file, common_chanlocs, common_pos);
    end
end

function [eeg_data, sfreq, chanlocs, channel_labels, pos] = load_eeg_matrix(eeg_file, fallback_sfreq)
    [~, ~, ext] = fileparts(eeg_file);
    ext = lower(ext);
    chanlocs = [];
    pos = [];

    switch ext
        case '.set'
            if exist('pop_loadset', 'file') ~= 2
                error('EEGLAB pop_loadset not found. Add EEGLAB to the MATLAB path before loading .set files.');
            end
            EEG = pop_loadset(eeg_file);
            eeg_data = double(EEG.data);
            sfreq = double(EEG.srate);
            if isfield(EEG, 'chanlocs')
                chanlocs = EEG.chanlocs;
            end
            channel_labels = channel_labels_from_chanlocs(chanlocs, size(eeg_data, 1));
            pos = pos_from_chanlocs(chanlocs, size(eeg_data, 1));

        case '.mat'
            S = load(eeg_file);
            if isfield(S, 'eeg_data')
                eeg_data = double(S.eeg_data);
            elseif isfield(S, 'data')
                eeg_data = double(S.data);
            elseif isfield(S, 'EEG') && isfield(S.EEG, 'data')
                eeg_data = double(S.EEG.data);
            else
                error('Could not find EEG data in .mat file. Expected eeg_data, data, or EEG.data: %s', eeg_file);
            end
            if ndims(eeg_data) > 2
                eeg_data = reshape(eeg_data, size(eeg_data, 1), []);
            end
            if isfield(S, 'sfreq')
                sfreq = double(S.sfreq);
            elseif isfield(S, 'srate')
                sfreq = double(S.srate);
            elseif isfield(S, 'EEG') && isfield(S.EEG, 'srate')
                sfreq = double(S.EEG.srate);
            else
                sfreq = fallback_sfreq;
            end
            if isfield(S, 'chanlocs')
                chanlocs = S.chanlocs;
            elseif isfield(S, 'EEG') && isfield(S.EEG, 'chanlocs')
                chanlocs = S.EEG.chanlocs;
            end
            if isfield(S, 'channel_labels')
                channel_labels = cellstr(string(S.channel_labels));
            elseif isfield(S, 'ch_labels')
                channel_labels = cellstr(string(S.ch_labels));
            else
                channel_labels = channel_labels_from_chanlocs(chanlocs, size(eeg_data, 1));
            end
            if isfield(S, 'pos')
                pos = double(S.pos);
            else
                pos = pos_from_chanlocs(chanlocs, size(eeg_data, 1));
            end

        otherwise
            error('Unsupported EEG file extension: %s. Use .set or .mat.', ext);
    end

    if size(eeg_data, 1) > size(eeg_data, 2) && size(eeg_data, 2) <= 512
        warning('EEG matrix has more rows than columns; assuming data are channels x time. Check orientation for %s.', eeg_file);
    end
end

function labels = channel_labels_from_chanlocs(chanlocs, n_channels)
    util = microstate_utilities();
    labels = util.channel_labels_from_chanlocs(chanlocs, n_channels);
    labels = labels(:);
end

function pos = pos_from_chanlocs(chanlocs, n_channels)
    util = microstate_utilities();
    pos = util.positions_from_chanlocs(chanlocs, n_channels);
    if all(isnan(pos(:)))
        pos = [];
    end
end

function mask = scalp_channel_mask(chanlocs, n_channels)
    util = microstate_utilities();
    mask = util.scalp_channel_mask(chanlocs, n_channels);
end

function [common_labels, common_index_by_file, common_chanlocs, common_pos] = prune_common_channels_by_geometry(common_labels, common_index_by_file, common_chanlocs, common_pos)
    n_channels = numel(common_labels);
    keep = true(1, n_channels);

    if ~isempty(common_pos)
        pos = double(common_pos);
        if size(pos, 1) == 3 && size(pos, 2) == n_channels
            pos = pos';
        end
        if size(pos, 1) == n_channels && size(pos, 2) == 3
            keep = keep & all(isfinite(pos), 2)' & (sqrt(sum(pos.^2, 2))' > eps);
            common_pos = pos;
        end
    end

    if ~isempty(common_chanlocs)
        keep = keep & common_chanloc_location_mask(common_chanlocs);
    end

    if ~any(keep)
        error('No common channels with usable scalp geometry remain after pruning.');
    end

    common_labels = common_labels(keep);
    for i = 1:numel(common_index_by_file)
        common_index_by_file{i} = common_index_by_file{i}(keep);
    end
    if ~isempty(common_chanlocs)
        common_chanlocs = common_chanlocs(keep);
    end
    if ~isempty(common_pos)
        common_pos = common_pos(keep, :);
    end
end

function keep = common_chanloc_location_mask(chanlocs)
    keep = false(1, numel(chanlocs));
    for i = 1:numel(chanlocs)
        has_xyz = isfield(chanlocs(i), 'X') && ~isempty(chanlocs(i).X) && ...
            isfield(chanlocs(i), 'Y') && ~isempty(chanlocs(i).Y) && ...
            isfield(chanlocs(i), 'Z') && ~isempty(chanlocs(i).Z) && ...
            all(isfinite(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]))) && ...
            norm(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z])) > eps;
        has_polar = isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && ...
            isfield(chanlocs(i), 'radius') && ~isempty(chanlocs(i).radius) && ...
            isfinite(double(chanlocs(i).theta)) && isfinite(double(chanlocs(i).radius)) && ...
            double(chanlocs(i).radius) <= 0.75;
        keep(i) = has_xyz || has_polar;
    end
end

function can = canonical_channel_labels(labels)
    util = microstate_utilities();
    can = util.canonical_channel_labels(labels);
end

function [templates_by_K, template_info] = load_template_initial_centers_by_K(cfg, common_labels, common_pos)
    templates_by_K = struct('K', {}, 'centers', {}, 'labels', {}, 'matched_channel_labels', {}, ...
        'channel_match_mode', {}, 'template_channel_labels', {});
    template_info = struct('K', {}, 'labels', {}, 'n_channels', {}, 'channel_match_mode', {}, ...
        'template_file', {});

    if ~cfg.use_template_initialisation
        return;
    end
    if exist('load_metamaps_templates', 'file') ~= 2
        warning('load_metamaps_templates.m not found. Canonical template initialisation disabled.');
        return;
    end
    if ~isfile(cfg.template_file)
        warning('Template file not found: %s. Canonical template initialisation disabled.', cfg.template_file);
        return;
    end

    K_values = unique(round(cfg.K_candidates(:)'));
    K_values = K_values(K_values >= 4 & K_values <= 7);
    for i = 1:numel(K_values)
        K = K_values(i);
        try
            [template_maps, template_labels, template_channel_labels, template_chanlocs] = ...
                load_metamaps_templates(cfg.template_file, 'K', K);
            [template_maps, match_mode, matched_channel_labels] = ...
                remap_template_maps_to_common_channels(template_maps, template_channel_labels, common_labels, ...
                template_chanlocs, common_pos);
            template_maps = normalize_maps_local(template_maps);

            templates_by_K(end+1).K = K; %#ok<AGROW>
            templates_by_K(end).centers = template_maps;
            templates_by_K(end).labels = template_labels;
            templates_by_K(end).matched_channel_labels = matched_channel_labels;
            templates_by_K(end).channel_match_mode = match_mode;
            templates_by_K(end).template_channel_labels = template_channel_labels;

            template_info(end+1).K = K; %#ok<AGROW>
            template_info(end).labels = template_labels;
            template_info(end).n_channels = size(template_maps, 2);
            template_info(end).channel_match_mode = match_mode;
            template_info(end).template_file = cfg.template_file;
        catch ME
            warning('Canonical template initialisation skipped for K=%d: %s', K, ME.message);
        end
    end
end

function template_reference = load_template_peak_reference(cfg, common_labels, common_pos)
    template_reference = struct('centers', [], 'labels', {{}}, 'channel_match_mode', '');
    if ~cfg.reject_template_misaligned_peaks
        return;
    end
    if exist('load_metamaps_templates', 'file') ~= 2 || ~isfile(cfg.template_file)
        warning('Template peak filtering disabled because MetaMaps loader/file is unavailable.');
        return;
    end

    try
        [template_maps, template_labels, template_channel_labels, template_chanlocs] = ...
            load_metamaps_templates(cfg.template_file, 'K', round(cfg.template_peak_template_K));
        [template_maps, match_mode] = ...
            remap_template_maps_to_common_channels(template_maps, template_channel_labels, common_labels, ...
            template_chanlocs, common_pos);
        template_reference.centers = normalize_maps_local(template_maps);
        template_reference.labels = template_labels;
        template_reference.channel_match_mode = match_mode;
    catch ME
        warning('Template peak filtering disabled: %s', ME.message);
    end
end

function cfg = ensure_spm_mix_available(cfg)
    if ~cfg.use_spm_initialisation || exist('spm_mix', 'file') == 2
        return;
    end

    candidate_paths = {};
    if ~isempty(cfg.spm_path)
        candidate_paths{end+1} = cfg.spm_path; %#ok<AGROW>
    end
    if isfield(cfg, 'config_spm_mixture_paths') && ~isempty(cfg.config_spm_mixture_paths)
        candidate_paths = [candidate_paths, cellstr(string(cfg.config_spm_mixture_paths(:)'))]; %#ok<AGROW>
    end
    candidate_paths = [candidate_paths, { ...
        fullfile(getenv('HOME'), 'spm', 'toolbox', 'mixture'), ...
        fullfile(getenv('HOME'), 'Downloads', 'spm', 'toolbox', 'mixture'), ...
        fullfile(getenv('HOME'), 'fieldtrip-20251118', 'external', 'spm12', 'toolbox', 'mixture')}]; %#ok<AGROW>

    for i = 1:numel(candidate_paths)
        pth = candidate_paths{i};
        if ~isempty(pth) && exist(pth, 'dir')
            addpath(pth);
            if exist('spm_mix', 'file') == 2
                cfg.spm_path = pth;
                return;
            end
        end
    end

    if cfg.require_spm_initialisation
        error(['SPM spm_mix was not found. Set ''spm_path'' to the SPM mixture toolbox ', ...
               'or set ''require_spm_initialisation'', false to allow topographic fallback.']);
    end
end

function [template_maps_common, match_mode, matched_channel_labels] = remap_template_maps_to_common_channels(template_maps, template_channel_labels, common_labels, template_chanlocs, common_pos)
    template_maps = double(template_maps);
    n_template_channels = size(template_maps, 2);
    common_labels = cellstr(string(common_labels(:)'));
    matched_channel_labels = common_labels;
    if nargin < 4
        template_chanlocs = [];
    end
    if nargin < 5
        common_pos = [];
    end

    if isempty(template_channel_labels)
        [template_maps_common, ok] = interpolate_template_maps_by_geometry(template_maps, template_chanlocs, common_pos);
        if ok
            match_mode = 'geometry_interpolation';
            warning(['Template channel labels are unavailable. Using electrode-position ', ...
                     'interpolation for canonical initialisation.']);
            return;
        elseif n_template_channels ~= numel(common_labels)
            error(['Template channel labels are unavailable, geometry interpolation failed, ', ...
                   'and template/data channel counts differ (%d vs %d).'], n_template_channels, numel(common_labels));
        end
        template_maps_common = template_maps;
        match_mode = 'count_only';
        return;
    end

    template_channel_labels = cellstr(string(template_channel_labels(:)'));
    n_template_labels = min(numel(template_channel_labels), n_template_channels);
    template_channel_labels = template_channel_labels(1:n_template_labels);
    template_can = canonical_channel_labels(template_channel_labels);
    common_can = canonical_channel_labels(common_labels);

    idx_template = nan(1, numel(common_can));
    all_found = true;
    for j = 1:numel(common_can)
        hit = find(strcmp(template_can, common_can{j}), 1, 'first');
        if isempty(hit)
            all_found = false;
            break;
        end
        idx_template(j) = hit;
    end

    if all_found
        template_maps_common = template_maps(:, idx_template);
        match_mode = 'labels';
        matched_channel_labels = common_labels;
        return;
    end

    [template_maps_common, ok] = interpolate_template_maps_by_geometry(template_maps, template_chanlocs, common_pos);
    if ok
        match_mode = 'geometry_interpolation';
        warning(['Template channel labels do not overlap the hierarchy labels. Using ', ...
                 'electrode-position interpolation for canonical initialisation.']);
        return;
    end

    if n_template_channels == numel(common_labels)
        template_maps_common = template_maps;
        match_mode = 'position_fallback';
        warning(['Template channel labels do not fully match the hierarchy common labels. ', ...
                 'Falling back to positional channel order for canonical initialisation.']);
        return;
    end

    error(['Could not remap template channels to the hierarchy common channels. ', ...
           'Matched %d/%d labels and channel counts differ (%d vs %d).'], ...
          sum(~isnan(idx_template)), numel(common_can), n_template_channels, numel(common_labels));
end

function [template_maps_common, ok] = interpolate_template_maps_by_geometry(template_maps, template_chanlocs, common_pos)
    template_maps_common = [];
    ok = false;

    if isempty(template_chanlocs) || isempty(common_pos)
        return;
    end

    template_pos = pos_from_chanlocs(template_chanlocs, size(template_maps, 2));
    if isempty(template_pos) || size(common_pos, 1) == 0
        return;
    end

    common_pos = double(common_pos);
    valid_template = all(isfinite(template_pos), 2) & all(isfinite(template_maps'), 2);
    valid_common = all(isfinite(common_pos), 2);
    if sum(valid_template) < 8 || sum(valid_common) < 8
        return;
    end

    source_pos = template_pos(valid_template, :);
    target_pos = common_pos(valid_common, :);
    source_maps = template_maps(:, valid_template);

    weights = inverse_distance_weights(target_pos, source_pos, 2);
    interpolated = zeros(size(template_maps, 1), size(common_pos, 1));
    interpolated(:, valid_common) = source_maps * weights';

    if any(~valid_common)
        nearest_idx = nearest_position_indices(common_pos(~valid_common, :), source_pos);
        interpolated(:, ~valid_common) = source_maps(:, nearest_idx);
    end

    template_maps_common = interpolated;
    ok = true;
end

function weights = inverse_distance_weights(target_pos, source_pos, power_value)
    n_target = size(target_pos, 1);
    n_source = size(source_pos, 1);
    dist2 = zeros(n_target, n_source);
    for d = 1:size(source_pos, 2)
        delta = target_pos(:, d) - source_pos(:, d)';
        dist2 = dist2 + delta .^ 2;
    end
    dist = sqrt(max(dist2, 0));
    weights = zeros(n_target, n_source);
    for i = 1:n_target
        [min_dist, min_idx] = min(dist(i, :));
        if min_dist < 1e-8
            weights(i, min_idx) = 1;
        else
            w = 1 ./ max(dist(i, :), 1e-6) .^ power_value;
            weights(i, :) = w ./ sum(w);
        end
    end
end

function nearest_idx = nearest_position_indices(target_pos, source_pos)
    nearest_idx = ones(size(target_pos, 1), 1);
    for i = 1:size(target_pos, 1)
        if ~all(isfinite(target_pos(i, :)))
            nearest_idx(i) = 1;
            continue;
        end
        dist2 = sum((source_pos - target_pos(i, :)) .^ 2, 2);
        [~, nearest_idx(i)] = min(dist2);
    end
end

function pre_cfg = build_preprocessing_config_from_cfg(cfg)
    pre_cfg = struct();
    pre_cfg.apply_average_reference = cfg.apply_average_reference;
    pre_cfg.filter_band = cfg.filter_band;
    pre_cfg.spatial_filter = cfg.spatial_filter;
    pre_cfg.spatial_filter_matrix = cfg.spatial_filter_matrix;
    pre_cfg.spatial_filter_neighbours = cfg.spatial_filter_neighbours;
    pre_cfg.spatial_filter_strength = cfg.spatial_filter_strength;
    pre_cfg.gfp_peak_min_distance = 3;
    pre_cfg.gfp_peak_threshold_schedule = [0.50, 0.60, 0.70, 0.80, 0.90];
    pre_cfg.reject_gfp_peak_outliers = cfg.reject_gfp_peak_outliers_individual;
    pre_cfg.gfp_outlier_mad_multiplier = cfg.gfp_outlier_mad_multiplier_individual;
    pre_cfg.gfp_outlier_upper_quantile = 0.995;
    pre_cfg.min_peak_count_after_gfp_rejection = 3;
end

function [records, summary] = apply_population_gfp_outlier_filter(records, cfg)
    summary = struct('enabled', cfg.reject_gfp_peak_outliers_population, ...
        'threshold', NaN, 'n_removed', 0, 'n_total', 0, 'mode', 'none');

    if ~cfg.reject_gfp_peak_outliers_population
        return;
    end

    pooled = [];
    counts = zeros(numel(records), 1);
    for i = 1:numel(records)
        counts(i) = numel(records(i).peak_gfp_values);
        if counts(i) > 0
            pooled = [pooled; records(i).peak_gfp_values(:)]; %#ok<AGROW>
        end
    end

    summary.n_total = numel(pooled);
    if numel(pooled) < 5
        return;
    end

    [keep_all, threshold, mode] = robust_upper_outlier_mask_local( ...
        pooled, cfg.gfp_outlier_mad_multiplier_population, 0.995, 3);
    summary.threshold = threshold;
    summary.mode = mode;
    summary.n_removed = 0;

    cursor = 0;
    for i = 1:numel(records)
        n_i = counts(i);
        if n_i == 0
            continue;
        end
        keep_i = keep_all((cursor + 1):(cursor + n_i));
        cursor = cursor + n_i;
        if sum(keep_i) < 3
            keep_i = true(size(keep_i));
        end
        summary.n_removed = summary.n_removed + sum(~keep_i);
        records(i) = subset_record_peaks(records(i), keep_i, threshold, mode);
    end
end

function rec = apply_template_peak_filter(rec, cfg)
    if ~cfg.reject_template_misaligned_peaks || ...
            ~isfield(cfg, 'template_peak_reference') || ...
            isempty(cfg.template_peak_reference) || ...
            ~isfield(cfg.template_peak_reference, 'centers') || ...
            isempty(cfg.template_peak_reference.centers) || ...
            isempty(rec.maps_norm)
        return;
    end

    refs = normalize_maps_local(cfg.template_peak_reference.centers);
    maps = normalize_maps_local(rec.maps_norm);
    max_corr = max(abs(maps * refs'), [], 2);
    keep = max_corr >= cfg.template_peak_min_abs_corr;

    rec.preprocessing.template_peak_max_abs_corr = max_corr;
    rec.preprocessing.template_peak_filter_threshold = cfg.template_peak_min_abs_corr;
    rec.preprocessing.template_peak_filter_removed = sum(~keep);
    rec.preprocessing.template_peak_filter_mode = 'metamaps_similarity';

    if sum(keep) < cfg.min_peak_count_after_template_rejection
        rec.preprocessing.template_peak_filter_mode = 'skipped_min_keep';
        return;
    end

    rec = subset_record_peak_rows(rec, keep);
end

function rec = subset_record_peak_rows(rec, keep_mask)
    keep_mask = logical(keep_mask(:));
    if isempty(rec.maps_norm) || numel(keep_mask) ~= size(rec.maps_norm, 1)
        return;
    end

    rec.maps_norm = rec.maps_norm(keep_mask, :);
    if ~isempty(rec.maps_original)
        rec.maps_original = rec.maps_original(keep_mask, :);
    end
    if ~isempty(rec.idx_peaks)
        rec.idx_peaks = rec.idx_peaks(keep_mask);
    end
    if ~isempty(rec.peak_gfp_values)
        rec.peak_gfp_values = rec.peak_gfp_values(keep_mask);
    end
    rec.n_maps = size(rec.maps_norm, 1);
    rec.n_fit_maps = rec.n_maps;
    rec.maps_fit = rec.maps_norm;
end

function rec = subset_record_peaks(rec, keep_mask, threshold, mode)
    if isempty(rec.maps_norm)
        return;
    end
    keep_mask = logical(keep_mask(:));
    if numel(keep_mask) ~= size(rec.maps_norm, 1)
        return;
    end

    rec = subset_record_peak_rows(rec, keep_mask);
    rec.preprocessing.population_gfp_outlier_threshold = threshold;
    rec.preprocessing.population_gfp_outlier_mode = mode;
    rec.preprocessing.population_gfp_outliers_removed = sum(~keep_mask);
end

function rec = finalize_record_fit_maps(rec, max_maps_per_file)
    maps_fit = rec.maps_norm;
    if ~isempty(max_maps_per_file) && size(maps_fit, 1) > max_maps_per_file
        idx = deterministic_subsample(size(maps_fit, 1), max_maps_per_file);
        maps_fit = maps_fit(idx, :);
    end
    rec.maps_fit = maps_fit;
    rec.n_fit_maps = size(maps_fit, 1);
end

function [keep_mask, threshold, mode] = robust_upper_outlier_mask_local(x, mad_multiplier, upper_quantile, min_keep)
    x = double(x(:));
    keep_mask = true(size(x));
    threshold = NaN;
    mode = 'none';
    if numel(x) < max(5, min_keep)
        return;
    end
    med_x = median(x);
    mad_x = median(abs(x - med_x));
    if isfinite(mad_x) && mad_x > eps
        threshold = med_x + mad_multiplier * 1.4826 * mad_x;
        mode = 'mad';
    else
        threshold = quantile(x, upper_quantile);
        mode = 'quantile_fallback';
    end
    if ~isfinite(threshold)
        threshold = NaN;
        mode = 'none';
        return;
    end
    keep_mask = x <= threshold;
    if sum(keep_mask) < min_keep
        keep_mask(:) = true;
    end
end

function rec = extract_file_maps(eeg_file, chan_idx, common_labels, common_chanlocs, common_pos, cfg, util)
    [X, sfreq, ~, ~, ~] = load_eeg_matrix(eeg_file, cfg.sfreq);
    X = double(X(chan_idx, :));
    Sim = struct();
    Sim.X_noisy = X;
    Sim.X_clean = X;
    Sim.sfreq = sfreq;
    Sim.duration_s = size(X, 2) / sfreq;
    Sim.n_channels = size(X, 1);
    Sim.n_samples = size(X, 2);
    Sim.chanlocs = common_chanlocs;
    Sim.channel_labels = common_labels(:);
    Sim.pos = common_pos;
    Sim.maps_true = [];
    Sim.K_true = NaN;
    Sim.SNR_dB = NaN;
    Sim.preprocessing = build_preprocessing_config_from_cfg(cfg);

    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, maps_original, preproc_info] = util.preprocess_maps(Sim);
    peak_gfp_values = [];
    if isfield(preproc_info, 'peak_gfp_values')
        peak_gfp_values = preproc_info.peak_gfp_values(:);
    end

    rec = empty_record_struct();
    rec.file_path = eeg_file;
    rec.sfreq = sfreq;
    rec.duration_s = Sim.duration_s;
    rec.n_channels = C_dims;
    rec.n_samples = size(X, 2);
    rec.n_maps = n_maps;
    rec.n_fit_maps = n_maps;
    rec.idx_peaks = idx_peaks;
    rec.gfp_vec = gfp_vec;
    rec.peak_gfp_values = peak_gfp_values;
    rec.maps_norm = maps_norm;
    rec.maps_original = maps_original;
    rec.maps_fit = maps_norm;
    rec.preprocessing = preproc_info;
end

function rec = empty_record_struct()
    rec = struct('file_index', NaN, 'file_path', '', 'participant', '', 'group', '', 'condition', '', ...
        'sfreq', NaN, 'duration_s', NaN, 'n_channels', NaN, 'n_samples', NaN, ...
        'n_maps', NaN, 'n_fit_maps', NaN, 'idx_peaks', [], 'gfp_vec', [], ...
        'peak_gfp_values', [], 'maps_norm', [], 'maps_original', [], 'maps_fit', [], ...
        'preprocessing', struct());
end

% =======================================================================
% Hierarchical fitting
% =======================================================================

function node = fit_node_or_inherit(name, level, group, condition, participant, file_idx, maps, K, parent_centers, prior_weight, cfg, global_centers)
    node = empty_node_struct();
    node.name = name;
    node.level = level;
    node.group = group;
    node.condition = condition;
    node.participant = participant;
    node.file_indices = file_idx(:)';
    node.n_maps = size(maps, 1);
    node.K_estimated = K;
    node.prior_weight = prior_weight;

    if isempty(maps)
        node.centers = parent_centers;
        node.labels = [];
        node.inherited = true;
        node.warning = 'No maps available; inherited parent templates.';
        return;
    end

    if size(maps, 1) < cfg.min_maps_per_node
        labels = assign_by_abs_correlation(maps, parent_centers);
        metrics = compute_fit_metrics(maps, labels, parent_centers, NaN);
        node.centers = parent_centers;
        node.labels = labels;
        node.metrics = metrics;
        node.inherited = true;
        node.warning = sprintf('Only %d maps (< min_maps_per_node=%d); assigned to parent templates without refitting.', ...
            size(maps, 1), cfg.min_maps_per_node);
        return;
    end

    fit = fit_microstate_maps_fixedK(maps, K, parent_centers, prior_weight, cfg, name);
    if ~isempty(parent_centers)
        [fit.centers, fit.labels] = align_centers_to_reference(fit.centers, fit.labels, parent_centers);
    elseif ~isempty(global_centers)
        [fit.centers, fit.labels] = align_centers_to_reference(fit.centers, fit.labels, global_centers);
    end
    node.centers = fit.centers;
    node.labels = fit.labels;
    node.metrics = fit.metrics;
    node.fit = rmfield_if_present(fit, {'centers', 'labels', 'metrics'});
    node.inherited = false;
end

function H = fit_microstate_maps_hierarchical(maps_norm, K_candidates, criterion, prior_centers, prior_weight, cfg, node_name)
    nK = numel(K_candidates);
    fits = cell(nK, 1);
    free_energy_vals = nan(nK, 1);
    silhouette_vals = nan(nK, 1);
    gev_vals = nan(nK, 1);
    within_ss = nan(nK, 1);

    for iK = 1:nK
        K = K_candidates(iK);
        if cfg.verbose
            fprintf('   %s K=%d... ', node_name, K);
        end
        fits{iK} = fit_microstate_maps_fixedK(maps_norm, K, prior_centers, prior_weight, cfg, sprintf('%s_K%d', node_name, K));
        free_energy_vals(iK) = fits{iK}.metrics.free_energy;
        silhouette_vals(iK) = fits{iK}.metrics.silhouette;
        gev_vals(iK) = fits{iK}.metrics.gev;
        within_ss(iK) = fits{iK}.metrics.within_ss;
        if cfg.verbose
            fprintf('F=%.2f, Sil=%.3f, GEV=%.3f, WSS=%.1f\n', ...
                free_energy_vals(iK), silhouette_vals(iK), gev_vals(iK), within_ss(iK));
        end
    end

    [K_est, best_idx, best_score] = select_K_global(K_candidates, criterion, free_energy_vals, silhouette_vals, gev_vals, within_ss);
    best_fit = fits{best_idx};

    H = struct();
    H.name = node_name;
    H.level = 'global';
    H.K_candidates = K_candidates(:)';
    H.K_estimated = K_est;
    H.best_criterion_value = best_score;
    H.criterion = criterion;
    H.free_energy_vals = free_energy_vals;
    H.silhouette_vals = silhouette_vals;
    H.gev_vals = gev_vals;
    H.within_ss = within_ss;
    H.fits = fits;
    H.centers = best_fit.centers;
    H.labels = best_fit.labels;
    H.metrics = best_fit.metrics;
    H.n_maps = size(maps_norm, 1);
end

function fit = fit_microstate_maps_fixedK(maps_norm, K, prior_centers, prior_weight, cfg, fit_name)
    maps_norm = normalize_maps_local(maps_norm);
    N = size(maps_norm, 1);
    C = size(maps_norm, 2);
    canonical_centers = get_canonical_template_centers_for_K(cfg, K, C);
    init_reference = prior_centers;
    if isempty(init_reference)
        init_reference = canonical_centers;
    end
    refinement_prior_centers = prior_centers;
    refinement_prior_weight = prior_weight;
    used_canonical_template_prior = false;
    if isempty(refinement_prior_centers) && ~isempty(canonical_centers) && cfg.canonical_prior_weight_global > 0
        refinement_prior_centers = canonical_centers;
        refinement_prior_weight = cfg.canonical_prior_weight_global;
        used_canonical_template_prior = true;
    end
    if N == 0
        error('Cannot fit fixed K=%d to empty map matrix.', K);
    end
    if K > N
        warning('K=%d exceeds number of maps N=%d for %s. Some states will be prior/random initialised.', K, N, fit_name);
    end

    used_spm = false;
    spm_error = '';
    free_energy = NaN;

    try
        if cfg.use_spm_initialisation && exist('spm_mix', 'file') == 2 && N > K
            [features, feature_info] = pca_projective_features(maps_norm);
            features_aug = features;
            if ~isempty(init_reference) && cfg.spm_prior_pseudocount > 0
                init_reference = normalize_maps_local(init_reference);
                prior_features = project_maps_to_projective_features(init_reference, feature_info);
                n_rep = max(1, round(cfg.spm_prior_pseudocount));
                features_aug = [features; repelem(prior_features, n_rep, 1)]; %#ok<AGROW>
            end
            evalc('vbmix = spm_mix(features_aug, K, 0);');
            if is_valid_spm_mix_result(vbmix)
                labels0 = assign_spm_samples(features, vbmix);
                centers0 = recover_centers_with_prior(maps_norm, labels0, K, refinement_prior_centers, refinement_prior_weight);
                free_energy = vbmix.fm;
                used_spm = true;
            else
                error('spm_mix returned an invalid mixture structure.');
            end
        else
            error('SPM initialisation disabled, unavailable, or N<=K.');
        end
    catch ME
        if cfg.use_spm_initialisation && cfg.require_spm_initialisation
            rethrow(ME);
        end
        spm_error = ME.message;
        centers0 = initialise_centers_topographic(maps_norm, K, init_reference);
        labels0 = assign_by_abs_correlation(maps_norm, centers0);
        feature_info = struct();
    end

    if ~isempty(init_reference)
        [centers0, labels0] = align_centers_to_reference(centers0, labels0, init_reference);
    end
    [labels, centers] = refine_topographic_map(maps_norm, labels0, centers0, K, ...
        refinement_prior_centers, refinement_prior_weight, cfg.n_refine_iter);
    if isempty(prior_centers) && ~isempty(canonical_centers)
        [centers, labels] = align_centers_to_reference(centers, labels, canonical_centers);
    end

    metrics = compute_fit_metrics(maps_norm, labels, centers, free_energy);
    metrics.used_spm_initialisation = used_spm;
    metrics.spm_error = spm_error;
    metrics.prior_weight = refinement_prior_weight;
    metrics.initialisation_reference = describe_initialisation_reference(prior_centers, canonical_centers);
    metrics.used_canonical_template_initialisation = isempty(prior_centers) && ~isempty(canonical_centers);
    metrics.used_canonical_template_prior = used_canonical_template_prior;

    fit = struct();
    fit.name = fit_name;
    fit.K = K;
    fit.centers = centers;
    fit.labels = labels;
    fit.metrics = metrics;
    fit.feature_info = feature_info;
end

function centers = get_canonical_template_centers_for_K(cfg, K, n_channels)
    centers = [];
    if ~isfield(cfg, 'template_initial_centers_by_K') || isempty(cfg.template_initial_centers_by_K)
        return;
    end
    idx = find([cfg.template_initial_centers_by_K.K] == K, 1, 'first');
    if isempty(idx)
        return;
    end
    candidate = cfg.template_initial_centers_by_K(idx).centers;
    if size(candidate, 2) ~= n_channels
        return;
    end
    centers = candidate;
end

function txt = describe_initialisation_reference(prior_centers, canonical_centers)
    if ~isempty(prior_centers)
        txt = 'parent_prior';
    elseif ~isempty(canonical_centers)
        txt = 'canonical_template';
    else
        txt = 'data_driven';
    end
end

function [global_fit, global_selected] = attach_canonical_reporting_alignment(global_fit, global_selected, cfg, common_labels, common_pos)
    if ~isfile(cfg.template_file)
        return;
    end

    try
        template_K = cfg.canonical_reporting_template_K;
        if isempty(template_K)
            template_K = size(global_selected.centers, 1);
        end
        [template_maps, template_labels, template_channel_labels, template_chanlocs] = ...
            load_metamaps_templates(cfg.template_file, 'K', round(template_K));
        [template_maps, match_mode, matched_channel_labels] = ...
            remap_template_maps_to_common_channels(template_maps, template_channel_labels, common_labels, ...
            template_chanlocs, common_pos);
        alignment = align_maps_to_template_matrix(global_selected.centers, template_maps, ...
            template_labels, matched_channel_labels, match_mode);
        global_selected.template_alignment = alignment;
        global_selected.canonical_state_labels = alignment.labels;
        global_selected.centers = alignment.aligned_maps;
        global_selected.metrics = add_template_alignment_metrics(global_selected.metrics, alignment);

        global_fit.template_alignment = alignment;
        global_fit.canonical_state_labels = alignment.labels;
        global_fit.centers = alignment.aligned_maps;
        global_fit.metrics = add_template_alignment_metrics(global_fit.metrics, alignment);
    catch ME
        global_selected.template_alignment_error = ME.message;
        global_fit.template_alignment_error = ME.message;
    end
end

function alignment = align_maps_to_template_matrix(estimated_maps, template_maps, template_labels, matched_channel_labels, channel_match_mode)
    estimated_maps = normalize_maps_local(double(estimated_maps));
    template_maps = normalize_maps_local(double(template_maps));

    signed_corr = estimated_maps * template_maps';
    corr_matrix = abs(signed_corr);
    template_idx = optimal_template_assignment_local(corr_matrix);

    n_est = size(estimated_maps, 1);
    template_corr = zeros(n_est, 1);
    polarity = ones(n_est, 1);
    labels = cell(n_est, 1);
    for e = 1:n_est
        t = template_idx(e);
        if ~isnan(t)
            template_corr(e) = corr_matrix(e, t);
            polarity(e) = sign_nonzero(signed_corr(e, t));
            labels{e} = template_labels{t};
        else
            labels{e} = sprintf('X%d', e);
        end
    end

    aligned_maps = estimated_maps;
    for e = 1:n_est
        aligned_maps(e, :) = polarity(e) * aligned_maps(e, :);
    end

    valid_corr = template_corr(template_corr > 0 & isfinite(template_corr));
    if isempty(valid_corr)
        valid_corr = NaN;
    end
    strong_threshold = 0.5;
    alignment = struct( ...
        'labels', {labels}, ...
        'correlations', template_corr, ...
        'template_indices', template_idx, ...
        'polarity', polarity, ...
        'aligned_maps', aligned_maps, ...
        'corr_matrix', corr_matrix, ...
        'template_labels', {template_labels}, ...
        'channel_match_mode', channel_match_mode, ...
        'matched_channel_labels', {matched_channel_labels}, ...
        'estimated_channel_indices', 1:size(estimated_maps, 2), ...
        'template_channel_indices', 1:size(template_maps, 2), ...
        'n_common_channels', size(estimated_maps, 2), ...
        'mean_correlation', mean(valid_corr, 'omitnan'), ...
        'median_correlation', median(valid_corr, 'omitnan'), ...
        'min_correlation', min(valid_corr, [], 'omitnan'), ...
        'n_strong_matches', sum(template_corr >= strong_threshold), ...
        'strong_threshold', strong_threshold);
end

function template_idx = optimal_template_assignment_local(corr_matrix)
    [n_est, n_template] = size(corr_matrix);
    template_idx = nan(n_est, 1);

    if n_est <= n_template
        combos = nchoosek(1:n_template, n_est);
        best_score = -Inf;
        best_assignment = [];
        for ci = 1:size(combos, 1)
            perms_this = perms(combos(ci, :));
            for pi = 1:size(perms_this, 1)
                assignment = perms_this(pi, :);
                score = sum(corr_matrix(sub2ind(size(corr_matrix), 1:n_est, assignment)));
                if score > best_score
                    best_score = score;
                    best_assignment = assignment;
                end
            end
        end
        template_idx(:) = best_assignment(:);
        return;
    end

    combos = nchoosek(1:n_est, n_template);
    best_score = -Inf;
    best_est = [];
    best_assignment = [];
    for ci = 1:size(combos, 1)
        est_idx = combos(ci, :);
        perms_this = perms(1:n_template);
        for pi = 1:size(perms_this, 1)
            assignment = perms_this(pi, :);
            score = sum(corr_matrix(sub2ind(size(corr_matrix), est_idx, assignment)));
            if score > best_score
                best_score = score;
                best_est = est_idx;
                best_assignment = assignment;
            end
        end
    end
    template_idx(best_est) = best_assignment;
end

function s = sign_nonzero(x)
    s = sign(x);
    if s == 0
        s = 1;
    end
end

function metrics = add_template_alignment_metrics(metrics, alignment)
    if isempty(metrics) || ~isstruct(metrics)
        metrics = struct();
    end
    if isfield(alignment, 'mean_correlation')
        metrics.template_alignment_mean_correlation = alignment.mean_correlation;
    end
    if isfield(alignment, 'median_correlation')
        metrics.template_alignment_median_correlation = alignment.median_correlation;
    end
    if isfield(alignment, 'min_correlation')
        metrics.template_alignment_min_correlation = alignment.min_correlation;
    end
    if isfield(alignment, 'n_strong_matches')
        metrics.template_alignment_n_strong_matches = alignment.n_strong_matches;
    end
end

function [labels, centers] = refine_topographic_map(maps_norm, labels, centers, K, prior_centers, prior_weight, n_iter)
    if isempty(centers)
        centers = initialise_centers_topographic(maps_norm, K, prior_centers);
    end
    centers = normalize_maps_local(centers);
    for it = 1:max(1, n_iter)
        labels = assign_by_abs_correlation(maps_norm, centers);
        centers_new = recover_centers_with_prior(maps_norm, labels, K, prior_centers, prior_weight);
        if ~isempty(prior_centers)
            [centers_new, labels] = align_centers_to_reference(centers_new, labels, prior_centers);
        end
        delta = mean(1 - max(min(abs(sum(centers .* centers_new, 2)), 1), -1));
        centers = centers_new;
        if delta < 1e-7
            break;
        end
    end
    labels = assign_by_abs_correlation(maps_norm, centers);
end

function centers = recover_centers_with_prior(maps, labels, K, prior_centers, prior_weight)
    [N, C] = size(maps);
    centers = zeros(K, C);
    maps = normalize_maps_local(maps);
    has_prior = ~isempty(prior_centers) && size(prior_centers, 1) >= K && prior_weight > 0;
    if has_prior
        prior_centers = normalize_maps_local(prior_centers);
    end

    for k = 1:K
        idx = find(labels == k);
        if isempty(idx)
            if has_prior
                centers(k, :) = prior_centers(k, :);
                continue;
            else
                centers(k, :) = maps(randi(N), :);
                continue;
            end
        end
        Xk = maps(idx, :);
        cvm = Xk' * Xk;
        if has_prior
            p = prior_centers(k, :);
            cvm = cvm + prior_weight * (p' * p);
        end
        try
            [v, ~] = eigs(double(cvm), 1);
            c = real(v(:, 1)');
        catch
            [v, d] = eig(double(cvm));
            [~, best] = max(real(diag(d)));
            c = real(v(:, best)');
        end
        c = c - mean(c);
        c = c ./ (norm(c) + eps);
        if has_prior && abs(c * prior_centers(k, :)') < abs((-c) * prior_centers(k, :)')
            c = -c;
        end
        centers(k, :) = c;
    end
    centers = normalize_maps_local(centers);
end

function centers = initialise_centers_topographic(maps_norm, K, prior_centers)
    maps_norm = normalize_maps_local(maps_norm);
    [N, C] = size(maps_norm);
    centers = zeros(K, C);
    if ~isempty(prior_centers) && size(prior_centers, 1) >= K && size(prior_centers, 2) == C
        centers = normalize_maps_local(prior_centers(1:K, :));
        return;
    end
    first = randi(N);
    centers(1, :) = maps_norm(first, :);
    min_dist = ones(N, 1);
    for k = 2:K
        sim = abs(maps_norm * centers(1:k-1, :)');
        d = 1 - max(sim, [], 2).^2;
        min_dist = min(min_dist, d);
        probs = min_dist / (sum(min_dist) + eps);
        cdf = cumsum(probs);
        r = rand();
        idx = find(cdf >= r, 1, 'first');
        if isempty(idx), idx = randi(N); end
        centers(k, :) = maps_norm(idx, :);
    end
    centers = normalize_maps_local(centers);
end

function labels = assign_by_abs_correlation(maps_norm, centers)
    maps_norm = normalize_maps_local(maps_norm);
    centers = normalize_maps_local(centers);
    sim = abs(maps_norm * centers');
    [~, labels] = max(sim, [], 2);
end

function [centers_aligned, labels_aligned] = align_centers_to_reference(centers, labels, reference)
    centers = normalize_maps_local(centers);
    reference = normalize_maps_local(reference);
    K = min(size(reference, 1), size(centers, 1));
    C = size(centers, 2);
    centers_aligned = zeros(K, C);
    labels_aligned = labels;
    used = false(1, size(centers, 1));
    old_to_new = zeros(1, size(centers, 1));
    S = abs(reference(1:K, :) * centers');
    for k = 1:K
        vals = S(k, :);
        vals(used) = -Inf;
        [~, j] = max(vals);
        if isempty(j) || isinf(vals(j))
            j = find(~used, 1, 'first');
        end
        used(j) = true;
        c = centers(j, :);
        if c * reference(k, :)' < 0
            c = -c;
        end
        centers_aligned(k, :) = c;
        old_to_new(j) = k;
    end
    for old = 1:numel(old_to_new)
        if old_to_new(old) > 0
            labels_aligned(labels == old) = old_to_new(old);
        end
    end
end

function metrics = compute_fit_metrics(maps_norm, labels, centers, free_energy)
    maps_norm = normalize_maps_local(maps_norm);
    centers = normalize_maps_local(centers);
    K = size(centers, 1);
    N = size(maps_norm, 1);
    sim = abs(maps_norm * centers');
    [max_sim, ~] = max(sim, [], 2);
    gev = mean(max_sim .^ 2);
    within_ss = sum(1 - max_sim .^ 2);
    silhouette = silhouette_cosine_fast(maps_norm, labels, centers);
    proportions = zeros(1, K);
    mean_corr = zeros(1, K);
    for k = 1:K
        idx = labels == k;
        proportions(k) = mean(idx);
        if any(idx)
            mean_corr(k) = mean(abs(maps_norm(idx, :) * centers(k, :)'));
        else
            mean_corr(k) = NaN;
        end
    end
    entropy_nat = -sum(proportions(proportions > 0) .* log(proportions(proportions > 0)));

    metrics = struct();
    metrics.free_energy = free_energy;
    metrics.silhouette = silhouette;
    metrics.gev = gev;
    metrics.within_ss = within_ss;
    metrics.state_proportions = proportions;
    metrics.mean_state_abs_corr = mean_corr;
    metrics.mean_abs_corr = mean(max_sim);
    metrics.assignment_entropy_nat = entropy_nat;
    metrics.n_maps = N;
    metrics.K = K;
end

function sil = silhouette_cosine_fast(X, labels, centers)
    X = normalize_maps_local(X);
    centers = normalize_maps_local(centers);
    K = size(centers, 1);
    N = size(X, 1);
    if K < 2 || N < K + 1
        sil = NaN;
        return;
    end
    sim = abs(X * centers');
    dist = 1 - sim;
    a = nan(N, 1);
    b = nan(N, 1);
    for i = 1:N
        own = labels(i);
        a(i) = dist(i, own);
        other = dist(i, :);
        other(own) = Inf;
        b(i) = min(other);
    end
    s = (b - a) ./ max(a, b);
    s(~isfinite(s)) = NaN;
    sil = mean(s, 'omitnan');
end

function [K_est, best_idx, best_score] = select_K_global(K_candidates, criterion, fe, sil, gev, wss)
    criterion = lower(char(criterion));
    valid = true(size(K_candidates(:)));
    if strcmp(criterion, 'free_energy')
        valid = isfinite(fe(:));
    end
    if ~any(valid)
        error('No valid global fits for K selection.');
    end
    K_valid = K_candidates(valid);
    idx_valid = find(valid);

    switch criterion
        case 'free_energy'
            [best_score, ii] = max(fe(valid));
        case 'silhouette'
            [best_score, ii] = max(sil(valid));
        case 'gev'
            [best_score, ii] = max(gev(valid));
        case 'elbow'
            ii = elbow_index(wss(valid));
            best_score = wss(idx_valid(ii));
        case 'elbow_sil_combined'
            elbow_i = elbow_index(wss(valid));
            sil_norm = normalise01(sil(valid));
            elbow_score = zeros(numel(K_valid), 1);
            elbow_score(elbow_i) = 1;
            % Smoothly favour the elbow, but use silhouette to break near-ties.
            score = 0.65 * local_elbow_scores(wss(valid), K_valid(:)) + 0.35 * sil_norm(:);
            if all(~isfinite(score)) || all(score == score(1))
                score = elbow_score + 0.1 * sil_norm(:);
            end
            [best_score, ii] = max(score);
        otherwise
            error('Unknown criterion: %s', criterion);
    end
    best_idx = idx_valid(ii);
    K_est = K_candidates(best_idx);
end

function idx = elbow_index(y)
    y = y(:);
    n = numel(y);
    if n <= 2
        [~, idx] = min(y);
        return;
    end
    x = (1:n)';
    yn = normalise01(y);
    xn = normalise01(x);
    p1 = [xn(1), yn(1)];
    p2 = [xn(end), yn(end)];
    d = zeros(n, 1);
    for i = 1:n
        p = [xn(i), yn(i)];
        d(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + p2(1)*p1(2) - p2(2)*p1(1)) / ...
               sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    d(1) = -Inf;
    d(end) = -Inf;
    [~, idx] = max(d);
    if isempty(idx) || ~isfinite(d(idx))
        [~, idx] = min(y);
    end
end

function s = local_elbow_scores(wss, K)
    wss = wss(:);
    K = K(:);
    n = numel(wss);
    s = zeros(n, 1);
    if n <= 2
        s(:) = 0;
        s(elbow_index(wss)) = 1;
        return;
    end
    yn = normalise01(wss);
    xn = normalise01(K);
    p1 = [xn(1), yn(1)];
    p2 = [xn(end), yn(end)];
    for i = 1:n
        p = [xn(i), yn(i)];
        s(i) = abs((p2(2)-p1(2))*p(1) - (p2(1)-p1(1))*p(2) + p2(1)*p1(2) - p2(2)*p1(1)) / ...
               sqrt((p2(2)-p1(2))^2 + (p2(1)-p1(1))^2 + eps);
    end
    s(1) = 0;
    s(end) = 0;
    s = normalise01(s);
end

function y = normalise01(x)
    x = x(:);
    finite = isfinite(x);
    y = zeros(size(x));
    if ~any(finite)
        return;
    end
    xmin = min(x(finite));
    xmax = max(x(finite));
    y(finite) = (x(finite) - xmin) ./ (xmax - xmin + eps);
end

% =======================================================================
% SPM feature helpers
% =======================================================================

function ok = is_valid_spm_mix_result(result)
    ok = isstruct(result) && isfield(result, 'fm') && isfinite(result.fm) && ...
         isfield(result, 'state') && ~isempty(result.state) && isfield(result, 'm');
end

function labels = assign_spm_samples(X, vbmix)
    [N, D] = size(X);
    K = vbmix.m;
    log_prob = zeros(N, K);
    for k = 1:K
        m = vbmix.state(k).m(:)';
        C = vbmix.state(k).C;
        if numel(m) ~= D
            m = m(1:min(D, numel(m)));
            if numel(m) < D
                m = [m, zeros(1, D - numel(m))]; %#ok<AGROW>
            end
        end
        if size(C, 1) ~= D || size(C, 2) ~= D
            C2 = eye(D);
            rr = min(D, size(C, 1));
            cc = min(D, size(C, 2));
            C2(1:rr, 1:cc) = C(1:rr, 1:cc);
            C = C2;
        end
        if isfield(vbmix.state(k), 'prior')
            prior = vbmix.state(k).prior;
        elseif isfield(vbmix, 'priors') && numel(vbmix.priors) >= k
            prior = vbmix.priors(k);
        else
            prior = 1 / K;
        end
        log_prob(:, k) = log(prior + eps) + log_mvnpdf_local(X, m, C);
    end
    [~, labels] = max(log_prob, [], 2);
end

function log_p = log_mvnpdf_local(X, mu, Sigma)
    [N, D] = size(X);
    mu = mu(:)';
    Sigma = double(Sigma);
    Sigma = (Sigma + Sigma') / 2;
    jitter = 1e-6;
    for attempt = 1:6
        [R, flag] = chol(Sigma + jitter * eye(D));
        if flag == 0
            break;
        end
        jitter = jitter * 10;
    end
    if flag ~= 0
        Sigma = diag(max(diag(Sigma), jitter));
        R = chol(Sigma + jitter * eye(D));
    end
    X0 = X - mu;
    Q = sum((X0 / R) .^ 2, 2);
    c = D * log(2*pi) + 2 * sum(log(diag(R)));
    log_p = -0.5 * (Q + c);
    if numel(log_p) ~= N
        log_p = log_p(:);
    end
end

function [features, info] = pca_projective_features(maps_norm)
    maps_norm = normalize_maps_local(maps_norm);
    [N, D] = size(maps_norm);
    mu = mean(maps_norm, 1);
    Xc = maps_norm - mu;
    [coeff, score, latent] = pca(Xc, 'Centered', false);
    latent = real(latent(:));
    total_var = sum(latent);
    if total_var <= eps
        n_dims = min([D, max(1, N - 1), 8]);
        coeff = eye(D, n_dims);
        score = Xc * coeff;
        var_pct = 100;
        rank_est = n_dims;
    else
        tol = max(size(maps_norm)) * eps(max(latent));
        rank_est = sum(latent > tol);
        var_explained = cumsum(latent) / total_var;
        n_dims = find(var_explained >= 0.999, 1, 'first');
        if isempty(n_dims), n_dims = rank_est; end
        n_dims = min([n_dims, rank_est, N - 1, D - 1, 8]);
        n_dims = max(1, n_dims);
        coeff = coeff(:, 1:n_dims);
        score = score(:, 1:n_dims);
        var_pct = 100 * var_explained(n_dims);
    end
    scale = std(score, 0, 1);
    scale(scale < eps) = 1;
    score = score ./ scale;
    features = polarity_invariant_embedding(score);
    info = struct('mean_map', mu, 'coeff', coeff, 'scale', scale, ...
        'n_dims', n_dims, 'rank_est', rank_est, 'variance_explained_pct', var_pct);
end

function features = project_maps_to_projective_features(maps, info)
    maps = normalize_maps_local(maps);
    score = (maps - info.mean_map) * info.coeff;
    score = score ./ info.scale;
    features = polarity_invariant_embedding(score);
end

function Z = polarity_invariant_embedding(Y)
    Y = normalize_rows_local(Y);
    D = size(Y, 2);
    n_features = D * (D + 1) / 2;
    Z = zeros(size(Y, 1), n_features);
    col = 0;
    for a = 1:D
        for b = a:D
            col = col + 1;
            if a == b
                Z(:, col) = Y(:, a) .* Y(:, b);
            else
                Z(:, col) = sqrt(2) * Y(:, a) .* Y(:, b);
            end
        end
    end
end

function Xn = normalize_rows_local(X)
    norms = sqrt(sum(X.^2, 2));
    norms(norms < eps) = 1;
    Xn = X ./ norms;
end

function maps_norm = normalize_maps_local(maps)
    if isempty(maps)
        maps_norm = maps;
        return;
    end
    maps_norm = double(maps) - mean(double(maps), 2);
    norms = sqrt(sum(maps_norm.^2, 2));
    norms(norms < eps) = 1;
    maps_norm = maps_norm ./ norms;
end

function idx = deterministic_subsample(N, maxN)
    maxN = min(N, round(maxN));
    if N <= maxN
        idx = (1:N)';
    else
        idx = unique(round(linspace(1, N, maxN)))';
        if numel(idx) < maxN
            missing = setdiff((1:N)', idx, 'stable');
            idx = [idx; missing(1:(maxN - numel(idx)))];
            idx = sort(idx);
        end
    end
end

function txt = ternary_str(tf)
    if tf
        txt = 'ON';
    else
        txt = 'OFF';
    end
end

function S = rmfield_if_present(S, fields)
    for i = 1:numel(fields)
        if isfield(S, fields{i})
            S = rmfield(S, fields{i});
        end
    end
end

function node = empty_node_struct()
    node = struct('name', '', 'level', '', 'group', '', 'condition', '', 'participant', '', ...
        'file_path', '', 'file_indices', [], 'n_maps', NaN, 'K_estimated', NaN, ...
        'prior_weight', NaN, 'centers', [], 'labels', [], 'metrics', struct(), ...
        'fit', struct(), 'inherited', false, 'warning', '');
end

% =======================================================================
% Summaries and saving
% =======================================================================

function T = make_node_summary(global_fit, groups, condition_nodes, participant_nodes, file_nodes, K)
    nodes = [node_from_global(global_fit), groups(:)', condition_nodes(:)', participant_nodes(:)', file_nodes(:)'];
    n = numel(nodes);
    level = cell(n, 1); name = cell(n, 1); participant = cell(n, 1); group = cell(n, 1); condition = cell(n, 1);
    n_maps = nan(n, 1); inherited = false(n, 1); prior_weight = nan(n, 1);
    free_energy = nan(n, 1); silhouette = nan(n, 1); gev = nan(n, 1); within_ss = nan(n, 1); mean_abs_corr = nan(n, 1);
    warning_msg = cell(n, 1);
    for i = 1:n
        level{i} = nodes(i).level;
        name{i} = nodes(i).name;
        participant{i} = nodes(i).participant;
        group{i} = nodes(i).group;
        condition{i} = nodes(i).condition;
        n_maps(i) = nodes(i).n_maps;
        inherited(i) = nodes(i).inherited;
        prior_weight(i) = nodes(i).prior_weight;
        warning_msg{i} = nodes(i).warning;
        if isfield(nodes(i), 'metrics') && ~isempty(nodes(i).metrics)
            m = nodes(i).metrics;
            if isfield(m, 'free_energy'), free_energy(i) = m.free_energy; end
            if isfield(m, 'silhouette'), silhouette(i) = m.silhouette; end
            if isfield(m, 'gev'), gev(i) = m.gev; end
            if isfield(m, 'within_ss'), within_ss(i) = m.within_ss; end
            if isfield(m, 'mean_abs_corr'), mean_abs_corr(i) = m.mean_abs_corr; end
        end
    end
    K_selected = repmat(K, n, 1);
    T = table(level, name, participant, group, condition, K_selected, n_maps, inherited, prior_weight, ...
        free_energy, silhouette, gev, within_ss, mean_abs_corr, warning_msg);
end

function node = node_from_global(global_fit)
    node = empty_node_struct();
    node.name = 'global';
    node.level = 'global';
    node.n_maps = global_fit.n_maps;
    node.K_estimated = global_fit.K_estimated;
    node.prior_weight = 0;
    node.centers = global_fit.centers;
    node.labels = global_fit.labels;
    node.metrics = global_fit.metrics;
end

function T = make_file_summary(manifest, records, file_nodes, K)
    n = height(manifest);
    file_path = manifest.file_path;
    participant = manifest.participant;
    group = manifest.group;
    condition = manifest.condition;
    duration_s = nan(n, 1); n_samples = nan(n, 1); n_maps = nan(n, 1); n_fit_maps = nan(n, 1);
    gev = nan(n, 1); silhouette = nan(n, 1); mean_abs_corr = nan(n, 1);
    state_props = nan(n, K);
    for i = 1:n
        duration_s(i) = records(i).duration_s;
        n_samples(i) = records(i).n_samples;
        n_maps(i) = records(i).n_maps;
        n_fit_maps(i) = records(i).n_fit_maps;
        if isfield(file_nodes(i), 'metrics') && ~isempty(file_nodes(i).metrics)
            m = file_nodes(i).metrics;
            if isfield(m, 'gev'), gev(i) = m.gev; end
            if isfield(m, 'silhouette'), silhouette(i) = m.silhouette; end
            if isfield(m, 'mean_abs_corr'), mean_abs_corr(i) = m.mean_abs_corr; end
            if isfield(m, 'state_proportions')
                state_props(i, 1:min(K, numel(m.state_proportions))) = m.state_proportions(1:min(K, numel(m.state_proportions)));
            end
        end
    end
    T = table(file_path, participant, group, condition, duration_s, n_samples, n_maps, n_fit_maps, gev, silhouette, mean_abs_corr);
    for k = 1:K
        T.(sprintf('state_%02d_prop', k)) = state_props(:, k);
    end
end

function T = make_global_model_comparison(global_fit)
    K = global_fit.K_candidates(:);
    free_energy = global_fit.free_energy_vals(:);
    silhouette = global_fit.silhouette_vals(:);
    gev = global_fit.gev_vals(:);
    within_ss = global_fit.within_ss(:);
    selected = K == global_fit.K_estimated;
    T = table(K, free_energy, silhouette, gev, within_ss, selected);
end

function save_template_matrices(H, output_dir)
    template_dir = fullfile(output_dir, 'templates');
    if ~exist(template_dir, 'dir')
        mkdir(template_dir);
    end
    global_templates = H.global.centers; %#ok<NASGU>
    save(fullfile(template_dir, 'global_templates.mat'), 'global_templates');
    write_matrix_csv(fullfile(template_dir, 'global_templates.csv'), H.global.centers);

    for i = 1:numel(H.groups)
        safe = safe_file_label(H.groups(i).name);
        centers = H.groups(i).centers; %#ok<NASGU>
        save(fullfile(template_dir, sprintf('%s_templates.mat', safe)), 'centers');
        write_matrix_csv(fullfile(template_dir, sprintf('%s_templates.csv', safe)), H.groups(i).centers);
    end
    for i = 1:numel(H.group_conditions)
        safe = safe_file_label(H.group_conditions(i).name);
        centers = H.group_conditions(i).centers; %#ok<NASGU>
        save(fullfile(template_dir, sprintf('%s_templates.mat', safe)), 'centers');
        write_matrix_csv(fullfile(template_dir, sprintf('%s_templates.csv', safe)), H.group_conditions(i).centers);
    end
end

function safe = safe_file_label(s)
    safe = regexprep(char(s), '[^A-Za-z0-9_\-]+', '_');
    safe = regexprep(safe, '_+', '_');
end

function write_matrix_csv(file_path, X)
    if isempty(X)
        return;
    end
    writematrix(X, file_path);
end
