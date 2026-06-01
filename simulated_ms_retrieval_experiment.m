function T = simulated_ms_retrieval_experiment(varargin)
% BAYESIAN_MS_COMPARISON_PIPELINE: ALL method-criterion combinations
%
% Each method is tested with EVERY criterion (where applicable)
%
% Requirements:
% spm on path (development version at spm/spm on github)
% matlab (version R2025a used for this - other versions not tested)
% Dr Rohan Kandasamy 8-11-2025
% 
% Experimental data run on 25-11-2025
%
% Using code from Thomas Koenig (MICROSTATELAB group)
% (the "microstates" repository at: ThomasKoenigBern/microstates on github)

    util = microstate_utilities();
    repo_cfg = util.load_config();
    sim_defaults = repo_cfg.simulation;
    path_defaults = repo_cfg.paths;

    if exist(path_defaults.koenig_code_dir, "dir")
        addpath(path_defaults.koenig_code_dir);
    end
    % This is a folder which should contain:
    %  - eeg_kMeans.m
    %  - L2NormDim.m
    %  - mywaitbar.m
    %  - popFitMSMaps.m
    % All from the "microstates" repository

    test = false;

    p = inputParser;
    if test
        addParameter(p, 'out_dir', char(sim_defaults.out_dir), @ischar);
        addParameter(p, 'reps', 1, @isnumeric);
        addParameter(p, 'rep_vals', [], @isnumeric);
        addParameter(p, 'K_true_vals', [4], @isnumeric);
        addParameter(p, 'SNR_dbs', [10], @isnumeric);
        addParameter(p, 'K_candidates', 4:5, @isnumeric);
    else
        addParameter(p, 'out_dir', char(sim_defaults.out_dir), @ischar);
        addParameter(p, 'reps', double(sim_defaults.reps), @isnumeric);
        addParameter(p, 'rep_vals', [], @isnumeric);
        addParameter(p, 'K_true_vals', double(sim_defaults.K_true_vals(:)'), @isnumeric);
        addParameter(p, 'SNR_dbs', double(sim_defaults.SNR_dbs(:)'), @isnumeric);
        addParameter(p, 'K_candidates', double(sim_defaults.K_candidates(:)'), @isnumeric);
    end;
    addParameter(p, 'duration_s', double(sim_defaults.duration_s), @isnumeric);
    addParameter(p, 'sfreq', double(sim_defaults.sfreq), @isnumeric);
    addParameter(p, 'set_file', char(path_defaults.template_file), @ischar);
    addParameter(p, 'n_workers', double(sim_defaults.n_workers), @isnumeric);
    addParameter(p, 'cleanup', true, @islogical);
    addParameter(p, 'spm_path', '', @ischar); % optional: point to SPM toolbox/mixture if not obvious
    addParameter(p, 'verbose', true, @islogical);
    addParameter(p, 'save_json', true, @islogical);
    addParameter(p, 'montages', cellstr(string(sim_defaults.montages)), @iscell);  % montage robustness analysis
    addParameter(p, 'overlap_probs', double(sim_defaults.overlap_probs(:)'), @isnumeric); % run with and without overlap
    addParameter(p, 'overlap_ms_range', double(sim_defaults.overlap_ms_range(:)'), @isnumeric);
    addParameter(p, 'overlap_strength', double(sim_defaults.overlap_strength), @isnumeric);
    addParameter(p, 'compute_backfit_diagnostics', logical(util.get_field(sim_defaults, 'compute_backfit_diagnostics', true)), @islogical);
    addParameter(p, 'save_backfit_details', logical(util.get_field(sim_defaults, 'save_backfit_details', true)), @islogical);
    addParameter(p, 'template_alignment_strong_threshold', double(util.get_field(sim_defaults, 'template_alignment_strong_threshold', 0.5)), @isnumeric);
    addParameter(p, 'validate_simulation', logical(sim_defaults.validate_simulation), @islogical);
    addParameter(p, 'preprocessing', sim_defaults.preprocessing, @isstruct);
    parse(p, varargin{:});
    
    CONFIG = p.Results;
    CONFIG.out_dir = util.resolve_path(CONFIG.out_dir, util.project_root());
    CONFIG.set_file = util.resolve_path(CONFIG.set_file, util.project_root());
    if isempty(CONFIG.rep_vals)
        rep_vals = 1:CONFIG.reps;
    else
        rep_vals = CONFIG.rep_vals(:)';
        CONFIG.reps = numel(rep_vals);
    end

    % Load channel information
    if CONFIG.verbose
        fprintf('Loading channel information from: %s\n', CONFIG.set_file);
    end
    [ch_labels, montage_pos] = load_eeg_montage(CONFIG.set_file);
    n_channels = length(ch_labels);
    if CONFIG.verbose
        fprintf('✓ Loaded %d channels\n', n_channels);
    end
    % Setup directories
    if ~exist(CONFIG.out_dir, 'dir'), mkdir(CONFIG.out_dir); end
    res_dir = fullfile(CONFIG.out_dir, 'results');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end
    json_dir = fullfile(CONFIG.out_dir, 'microstates_json');
    if CONFIG.save_json && ~exist(json_dir, 'dir'), mkdir(json_dir); end
    backfit_dir = fullfile(res_dir, 'backfit_diagnostics');
    if CONFIG.compute_backfit_diagnostics && CONFIG.save_backfit_details && ~exist(backfit_dir, 'dir'), mkdir(backfit_dir); end
    confusion_dir = fullfile(res_dir, 'backfit_confusions');
    plots_dir = fullfile(CONFIG.out_dir, 'plots');
    if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('Microstate Comparison Pipeline\n');
        fprintf('ALL METHOD × CRITERION COMBINATIONS\n');
        fprintf('========================================\n');
        fprintf('Output: %s\n', CONFIG.out_dir);
        fprintf('Reps: %s | K true: %s | SNR: %s\n', ...
            mat2str(rep_vals), mat2str(CONFIG.K_true_vals), mat2str(CONFIG.SNR_dbs));
        fprintf('Channels: %d | Workers: %d\n', n_channels, CONFIG.n_workers);
        fprintf('Montages: %s\n', strjoin(CONFIG.montages, ', '));
        fprintf('Overlap probs: %s (ms range %s, strength %.2f)\n\n', ...
            mat2str(CONFIG.overlap_probs), mat2str(CONFIG.overlap_ms_range), CONFIG.overlap_strength);
    end

    % ===== ALL CRITERIA (universal) =====
    all_criteria = {'silhouette', 'free_energy', 'elbow', 'elbow_sil_combined', 'gev'};
    
    % First-line validation is focused on the VB method.
    method_names = {'spm_vb', 'kmeans_koenig'};
    
    % Guard: if SPM methods requested but SPM not available, stop early with a clear message
    needs_spm = any(contains(method_names, 'spm'));
    [spm_ok, spm_info] = util.ensure_spm_mix(CONFIG.spm_path, path_defaults.spm_mixture_paths, CONFIG.verbose);
    if CONFIG.verbose
        fprintf('SPM root detected: %s\n', local_which_text('spm'));
        fprintf('SPM mixture detected: %s\n', local_which_text('spm_mix'));
    end
    if needs_spm && ~spm_ok
        error(['SPM mixture toolbox not found on MATLAB path. ', ...
               'Checked explicit spm_path, SPM_PATH/SPM_MIXTURE_PATH, configured paths, and the folder inferred from spm.m. ', ...
               'spm.m=''%s''; spm_mix=''%s''; attempted={%s}.'], ...
               local_which_text('spm'), local_which_text('spm_mix'), strjoin(spm_info.attempted, ', '));
    end
    
    % Montages to test
    n_montages = length(CONFIG.montages);
    overlap_probs = CONFIG.overlap_probs;
    n_overlap_conditions = length(overlap_probs);
    
    % Build list of EEG conditions
    eeg_conditions = [];
    for rep = rep_vals
        for K_true = CONFIG.K_true_vals
            for SNR_dB = CONFIG.SNR_dbs
                for ov = 1:n_overlap_conditions
                    overlap_prob = overlap_probs(ov);
                    eeg_conditions = [eeg_conditions; rep, K_true, SNR_dB, overlap_prob]; %#ok<AGROW>
                end
            end
        end
    end
    
    n_eeg_conditions = size(eeg_conditions, 1);
    n_methods = length(method_names);
    n_criteria = length(all_criteria);
    n_fits = n_eeg_conditions * n_montages * n_methods;
    n_total_results = n_fits * n_criteria;
    
    if CONFIG.verbose
        fprintf('PIPELINE STRUCTURE:\n');
        fprintf('  EEG conditions: %d (includes %d overlap settings)\n', n_eeg_conditions, n_overlap_conditions);
        fprintf('  Montages: %d\n', n_montages);
        fprintf('  Methods: %d\n', n_methods);
        fprintf('  Criteria (applied to all): %d\n', n_criteria);
        fprintf('  Total FITS: %d (one per montage+method+EEG)\n', n_fits);
        fprintf('  Total RESULTS: %d (fits × criteria)\n\n', n_total_results);
    end

    % ===== UNIFIED PIPELINE: GENERATE → FIT → CRITERIA → SAVE (per EEG) =====
    if CONFIG.verbose
        fprintf('Pipeline: Generate EEG → Montage → Fit Methods → Apply Criteria → Save\n');
        fprintf('Processing %d EEG conditions with %d montages, %d methods and %d criteria each...\n', ...
            n_eeg_conditions, n_montages, n_methods, n_criteria);
    end
    
    rows = [];
    run_id = 0;
    n_failed = 0;
    n_successful_fits = 0;
    error_types = containers.Map();
    fit_id = 0;
    
    % Initialize parallel pool if needed (will be used within EEG loop for method fitting)
    try
        p_pool = gcp('nocreate');
        if isempty(p_pool)
            % Check available cores and set pool size
            max_workers = maxNumCompThreads();
            n_workers = min(CONFIG.n_workers, max_workers);
            fprintf('Starting parallel pool with %d workers (max available: %d)\n', n_workers, max_workers);
            parpool('local', n_workers);
        else
            fprintf('Using existing parallel pool with %d workers\n', p_pool.NumWorkers);
        end
    catch ME
        fprintf('Warning: Could not start parallel pool: %s\n', ME.message);
        fprintf('Falling back to serial execution\n');
    end
    
    % Prepare for parallel execution
    all_results = cell(n_eeg_conditions, 1);
    
    % We cannot use the progress bar object inside parfor easily
    fprintf('Starting parallel processing of %d EEG conditions...\n', n_eeg_conditions);
    
    parfor eeg_idx = 1:n_eeg_conditions
        % Local variables for this iteration
        local_rows = [];
        
        rep = eeg_conditions(eeg_idx, 1);
        K_true = eeg_conditions(eeg_idx, 2);
        SNR_dB = eeg_conditions(eeg_idx, 3);
        overlap_prob = eeg_conditions(eeg_idx, 4);
        
        % ===== STEP 1: GENERATE one EEG (full montage, 71 channels) =====
        try
            sim_seed = 42 + rep*1000 + K_true*100 + round((SNR_dB+10)*10);
            [Sim, ~, ~] = generate_microstate_eeg(K_true, SNR_dB, CONFIG.duration_s, ...
                CONFIG.sfreq, sim_seed, struct(...
                    'prob', overlap_prob, ...
                    'ms_range', CONFIG.overlap_ms_range, ...
                    'strength', CONFIG.overlap_strength));
            if CONFIG.validate_simulation && exist('validate_simulated_eeg_representativeness', 'file') == 2
                Sim.sim_qc = validate_simulated_eeg_representativeness(Sim, 'verbose', false);
            elseif CONFIG.validate_simulation && CONFIG.verbose
                fprintf('\nSimulation QC requested but validate_simulated_eeg_representativeness.m is not on the path; skipping QC.\n');
            end
        catch ME
            if CONFIG.verbose
                fprintf('\n✗ EEG Generation Error (Rep %d, K=%d, SNR=%+.1f): %s\n', ...
                    rep, K_true, SNR_dB, ME.message);
            end
            continue;
        end
        
        % Store full Sim before montage reduction
        Sim_full = Sim;
        
        % ===== STEP 2: LOOP over MONTAGES =====
        for montage_idx = 1:n_montages
            montage_type = CONFIG.montages{montage_idx};
            
            % Apply montage selection
            if strcmp(montage_type, 'full')
                % Use full montage
                Sim = Sim_full;
                Sim.n_channels = size(Sim_full.X_noisy, 1);
            else
                % Reduce to specific montage
                [EEG_reduced, pos_reduced, chanlocs_reduced, labels_reduced, ~] = ...
                    select_montage_subset(Sim_full.X_noisy, Sim_full.pos, ...
                    Sim_full.chanlocs, ch_labels, montage_type);
                
                % Also reduce clean signal and true maps
                [X_clean_reduced, ~, ~, ~, ~] = ...
                    select_montage_subset(Sim_full.X_clean, Sim_full.pos, ...
                    Sim_full.chanlocs, ch_labels, montage_type);
                
                [maps_true_reduced, ~, ~, ~, ~] = ...
                    select_montage_subset(Sim_full.maps_true, Sim_full.pos, ...
                    Sim_full.chanlocs, ch_labels, montage_type);
                
                % Create reduced Sim structure
                Sim = Sim_full;
                Sim.X_noisy = EEG_reduced;
                Sim.X_clean = X_clean_reduced;
                Sim.pos = pos_reduced;
                Sim.chanlocs = chanlocs_reduced;
                Sim.channel_labels = labels_reduced;
                Sim.maps_true = maps_true_reduced;
                Sim.n_channels = length(labels_reduced);
            end
            
            % Add montage info to Sim
            Sim.montage_type = montage_type;
            Sim.n_leads = Sim.n_channels;
            Sim.preprocessing = CONFIG.preprocessing;
            if CONFIG.compute_backfit_diagnostics
                try
                    Sim.true_template_alignment = align_microstates_to_template(Sim.maps_true, CONFIG.set_file, ...
                        'estimated_channel_labels', Sim.channel_labels, ...
                        'strong_threshold', CONFIG.template_alignment_strong_threshold);
                catch
                    Sim.true_template_alignment = [];
                end
            end
            
            % ===== STEP 3: FIT all methods to this montage =====
            fit_results_for_montage = cell(n_methods, 1);
            
            for m_idx = 1:n_methods
                method_str = method_names{m_idx};
                
                try
                    % Call appropriate fitting function with a default criterion
                    % (Results will be reused with all criteria via select_K_by_criterion)
                    if strcmp(method_str, 'spm_vb')
                        Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, 'elbow_sil_combined');
                    elseif strcmp(method_str, 'kmeans_koenig')
                        Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_candidates, 'silhouette');
                    elseif strcmp(method_str, 'spm_kmeans')
                        Results = fit_microstate_spm_kmeans(Sim, CONFIG.K_candidates, 'silhouette');
                    elseif strcmp(method_str, 'vb_kmeans')
                        Results = fit_microstate_vb_kmeans(Sim, CONFIG.K_candidates, 'free_energy');
                    elseif strcmp(method_str, 'dp_mixture')
                        Results = fit_microstate_dp_mixture(Sim, CONFIG.K_candidates, 'free_energy');
                    else
                        Results = [];
                    end
                    
                    if ~isempty(Results) && isfield(Results, 'valid_fit') && Results.valid_fit
                        nan_centers = isfield(Results, 'centers') && any(isnan(Results.centers(:)));
                        nan_scores = (isfield(Results, 'free_energy_vals') && any(isnan(Results.free_energy_vals(:)))) || ...
                                     (isfield(Results, 'silhouette_vals') && any(isnan(Results.silhouette_vals(:))));
                        if nan_centers || nan_scores
                            fprintf('Fit Warning (%s, montage %s): NaNs detected (centers=%d, scores=%d)\n', ...
                                method_str, montage_type, nan_centers, nan_scores);
                        end
                        % Store ONLY essential metadata, NOT full Results or Sim
                        % (Sim will be accessible in the save step, Results is available above)
                        fit_results_for_montage{m_idx} = struct(...
                            'eeg_idx', eeg_idx, ...
                            'rep', rep, ...
                            'K_true', K_true, ...
                            'SNR_dB', SNR_dB, ...
                            'method', method_str, ...
                            'montage_type', montage_type, ...
                            'n_leads', Sim.n_leads, ...
                            'Results', Results);
                    else
                        fit_results_for_montage{m_idx} = [];
                        fprintf('Fit Warning (%s, montage %s): invalid or failed fit (valid_fit missing/false)\n', method_str, montage_type);
                    end
                    
                catch ME
                    fit_results_for_montage{m_idx} = [];
                    fprintf('Fit Error (%s, montage %s): %s\n', method_str, montage_type, ME.message);
                end
            end
            
            % ===== STEP 4 & 5: APPLY CRITERIA and SAVE (for each fit) =====
            for m_idx = 1:n_methods
                if isempty(fit_results_for_montage{m_idx})
                    continue;
                end
                
                fit_result = fit_results_for_montage{m_idx};
                Results = fit_result.Results;
                method_str = char(string(fit_result.method));
                selected_solution_cache = struct();
                
                % Apply ALL criteria to this ONE fit
                for c_idx = 1:n_criteria
                    criterion_str = char(string(all_criteria{c_idx}));
                    
                    % Extract K using this criterion
                    K_selected = select_K_by_criterion(Results, criterion_str);
                    
                    if isnan(K_selected)
                        continue;
                    end
                    
                    cache_key = sprintf('K_%d', K_selected);
                    if isfield(selected_solution_cache, cache_key)
                        SelectedResults = selected_solution_cache.(cache_key);
                    else
                        SelectedResults = extract_microstate_solution_for_k(Results, K_selected, ...
                            'Sim', Sim, ...
                            'criterion', criterion_str, ...
                            'template_file', CONFIG.set_file, ...
                            'estimated_channel_labels', Sim.channel_labels, ...
                            'strong_threshold', CONFIG.template_alignment_strong_threshold);
                        if CONFIG.compute_backfit_diagnostics
                            try
                                BackfitDiagnostics = compute_simulation_backfit_diagnostics(Sim, SelectedResults, CONFIG.set_file, ...
                                    'strong_threshold', CONFIG.template_alignment_strong_threshold);
                                SelectedResults.backfit_diagnostics = BackfitDiagnostics;
                            catch
                            end
                        end
                        selected_solution_cache.(cache_key) = SelectedResults;
                    end
                    SelectedResults.criterion = criterion_str;
                    SelectedResults.best_criterion_value = criterion_specific_best_score(Results, K_selected, criterion_str);
                    rec_metrics = SelectedResults.recovery_metrics;
                    recovery_corr = util.padded_vector(rec_metrics.match_similarities, 10);
                    json_file = '';
                    backfit_diagnostic_file = '';
                    backfit_coverage_corr = NaN;
                    backfit_coverage_spearman = NaN;
                    backfit_coverage_mae = NaN;
                    backfit_coverage_rmse = NaN;
                    backfit_coverage_l1 = NaN;
                    selected_cov_trace_mean = NaN;
                    selected_cov_trace_median = NaN;
                    selected_cov_logdet_mean = NaN;
                    selected_spm_feature_dim = NaN;
                    if isfield(SelectedResults, 'selected_spm_mix_model') && ~isempty(SelectedResults.selected_spm_mix_model)
                        spm_model = SelectedResults.selected_spm_mix_model;
                        if isfield(spm_model, 'covariance_traces') && ~isempty(spm_model.covariance_traces)
                            selected_cov_trace_mean = mean(spm_model.covariance_traces, 'omitnan');
                            selected_cov_trace_median = median(spm_model.covariance_traces, 'omitnan');
                        end
                        if isfield(spm_model, 'covariance_logdets') && ~isempty(spm_model.covariance_logdets)
                            selected_cov_logdet_mean = mean(spm_model.covariance_logdets, 'omitnan');
                        end
                        if isfield(spm_model, 'feature_dim') && ~isempty(spm_model.feature_dim)
                            selected_spm_feature_dim = spm_model.feature_dim;
                        end
                    end
                    
                    % Generate a temporary subject name (IDs will be fixed later)
                    montage_name = char(string(montage_type));
                    overlap_tag = sprintf('ovl%02d', round(100 * Sim.overlap_prob));
                    subj_name = sprintf('fit_E%d_M%d_K%d_SNR%+d_%s_%s_%s_%s', ...
                        eeg_idx, m_idx, fit_result.K_true, round(fit_result.SNR_dB), ...
                        montage_name, overlap_tag, method_str, criterion_str);
                    if CONFIG.compute_backfit_diagnostics && isfield(SelectedResults, 'backfit_diagnostics')
                        BackfitDiagnostics = SelectedResults.backfit_diagnostics;
                        if isfield(BackfitDiagnostics, 'ok') && BackfitDiagnostics.ok
                            backfit_coverage_corr = BackfitDiagnostics.coverage_corr;
                            backfit_coverage_spearman = BackfitDiagnostics.coverage_spearman;
                            backfit_coverage_mae = BackfitDiagnostics.coverage_mae;
                            backfit_coverage_rmse = BackfitDiagnostics.coverage_rmse;
                            backfit_coverage_l1 = BackfitDiagnostics.coverage_l1;
                            if CONFIG.save_backfit_details
                                backfit_diagnostic_file = fullfile(backfit_dir, [subj_name '_backfit.mat']);
                                backfit_payload = struct('BackfitDiagnostics', BackfitDiagnostics, ...
                                    'method_str', method_str, ...
                                    'criterion_str', criterion_str, ...
                                    'K_selected', K_selected, ...
                                    'subj_name', subj_name);
                                save(backfit_diagnostic_file, '-fromstruct', backfit_payload, '-v7.3');
                            end
                        end
                    end
                    sim_qc_status = 'NOT_RUN';
                    sim_qc_psd_slope = NaN;
                    sim_qc_mean_dwell_ms = NaN;
                    sim_qc_gfp_median_uv = NaN;
                    if isfield(Sim, 'sim_qc') && ~isempty(Sim.sim_qc)
                        sim_qc_status = Sim.sim_qc.overall_status;
                        sim_qc_psd_slope = Sim.sim_qc.spectrum.power_slope_2_40hz;
                        sim_qc_gfp_median_uv = Sim.sim_qc.amplitude.gfp_median_uv;
                        if isfield(Sim.sim_qc.microstate_dynamics, 'mean_dwell_ms')
                            sim_qc_mean_dwell_ms = Sim.sim_qc.microstate_dynamics.mean_dwell_ms;
                        end
                    end
                    
                    result_row = struct( ...
                        'fit_id', 0, ... % Placeholder
                        'subject', subj_name, ...
                        'rep', fit_result.rep, ...
                        'method', method_str, ...
                        'criterion', criterion_str, ...
                        'K_true', fit_result.K_true, ...
                        'SNR_dB', fit_result.SNR_dB, ...
                        'overlap_prob', Sim.overlap_prob, ...
                        'overlap_strength', Sim.overlap_strength, ...
                        'overlap_ms_min', Sim.overlap_ms_range(1), ...
                        'overlap_ms_max', Sim.overlap_ms_range(end), ...
                        'montage_type', montage_name, ...
                        'n_leads', fit_result.n_leads, ...
                        'sim_qc_status', sim_qc_status, ...
                        'sim_qc_psd_slope_2_40hz', sim_qc_psd_slope, ...
                        'sim_qc_gfp_median_uv', sim_qc_gfp_median_uv, ...
                        'sim_qc_mean_dwell_ms', sim_qc_mean_dwell_ms, ...
                        'K_estimated', K_selected, ...
                        'K_correct', fit_result.K_true == K_selected, ...
                        'K_error', abs(fit_result.K_true - K_selected), ...
                        'n_maps', Results.n_maps, ...
                        'n_matched', rec_metrics.n_matched, ...
                        'mean_recovery_matched', rec_metrics.mean_recovery_matched, ...
                        'mean_recovery_padded', rec_metrics.mean_recovery_padded, ...
                        'sensitivity', rec_metrics.sensitivity, ...
                        'precision', rec_metrics.precision, ...
                        'f1_score', rec_metrics.f1_score, ...
                        'recovery_01', recovery_corr(1), ...
                        'recovery_02', recovery_corr(2), ...
                        'recovery_03', recovery_corr(3), ...
                        'selected_spm_cov_trace_mean', selected_cov_trace_mean, ...
                        'selected_spm_cov_trace_median', selected_cov_trace_median, ...
                        'selected_spm_cov_logdet_mean', selected_cov_logdet_mean, ...
                        'selected_spm_feature_dim', selected_spm_feature_dim, ...
                        'backfit_coverage_corr', backfit_coverage_corr, ...
                        'backfit_coverage_spearman', backfit_coverage_spearman, ...
                        'backfit_coverage_mae', backfit_coverage_mae, ...
                        'backfit_coverage_rmse', backfit_coverage_rmse, ...
                        'backfit_coverage_l1', backfit_coverage_l1, ...
                        'backfit_diagnostic_file', backfit_diagnostic_file, ...
                        'best_score', SelectedResults.best_criterion_value, ...
                        'json_file', json_file, ...
                        'runtime_s', SelectedResults.runtime);
                    
                    local_rows = [local_rows; result_row]; 
                    
                    % Save JSON with metadata for plotting & downstream use
                    if CONFIG.save_json
                        json_file = fullfile(json_dir, [subj_name '.json']);
                        try
                            META = struct();
                            META.subject = subj_name;
                            META.method = method_str;
                            META.criterion = criterion_str;
                            META.K_true = fit_result.K_true;
                            META.K_estimated = K_selected;
                            META.SNR_dB = fit_result.SNR_dB;
                            META.rep = fit_result.rep;
                            META.montage_type = montage_name;
                            META.n_leads = fit_result.n_leads;
                            META.overlap_prob = Sim.overlap_prob;
                            META.overlap_ms_range = Sim.overlap_ms_range;
                            META.overlap_strength = Sim.overlap_strength;
                            META.n_overlap_events = Sim.n_overlap_events;
                            META.sim_qc_status = sim_qc_status;
                            META.sim_qc_psd_slope_2_40hz = sim_qc_psd_slope;
                            META.sim_qc_gfp_median_uv = sim_qc_gfp_median_uv;
                            META.sim_qc_mean_dwell_ms = sim_qc_mean_dwell_ms;
                            META.runtime_s = SelectedResults.runtime;
                            META.channel_labels = Sim.channel_labels;  % Use Sim.channel_labels (montage-specific)
                            save_microstate_json(SelectedResults, Sim, json_file, META);
                            local_rows(end).json_file = json_file;
                        catch ME
                            if CONFIG.verbose
                                fprintf('⚠ Warning: Could not save JSON for %s: %s\n', subj_name, ME.message);
                            end
                        end
                    end
                end
            end
        end  % End montage loop
        
        % Store results for this EEG condition
        all_results{eeg_idx} = local_rows;
        
    end  % End PARFOR loop
    
    % Combine all results
    rows = vertcat(all_results{:});
    
    % Fix IDs
    if ~isempty(rows)
        for i = 1:length(rows)
            rows(i).fit_id = i;
        end
    end
    
    if isempty(rows)
        error('✗✗✗ NO SUCCESSFUL RUNS! ✗✗✗\nCheck error summary above for details.');
    end
    
    % Convert to table
    T = struct2table(rows);
    total_expected_fits = n_eeg_conditions * n_montages * n_methods;
    successful_fit_count = count_unique_method_fits(T);
    failed_fit_count = total_expected_fits - successful_fit_count;

    fprintf('\n========================================\n');
    fprintf('FIT SUMMARY\n');
    fprintf('========================================\n');
    fprintf('Successful method fits: %d / %d\n', successful_fit_count, total_expected_fits);
    fprintf('Failed or invalid method fits: %d\n', failed_fit_count);
    fprintf('Criterion result rows: %d\n', height(T));
    fprintf('========================================\n\n');
    
    % Save CSV
    summary_csv = fullfile(res_dir, 'comparison_results.csv');
    writetable(T, summary_csv);
    criterion_summary_csv = fullfile(res_dir, 'k_selection_summary_by_method_criterion.csv');
    Tcrit = k_selection_summary_by_method_criterion(T);
    writetable(Tcrit, criterion_summary_csv);
    if CONFIG.compute_backfit_diagnostics
        if ~exist(confusion_dir, 'dir'), mkdir(confusion_dir); end
        write_simulation_backfit_reports(T, confusion_dir);
    end
    
    if CONFIG.verbose
        fprintf('✓ Saved: %s\n', summary_csv);
        fprintf('âœ“ Saved: %s\n', criterion_summary_csv);
        fprintf('  Successful: %d rows | Failed: %d\n', height(T), n_failed);
    end
    
    save_summary_info(CONFIG, ch_labels, montage_pos, res_dir);
    print_summary(T);
    
    fprintf('\n========================================\n');
    fprintf('Pipeline Complete!\n');
    fprintf('EEG Conditions: %d (overlap settings: %s)\n', n_eeg_conditions, mat2str(overlap_probs));
    fprintf('Montages: %d\n', n_montages);
    fprintf('Methods: %d\n', n_methods);
    fprintf('Criteria (universal): %d\n', n_criteria);
    fprintf('Total Fits: %d (EEG conditions × montages × methods)\n', n_eeg_conditions * n_montages * n_methods);
    fprintf('Total Results: %d (one per method×criterion per montage per EEG)\n', height(T));
    if CONFIG.save_json
        fprintf('JSON Microstates: %s\n', json_dir);
    else
        fprintf('JSON Microstates: disabled\n');
    end
    fprintf('Results CSV: %s\n', res_dir);
    fprintf('========================================\n');
end

% ======================== HELPERS ========================

function n = count_unique_method_fits(T)
    keys = strcat( ...
        string(T.rep), "|", ...
        string(T.K_true), "|", ...
        string(T.SNR_dB), "|", ...
        string(T.overlap_prob), "|", ...
        string(T.montage_type), "|", ...
        string(T.method));
    n = numel(unique(keys));
end

function Tcrit = k_selection_summary_by_method_criterion(T)
    Tcrit = groupsummary(T, {'method', 'criterion'}, {'mean', 'std'}, {'K_correct', 'K_error'});
    Tcrit.accuracy_pct = 100 * Tcrit.mean_K_correct;
end

function K_selected = select_K_by_criterion(Results, criterion)
    switch criterion
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && ~isempty(Results.silhouette_vals)
                sil = Results.silhouette_vals;
                if length(sil) > 4
                    [~, idx] = max(sil(2:(end-1)));
                    idx = idx + 1;
                else
                    [~, idx] = max(sil);
                end
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
            
        case 'gev'
            if isfield(Results, 'gev_vals') && ~isempty(Results.gev_vals)
                gev = Results.gev_vals;
                valid_idx = gev > 0 & isfinite(gev);
                if any(valid_idx)
                    [~, idx] = max(gev(valid_idx));
                    valid_k = find(valid_idx);
                    K_selected = Results.K_candidates(valid_k(idx));
                else
                    K_selected = NaN;
                end
            else
                K_selected = NaN;
            end
            
        case 'elbow'
            if isfield(Results, 'within_ss') && ~isempty(Results.within_ss)
                wss = Results.within_ss;
                valid_idx = isfinite(wss) & wss > 0;
                if any(valid_idx)
                    wss_v = wss(valid_idx);
                    k_cand_v = Results.K_candidates(valid_idx);
                    [K_selected, ~] = select_K_from_elbow_helper(wss_v, k_cand_v);
                else
                    K_selected = NaN;
                end
            else
                K_selected = NaN;
            end
            
        case 'free_energy'
            if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
                fe = Results.free_energy_vals;
                valid_idx = ~isinf(fe) & fe ~= 0;
                if any(valid_idx)
                    [~, idx] = max(fe(valid_idx));
                    valid_k = find(valid_idx);
                    K_selected = Results.K_candidates(valid_k(idx));
                else
                    K_selected = NaN;
                end
            else
                K_selected = NaN;
            end
            
        case {'elbow_sil_combined', 'elbow_only'}
            K_selected = Results.K_estimated;
            
        otherwise
            K_selected = NaN;
    end
end

function score = criterion_specific_best_score(Results, K_selected, criterion)
    score = NaN;
    idx = find(double(Results.K_candidates(:)) == double(K_selected), 1, 'first');
    if isempty(idx)
        if isfield(Results, 'best_criterion_value')
            score = Results.best_criterion_value;
        end
        return;
    end
    switch criterion
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && numel(Results.silhouette_vals) >= idx
                score = Results.silhouette_vals(idx);
            end
        case 'free_energy'
            if isfield(Results, 'free_energy_vals') && numel(Results.free_energy_vals) >= idx
                score = Results.free_energy_vals(idx);
            end
        case 'gev'
            if isfield(Results, 'gev_vals') && numel(Results.gev_vals) >= idx
                score = Results.gev_vals(idx);
            end
        case 'elbow'
            if isfield(Results, 'within_ss') && numel(Results.within_ss) >= idx
                score = Results.within_ss(idx);
            end
        otherwise
            if isfield(Results, 'best_criterion_value')
                score = Results.best_criterion_value;
            end
    end
end

function [K_est, score] = select_K_from_elbow_helper(wss_vals, K_candidates)
    n = length(wss_vals);
    if n < 4
        [~, idx] = min(wss_vals);
        K_est = K_candidates(idx);
        score = wss_vals(idx);
        return;
    end
    
    wss_norm = (wss_vals - min(wss_vals)) / (max(wss_vals) - min(wss_vals) + eps);
    
    curvature = zeros(n, 1);
    for i = 2:(n-1)
        dy1 = wss_norm(i) - wss_norm(i-1);
        dy2 = wss_norm(i+1) - wss_norm(i);
        curvature(i) = abs(dy2 - dy1);
    end
    
    [~, idx] = max(curvature(2:(n-1)));
    idx = idx + 1;
    
    K_est = K_candidates(idx);
    score = curvature(idx);
end

function save_summary_info(CONFIG, ch_labels, montage_pos, output_dir)
    info = struct();
    info.analysis_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    info.user = getenv('USERNAME');
    info.config = CONFIG;
    info.channel_labels = ch_labels;
    info.n_channels = length(ch_labels);
    info.montage_positions = montage_pos;
    if isfield(CONFIG, 'rep_vals') && ~isempty(CONFIG.rep_vals)
        n_rep_conditions = numel(CONFIG.rep_vals);
    else
        n_rep_conditions = CONFIG.reps;
    end
    info.n_eeg_conditions = n_rep_conditions * numel(CONFIG.K_true_vals) * numel(CONFIG.SNR_dbs) * numel(CONFIG.overlap_probs);
    info.overlap_probs = CONFIG.overlap_probs;
    
    try
        json_info = jsonencode(info);
        fid = fopen(fullfile(output_dir, 'analysis_info.json'), 'w');
        fprintf(fid, '%s', json_info);
        fclose(fid);
    catch
    end
end

function print_summary(T)
    fprintf('\n========================================\n');
    fprintf('COMPARISON SUMMARY\n');
    fprintf('========================================\n');
    
    util = microstate_utilities();
    methods = unique(T.method);
    if istable(methods), methods = table2cell(methods); end
    
    fprintf('\nPerformance by Method (All Criteria Combined):\n');
    fprintf('%-20s %10s %10s %10s %10s\n', 'Method', 'Accuracy', 'F1 Score', 'Matched', 'Runtime');
    fprintf('%s\n', repmat('-', 1, 65));
    
    for m_idx = 1:length(methods)
        m = methods{m_idx};
        mask = strcmp(T.method, m);
        
        acc = 100 * mean(T.K_correct(mask));
        f1 = mean(T.f1_score(mask), 'omitnan');
        matched = mean(T.n_matched(mask), 'omitnan');
        rt = mean(T.runtime_s(mask), 'omitnan');
        
        % Format method name for display
        m_display = util.format_method_name(m);
        fprintf('%-20s %9.1f%% %10.3f %10.1f %9.1fs\n', m_display, acc, f1, matched, rt);
    end

    criteria = unique(T.criterion);
    if istable(criteria), criteria = table2cell(criteria); end

    fprintf('\nK Selection Accuracy by Method and Criterion:\n');
    fprintf('%-20s %-22s %10s %10s %8s\n', 'Method', 'Criterion', 'Accuracy', 'Mean |err|', 'N');
    fprintf('%s\n', repmat('-', 1, 76));

    for m_idx = 1:length(methods)
        m = methods{m_idx};
        for c_idx = 1:length(criteria)
            c = criteria{c_idx};
            mask = strcmp(T.method, m) & strcmp(T.criterion, c);
            if ~any(mask)
                continue;
            end
            acc = 100 * mean(T.K_correct(mask));
            kerr = mean(T.K_error(mask), 'omitnan');
            m_display = util.format_method_name(m);
            fprintf('%-20s %-22s %9.1f%% %10.3f %8d\n', ...
                m_display, c, acc, kerr, nnz(mask));
        end
    end
    
    fprintf('\n');
end

function [ch_labels, montage_pos] = load_eeg_montage(set_file)
    if nargin > 0 && ~isempty(set_file) && ischar(set_file)
        if isfile(set_file)
            try
                if exist('pop_loadset', 'file')
                    EEG = pop_loadset(set_file);
                    if ~isempty(EEG.chanlocs)
                        ch_labels = {EEG.chanlocs.labels}';
                        
                        X = [EEG.chanlocs.X]';
                        Y = [EEG.chanlocs.Y]';
                        Z = [EEG.chanlocs.Z]';
                        
                        montage_pos = [X, Y, Z];
                        norms = sqrt(sum(montage_pos.^2, 2));
                        montage_pos = montage_pos ./ (norms + eps);
                        
                        fprintf('✓ Loaded %d channels from: %s\n', length(ch_labels), set_file);
                        return;
                    end
                end
                
                if endsWith(set_file, '.mat')
                    data = load(set_file);
                    if isfield(data, 'ch_labels') && isfield(data, 'montage_pos')
                        ch_labels = data.ch_labels;
                        montage_pos = data.montage_pos;
                        fprintf('✓ Loaded %d channels from: %s\n', length(ch_labels), set_file);
                        return;
                    end
                end
            catch ME
                fprintf('⚠ Could not load %s: %s\n', set_file, ME.message);
            end
        end
    end
    
    fprintf('Using standard 10-20 Extended montage (64 channels)\n');
    
    ch_labels = get_standard_10_20_labels();
    montage_pos = get_standard_10_20_positions();
end

function ch_labels = get_standard_10_20_labels()
    ch_labels = {
        'Cz', 'Fz', 'Pz', 'Oz', ...
        'Fp1', 'Fp2', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', ...
        'AF3', 'AF4', 'AF7', 'AF8', ...
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', ...
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', ...
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', ...
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', ...
        'T3', 'T4', 'T5', 'T6', ...
        'O1', 'O2', ...
        'Iz', 'POz', 'FCz', 'CPz', ...
        'PO3', 'PO4', 'PO7', 'PO8'
    };
end

function montage_pos = get_standard_10_20_positions()
    montage_pos = [
        0.000,  0.000,  1.000;
        0.000,  0.809,  0.588;
        0.000, -0.809,  0.588;
        0.000,  0.000, -1.000;
        
        0.000,  1.000,  0.000;
        0.000,  1.000,  0.000;
        0.156,  0.951,  0.268;
        -0.156,  0.951,  0.268;
        0.309,  0.904,  0.293;
        -0.309,  0.904,  0.293;
        0.454,  0.809,  0.372;
        -0.454,  0.809,  0.372;
        0.588,  0.676,  0.447;
        -0.588,  0.676,  0.447;
        0.125,  0.968,  0.224;
        -0.125,  0.968,  0.224;
        0.274,  0.942,  0.195;
        -0.274,  0.942,  0.195;
        
        0.156,  0.000,  0.988;
        -0.156,  0.000,  0.988;
        0.309,  0.000,  0.951;
        -0.309,  0.000,  0.951;
        0.454,  0.000,  0.891;
        -0.454,  0.000,  0.891;
        0.250,  0.484,  0.835;
        -0.250,  0.484,  0.835;
        0.407,  0.809,  0.424;
        -0.407,  0.809,  0.424;
        0.588,  0.676,  0.447;
        -0.588,  0.676,  0.447;
        0.809,  0.484,  0.327;
        -0.809,  0.484,  0.327;
        
        0.156, -0.484,  0.859;
        -0.156, -0.484,  0.859;
        0.309, -0.809,  0.495;
        -0.309, -0.809,  0.495;
        0.454, -0.809,  0.372;
        -0.454, -0.809,  0.372;
        0.588, -0.676,  0.447;
        -0.588, -0.676,  0.447;
        0.250, -0.484,  0.835;
        -0.250, -0.484,  0.835;
        0.407, -0.809,  0.424;
        -0.407, -0.809,  0.424;
        0.588, -0.676,  0.447;
        -0.588, -0.676,  0.447;
        0.809, -0.484,  0.327;
        -0.809, -0.484,  0.327;
        
        0.951,  0.309,  0.000;
        -0.951,  0.309,  0.000;
        0.951, -0.309,  0.000;
        -0.951, -0.309,  0.000;
        
        0.156, -0.951,  0.268;
        -0.156, -0.951,  0.268;
        0.000, -1.000,  0.000;
        0.000, -0.951,  0.309;
        0.000,  0.485,  0.875;
        0.000, -0.485,  0.875;
        0.125, -0.951,  0.283;
        -0.125, -0.951,  0.283;
        0.274, -0.942,  0.195;
        -0.274, -0.942,  0.195;
    ];
end

function txt = local_which_text(name)
    txt = which(name);
    if isempty(txt)
        txt = '<not found>';
    end
end
