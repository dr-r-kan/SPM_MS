function T = Bayesian_MS_Comparison_Pipeline(varargin)
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

    addpath("Koenig_code");
    % This is a folder which should contain:
    %  - eeg_kMeans.m
    %  - L2NormDim.m
    %  - mywaitbar.m
    %  - popFitMSMaps.m
    % All from the "microstates" repository

    % we also need to make sure the spm toolbox is to hand:
    addpath("/home/rohan/spm/toolbox/mixture/"); % modify to the correct path on your system if needed
    
    test = false;

    p = inputParser;
    if test
        addParameter(p, 'out_dir', 'Output', @ischar);
        addParameter(p, 'reps', 1, @isnumeric);
        addParameter(p, 'K_true_vals', [4], @isnumeric);
        addParameter(p, 'SNR_dbs', [10], @isnumeric);
        addParameter(p, 'K_candidates', 5:6, @isnumeric);
    else
        addParameter(p, 'out_dir', 'Output', @ischar);
        addParameter(p, 'reps', 8, @isnumeric);
        addParameter(p, 'K_true_vals', [4 5 6 7], @isnumeric);
        addParameter(p, 'SNR_dbs', [-9 -3  1 0 1 3], @isnumeric);
        addParameter(p, 'K_candidates', 2:10, @isnumeric);
    end;
    addParameter(p, 'duration_s', 300, @isnumeric);
    addParameter(p, 'sfreq', 250, @isnumeric);
    addParameter(p, 'set_file', 'MetaMaps_2023_06.set', @ischar);
    addParameter(p, 'n_workers', 12, @isnumeric);
    addParameter(p, 'cleanup', true, @islogical);
    addParameter(p, 'verbose', true, @islogical);
    addParameter(p, 'montages', {'full', '10-20-20', '10-20-12'}, @iscell);  % montage robustness analysis
    addParameter(p, 'overlap_probs', [0 0.5 1.0], @isnumeric); % run with and without overlap
    addParameter(p, 'overlap_ms_range', [10 40], @isnumeric);
    addParameter(p, 'overlap_strength', 0.5, @isnumeric);
    parse(p, varargin{:});
    
    CONFIG = p.Results;
    util = microstate_utilities_SHARED();

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
    if ~exist(json_dir, 'dir'), mkdir(json_dir); end
    plots_dir = fullfile(CONFIG.out_dir, 'plots');
    if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

    if CONFIG.verbose
        fprintf('\n========================================\n');
        fprintf('Microstate Comparison Pipeline\n');
        fprintf('ALL METHOD × CRITERION COMBINATIONS\n');
        fprintf('========================================\n');
        fprintf('Output: %s\n', CONFIG.out_dir);
        fprintf('Reps: %d | K true: %s | SNR: %s\n', ...
            CONFIG.reps, mat2str(CONFIG.K_true_vals), mat2str(CONFIG.SNR_dbs));
        fprintf('Channels: %d | Workers: %d\n', n_channels, CONFIG.n_workers);
        fprintf('Montages: %s\n', strjoin(CONFIG.montages, ', '));
        fprintf('Overlap probs: %s (ms range %s, strength %.2f)\n\n', ...
            mat2str(CONFIG.overlap_probs), mat2str(CONFIG.overlap_ms_range), CONFIG.overlap_strength);
    end

    % ===== ALL CRITERIA (universal) =====
    all_criteria = {'silhouette', 'free_energy', 'elbow', 'elbow_sil_combined', 'gev'};
    
    method_names = {'kmeans_koenig', 'spm_vb'};
    
    % Montages to test
    n_montages = length(CONFIG.montages);
    overlap_probs = CONFIG.overlap_probs;
    n_overlap_conditions = length(overlap_probs);
    
    % Build list of EEG conditions
    eeg_conditions = [];
    for rep = 1:CONFIG.reps
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
                    elseif strcmp(method_str, 'vb_kmeans')
                        Results = fit_microstate_vb_kmeans(Sim, CONFIG.K_candidates, 'free_energy');
                    elseif strcmp(method_str, 'dp_mixture')
                        Results = fit_microstate_dp_mixture(Sim, CONFIG.K_candidates, 'free_energy');
                    else
                        Results = [];
                    end
                    
                    if ~isempty(Results) && Results.valid_fit
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
                    end
                    
                catch ME
                    fit_results_for_montage{m_idx} = [];
                    fprintf('Fit Error (%s): %s\n', method_str, ME.message);
                end
            end
            
            % ===== STEP 4 & 5: APPLY CRITERIA and SAVE (for each fit) =====
            for m_idx = 1:n_methods
                if isempty(fit_results_for_montage{m_idx})
                    continue;
                end
                
                fit_result = fit_results_for_montage{m_idx};
                Results = fit_result.Results;
                method_str = fit_result.method;
                
                % Apply ALL criteria to this ONE fit
                for c_idx = 1:n_criteria
                    criterion_str = all_criteria{c_idx};
                    
                    % Extract K using this criterion
                    K_selected = select_K_by_criterion(Results, criterion_str);
                    
                    if isnan(K_selected)
                        continue;
                    end
                    
                    rec_metrics = Results.recovery_metrics;
                    recovery_corr = util.padded_vector(rec_metrics.match_similarities, 10);
                    
                    % Generate a temporary subject name (IDs will be fixed later)
                    overlap_tag = sprintf('ovl%02d', round(100 * Sim.overlap_prob));
                    subj_name = sprintf('fit_E%d_M%d_K%d_SNR%+d_%s_%s_%s_%s', ...
                        eeg_idx, m_idx, fit_result.K_true, round(fit_result.SNR_dB), ...
                        montage_type, overlap_tag, method_str, criterion_str);
                    
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
                        'montage_type', montage_type, ...
                        'n_leads', fit_result.n_leads, ...
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
                        'best_score', Results.best_criterion_value, ...
                        'runtime_s', Results.runtime);
                    
                    local_rows = [local_rows; result_row]; 
                    
                    % Save JSON with metadata for plotting & downstream use
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
                        META.montage_type = montage_type;
                        META.n_leads = fit_result.n_leads;
                        META.overlap_prob = Sim.overlap_prob;
                        META.overlap_ms_range = Sim.overlap_ms_range;
                        META.overlap_strength = Sim.overlap_strength;
                        META.n_overlap_events = Sim.n_overlap_events;
                        META.runtime_s = Results.runtime;
                        META.channel_labels = Sim.channel_labels;  % Use Sim.channel_labels (montage-specific)
                        save_microstate_json(Results, Sim, json_file, META);
                    catch ME
                        if CONFIG.verbose
                            fprintf('⚠ Warning: Could not save JSON for %s: %s\n', subj_name, ME.message);
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
    
    % Print error summary
    fprintf('\n========================================\n');
    fprintf('FIT ERROR SUMMARY\n');
    fprintf('========================================\n');
    fprintf('Successful fits: %d / %d\n', n_successful_fits, n_eeg_conditions * n_montages * n_methods);
    fprintf('Failed or invalid: %d\n', n_eeg_conditions * n_montages * n_methods - n_successful_fits);
    if error_types.Count > 0
        fprintf('\nError breakdown:\n');
        error_keys = keys(error_types);
        for k = 1:length(error_keys)
            fprintf('  %s: %d occurrences\n', error_keys{k}, error_types(error_keys{k}));
        end
    end
    fprintf('========================================\n\n');
    
    if isempty(rows)
        error('✗✗✗ NO SUCCESSFUL RUNS! ✗✗✗\nCheck error summary above for details.');
    end
    
    % Convert to table
    T = struct2table(rows);
    
    % Save CSV
    summary_csv = fullfile(res_dir, 'comparison_results.csv');
    writetable(T, summary_csv);
    
    if CONFIG.verbose
        fprintf('✓ Saved: %s\n', summary_csv);
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
    fprintf('JSON Microstates: %s\n', json_dir);
    fprintf('Results CSV: %s\n', res_dir);
    fprintf('========================================\n');
end

% ======================== HELPERS ========================

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
    info.n_eeg_conditions = CONFIG.reps * numel(CONFIG.K_true_vals) * numel(CONFIG.SNR_dbs) * numel(CONFIG.overlap_probs);
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
    
    util = microstate_utilities_SHARED();
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
