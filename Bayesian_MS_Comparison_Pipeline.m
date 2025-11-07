function T = Bayesian_MS_Comparison_Pipeline(varargin)
% BAYESIAN_MS_COMPARISON_PIPELINE: ALL method-criterion combinations
%
% Each method is tested with EVERY criterion (where applicable)
% Results in: n_eeg_conditions × n_methods × n_criteria rows

    addpath("Koenig_code");
    
    test = true;

    p = inputParser;
    if test
        addParameter(p, 'out_dir', './out_microstate_comparison', @ischar);
        addParameter(p, 'reps', 1, @isnumeric);
        addParameter(p, 'K_true_vals', [4], @isnumeric);
        addParameter(p, 'SNR_dbs', [10], @isnumeric);
        addParameter(p, 'K_candidates', 2:3, @isnumeric);
    else
        addParameter(p, 'out_dir', './out_microstate_comparison', @ischar);
        addParameter(p, 'reps', 8, @isnumeric);
        addParameter(p, 'K_true_vals', [4 5 6 7], @isnumeric);
        addParameter(p, 'SNR_dbs', [-10 -7.5 -5 -2.5 0 2.5 5 7.5 10], @isnumeric);
        addParameter(p, 'K_candidates', 2:10, @isnumeric);
    end;
    addParameter(p, 'duration_s', 300, @isnumeric);
    addParameter(p, 'sfreq', 250, @isnumeric);
    addParameter(p, 'set_file', 'MetaMaps_2023_06.set', @ischar);
    addParameter(p, 'n_workers', 10, @isnumeric);
    addParameter(p, 'cleanup', true, @islogical);
    addParameter(p, 'verbose', true, @islogical);
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
        fprintf('Channels: %d | Workers: %d\n\n', n_channels, CONFIG.n_workers);
    end

    % ===== ALL CRITERIA (universal) =====
    all_criteria = {'silhouette', 'free_energy', 'elbow', 'elbow_sil_combined', 'gev'};
    
    method_names = {'kmeans_koenig', 'spm_vb', 'vb_kmeans'};
    
    % Build list of EEG conditions
    eeg_conditions = [];
    for rep = 1:CONFIG.reps
        for K_true = CONFIG.K_true_vals
            for SNR_dB = CONFIG.SNR_dbs
                eeg_conditions = [eeg_conditions; rep, K_true, SNR_dB]; %#ok<AGROW>
            end
        end
    end
    
    n_eeg_conditions = size(eeg_conditions, 1);
    n_methods = length(method_names);
    n_criteria = length(all_criteria);
    n_fits = n_eeg_conditions * n_methods;
    n_total_results = n_fits * n_criteria;
    
    if CONFIG.verbose
        fprintf('PIPELINE STRUCTURE:\n');
        fprintf('  EEG conditions: %d\n', n_eeg_conditions);
        fprintf('  Methods: %d\n', n_methods);
        fprintf('  Criteria (applied to all): %d\n', n_criteria);
        fprintf('  Total FITS: %d (one per method+EEG)\n', n_fits);
        fprintf('  Total RESULTS: %d (fits × criteria)\n\n', n_total_results);
    end

    % ===== STAGE 1: GENERATE ALL EEGs =====
    if CONFIG.verbose
        fprintf('Stage 1: Generating %d EEG conditions...\n', n_eeg_conditions);
    end
    
    eeg_bank = cell(n_eeg_conditions, 1);
    pb = util.progress_bar(n_eeg_conditions, 'EEG Gen');
    
    for eeg_idx = 1:n_eeg_conditions
        rep = eeg_conditions(eeg_idx, 1);
        K_true = eeg_conditions(eeg_idx, 2);
        SNR_dB = eeg_conditions(eeg_idx, 3);
        
        try
            sim_seed = 42 + rep*1000 + K_true*100 + round((SNR_dB+10)*10);
            [Sim, ~, ~] = generate_microstate_eeg(K_true, SNR_dB, CONFIG.duration_s, ...
                CONFIG.sfreq, sim_seed);
            eeg_bank{eeg_idx} = Sim;
        catch ME
            fprintf('\n✗ EEG Generation Error (Rep %d, K=%d, SNR=%+.1f): %s\n', ...
                rep, K_true, SNR_dB, ME.message);
            fprintf('  Stack trace: %s\n', ME.getReport());
            eeg_bank{eeg_idx} = [];
        end
        
        pb.update();
    end
    pb.done();

    % ===== STAGE 2: PARALLEL FITTING =====
    if CONFIG.verbose
        fprintf('\nStage 2: Fitting %d method instances (parallel)...\n', n_fits);
    end
    
    fit_tasks = [];
    fit_id = 0;
    
    for eeg_idx = 1:n_eeg_conditions
        rep = eeg_conditions(eeg_idx, 1);
        K_true = eeg_conditions(eeg_idx, 2);
        SNR_dB = eeg_conditions(eeg_idx, 3);
        
        for m_idx = 1:n_methods
            fit_id = fit_id + 1;
            method_str = method_names{m_idx};
            
            fit_tasks = [fit_tasks; struct( ...
                'fit_id', fit_id, ...
                'eeg_idx', eeg_idx, ...
                'rep', rep, ...
                'K_true', K_true, ...
                'SNR_dB', SNR_dB, ...
                'method', {method_str})]; %#ok<AGROW>
        end
    end
    
    n_fit_tasks = length(fit_tasks);
    fit_results = cell(n_fit_tasks, 1);
    fit_errors = cell(n_fit_tasks, 1);
    
    try
        p_pool = gcp('nocreate');
        if isempty(p_pool)
            parpool('local', CONFIG.n_workers);
        end
        
        pb = util.progress_bar(n_fit_tasks, 'Fits');
        
        parfor fit_idx = 1:n_fit_tasks
            fit_task = fit_tasks(fit_idx);
            Sim = eeg_bank{fit_task.eeg_idx};
            
            if isempty(Sim)
                fit_results{fit_idx} = [];
                fit_errors{fit_idx} = 'Empty EEG';
                pb.update();
                continue;
            end
            
            try
                % FIT ONCE with ANY valid criterion
                if strcmp(fit_task.method, 'spm_vb')
                    Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, 'elbow_sil_combined');
                elseif strcmp(fit_task.method, 'kmeans_koenig')
                    Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_candidates, 'silhouette');
                elseif strcmp(fit_task.method, 'vb_kmeans')
                    Results = fit_microstate_vb_kmeans(Sim, CONFIG.K_candidates, 'free_energy');
                elseif strcmp(fit_task.method, 'dp_mixture')
                    Results = fit_microstate_dp_mixture(Sim, CONFIG.K_candidates, 'free_energy');
                else
                    Results = [];
                end
                
                if ~isempty(Results) && Results.valid_fit
                    fit_results{fit_idx} = struct(...
                        'fit_id', fit_task.fit_id, ...
                        'eeg_idx', fit_task.eeg_idx, ...
                        'rep', fit_task.rep, ...
                        'K_true', fit_task.K_true, ...
                        'SNR_dB', fit_task.SNR_dB, ...
                        'method', fit_task.method, ...
                        'Results', Results, ...
                        'Sim', Sim);
                    fit_errors{fit_idx} = [];
                else
                    fit_results{fit_idx} = [];
                    fit_errors{fit_idx} = 'Invalid fit';
                end
                
            catch ME
                fit_results{fit_idx} = [];
                fit_errors{fit_idx} = sprintf('%s: %s', ME.identifier, ME.message);
            end
            
            pb.update();
        end
        pb.done();
        
    catch
        % Fallback: sequential
        if CONFIG.verbose
            fprintf('\n⚠ Parallel pool unavailable. Running sequentially...\n');
        end
        pb = util.progress_bar(n_fit_tasks, 'Fits');
        
        for fit_idx = 1:n_fit_tasks
            fit_task = fit_tasks(fit_idx);
            Sim = eeg_bank{fit_task.eeg_idx};
            
            if isempty(Sim)
                fit_results{fit_idx} = [];
                fit_errors{fit_idx} = 'Empty EEG';
                pb.update();
                continue;
            end
            
            try
                if strcmp(fit_task.method, 'spm_vb')
                    Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, 'elbow_sil_combined');
                elseif strcmp(fit_task.method, 'kmeans_koenig')
                    Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_candidates, 'silhouette');
                elseif strcmp(fit_task.method, 'vb_kmeans')
                    Results = fit_microstate_vb_kmeans(Sim, CONFIG.K_candidates, 'free_energy');
                elseif strcmp(fit_task.method, 'dp_mixture')
                    Results = fit_microstate_dp_mixture(Sim, CONFIG.K_candidates, 'free_energy');
                else
                    Results = [];
                end
                
                if ~isempty(Results) && Results.valid_fit
                    fit_results{fit_idx} = struct(...
                        'fit_id', fit_task.fit_id, ...
                        'eeg_idx', fit_task.eeg_idx, ...
                        'rep', fit_task.rep, ...
                        'K_true', fit_task.K_true, ...
                        'SNR_dB', fit_task.SNR_dB, ...
                        'method', fit_task.method, ...
                        'Results', Results, ...
                        'Sim', Sim);
                    fit_errors{fit_idx} = [];
                else
                    fit_results{fit_idx} = [];
                    fit_errors{fit_idx} = 'Invalid fit';
                end
                
            catch ME
                fit_results{fit_idx} = [];
                fit_errors{fit_idx} = sprintf('%s: %s', ME.identifier, ME.message);
            end
            
            pb.update();
        end
        pb.done();
    end

    % Print error summary
    fprintf('\n========================================\n');
    fprintf('FIT ERROR SUMMARY\n');
    fprintf('========================================\n');
    n_successful = 0;
    error_types = containers.Map();
    for fit_idx = 1:n_fit_tasks
        if isempty(fit_results{fit_idx})
            error_msg = fit_errors{fit_idx};
            if isKey(error_types, error_msg)
                error_types(error_msg) = error_types(error_msg) + 1;
            else
                error_types(error_msg) = 1;
            end
        else
            n_successful = n_successful + 1;
        end
    end
    
    fprintf('Successful: %d / %d\n', n_successful, n_fit_tasks);
    fprintf('\nError breakdown:\n');
    error_keys = keys(error_types);
    for k = 1:length(error_keys)
        fprintf('  %s: %d occurrences\n', error_keys{k}, error_types(error_keys{k}));
    end
    fprintf('========================================\n\n');

    % ===== STAGE 3: APPLY ALL CRITERIA =====
    if CONFIG.verbose
        fprintf('Stage 3: Applying all criteria to fits...\n');
    end
    
    rows = [];
    run_id = 0;
    n_failed = 0;
    
    for fit_idx = 1:n_fit_tasks
        if isempty(fit_results{fit_idx})
            n_failed = n_failed + n_criteria;
            continue;
        end
        
        fit_result = fit_results{fit_idx};
        Results = fit_result.Results;
        Sim = fit_result.Sim;
        method_str = fit_result.method;
        
        % Apply ALL criteria to this ONE fit
        for c_idx = 1:n_criteria
            run_id = run_id + 1;
            criterion_str = all_criteria{c_idx};
            
            % Extract K using this criterion
            K_selected = select_K_by_criterion(Results, criterion_str);
            
            if isnan(K_selected)
                continue;
            end
            
            rec_metrics = Results.recovery_metrics;
            recovery_corr = util.padded_vector(rec_metrics.match_similarities, 10);
            
            subj_name = sprintf('fit_%03d_K%d_SNR%+d_%s_%s', ...
                run_id, fit_result.K_true, round(fit_result.SNR_dB), ...
                method_str, criterion_str);
            
            result_row = struct( ...
                'fit_id', fit_result.fit_id, ...
                'subject', subj_name, ...
                'rep', fit_result.rep, ...
                'method', method_str, ...
                'criterion', criterion_str, ...
                'K_true', fit_result.K_true, ...
                'SNR_dB', fit_result.SNR_dB, ...
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
            
            rows = [rows; result_row]; %#ok<AGROW>
            
            % Save JSON
            json_file = fullfile(json_dir, [subj_name '.json']);
            try
                save_microstate_json(Results, Sim, json_file);
            catch ME
                fprintf('⚠ Warning: Could not save JSON for %s: %s\n', subj_name, ME.message);
            end
        end
    end
    
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
    fprintf('EEG Conditions: %d\n', n_eeg_conditions);
    fprintf('Methods: %d\n', n_methods);
    fprintf('Criteria (universal): %d\n', n_criteria);
    fprintf('Total Fits: %d\n', n_fit_tasks);
    fprintf('Total Results: %d (one per method×criterion per EEG)\n', height(T));
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
    k_norm = (0:(n-1)) / (n-1);
    
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
    info.n_eeg_conditions = CONFIG.reps * numel(CONFIG.K_true_vals) * numel(CONFIG.SNR_dbs);
    
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
        
        fprintf('%-20s %9.1f%% %10.3f %10.1f %9.1fs\n', m, acc, f1, matched, rt);
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