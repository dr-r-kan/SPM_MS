function T = VBGMM_MS_Comparison_Pipeline(varargin)
% VBGMM_MS_COMPARISON_PIPELINE: Compare VB methods and model selection criteria
%
% Compares:
%   - SPM VB-GMM (elbow+silhouette combined, elbow only)
%   - Standard K-means (silhouette, GEV, elbow)
%   - VB K-means (free energy, silhouette)
%   - Dirichlet Process Mixtures (free energy, silhouette)
%
% USAGE:
%   T = VBGMM_MS_Comparison_Pipeline();
%   T = VBGMM_MS_Comparison_Pipeline('reps', 5, 'K_true_vals', [4 5 6]);
%
% OUTPUTS:
%   T - Results table comparing all methods and criteria

    % Parse inputs
    p = inputParser;
    addParameter(p, 'out_dir', './out_microstate_comparison_final', @ischar);
    addParameter(p, 'reps', 5, @isnumeric);
    addParameter(p, 'K_true_vals', [4 5 6], @isnumeric);
    addParameter(p, 'SNR_dbs', [-5 0 2.5 5 7.5], @isnumeric);
    addParameter(p, 'K_candidates', 2:10, @isnumeric);
    addParameter(p, 'duration_s', 300, @isnumeric);
    addParameter(p, 'sfreq', 250, @isnumeric);
    parse(p, varargin{:});
    
    CONFIG = p.Results;

    % Setup output directory
    if ~exist(CONFIG.out_dir, 'dir'), mkdir(CONFIG.out_dir); end
    res_dir = fullfile(CONFIG.out_dir, 'results');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end
    plots_dir = fullfile(CONFIG.out_dir, 'plots');
    if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

    fprintf('\n========================================\n');
    fprintf('Microstate Method Comparison - FINAL\n');
    fprintf('========================================\n');
    fprintf('Methods & Criteria:\n');
    fprintf('  1. SPM VB-GMM: elbow+sil, elbow_only\n');
    fprintf('  2. Standard K-means: silhouette, gev, elbow\n');
    fprintf('  3. VB K-means: free_energy, silhouette\n');
    fprintf('  4. Dirichlet Process: free_energy, silhouette\n');
    fprintf('Output: %s\n', CONFIG.out_dir);
    fprintf('Reps: %d | K_true: %s | SNR: %s\n\n', ...
        CONFIG.reps, mat2str(CONFIG.K_true_vals), mat2str(CONFIG.SNR_dbs));

    % Methods and their criteria
    methods_criteria = struct(...
        'spm_vb', {{'elbow_sil_combined', 'elbow_only'}}, ...
        'kmeans_standard', {{'silhouette', 'gev', 'elbow'}}, ...
        'vb_kmeans', {{'free_energy', 'silhouette'}}, ...
        'dp_mixture', {{'free_energy', 'silhouette'}});
    
    method_names = fieldnames(methods_criteria);
    
    % Calculate total runs
    total_runs = 0;
    for m_idx = 1:length(method_names)
        criteria = methods_criteria.(method_names{m_idx});
        total_runs = total_runs + length(criteria);
    end
    total_runs = total_runs * CONFIG.reps * numel(CONFIG.K_true_vals) * numel(CONFIG.SNR_dbs);
    
    % Main grid
    rows = [];
    run_id = 0;
    
    pb = progbar(total_runs, 'Comparison');

    for rep = 1:CONFIG.reps
        for K_true = CONFIG.K_true_vals
            for SNR_dB = CONFIG.SNR_dbs
                
                % Generate EEG once per condition
                fprintf('\n[Rep %d/%d] K=%d, SNR=%+.1f dB\n', ...
                    rep, CONFIG.reps, K_true, SNR_dB);
                    
                Sim = generate_microstate_eeg(K_true, SNR_dB, CONFIG.duration_s, ...
                    CONFIG.sfreq, 42 + rep*1000 + K_true*100 + round((SNR_dB+10)*10));
                
                % Test all method x criterion combinations
                for m_idx = 1:length(method_names)
                    method_str = method_names{m_idx};
                    criteria = methods_criteria.(method_str);
                    
                    for criterion = criteria
                        run_id = run_id + 1;
                        criterion_str = criterion{1};
                        
                        subj_name = sprintf('run_%03d_K%d_SNR%+d_%s_%s', ...
                            run_id, K_true, round(SNR_dB), method_str, criterion_str);
                        
                        fprintf('  [%d/%d] %s+%s... ', ...
                            run_id, total_runs, method_str, criterion_str);
                        
                        try
                            % Fit model based on method
                            if strcmp(method_str, 'spm_vb')
                                Results = fit_microstate_spm_vb(Sim, CONFIG.K_candidates, criterion_str);
                                
                            elseif strcmp(method_str, 'kmeans_standard')
                                Results = fit_microstate_kmeans_standard(Sim, CONFIG.K_candidates, criterion_str);
                                
                            elseif strcmp(method_str, 'vb_kmeans')
                                Results = fit_microstate_vb_kmeans(Sim, CONFIG.K_candidates, criterion_str);
                                
                            elseif strcmp(method_str, 'dp_mixture')
                                Results = fit_microstate_dp_mixture(Sim, CONFIG.K_candidates, criterion_str);
                                
                            else
                                error('Unknown method: %s', method_str);
                            end
                            
                            if ~Results.valid_fit
                                fprintf('FAILED\n');
                                pb.update();
                                continue;
                            end
                            
                            fprintf('K=%d, Rec=%.3f, AvgPerState=%.3f\n', ...
                                Results.K_estimated, Results.mean_recovery, ...
                                Results.avg_recovery_per_state);
                            
                            % Save data
                            data_file = fullfile(res_dir, [subj_name '.mat']);
                            save(data_file, 'Sim', 'Results', '-v7');
                            
                            % Accumulate results
                            recovery_corr = pad_vector(Results.recovery_corr, 10);
                            
                            row = struct( ...
                                'run_id', run_id, ...
                                'subject', {subj_name}, ...
                                'rep', rep, ...
                                'method', {method_str}, ...
                                'criterion', {criterion_str}, ...
                                'K_true', K_true, ...
                                'SNR_dB', SNR_dB, ...
                                'K_estimated', Results.K_estimated, ...
                                'K_correct', K_true == Results.K_estimated, ...
                                'K_error', abs(K_true - Results.K_estimated), ...
                                'n_maps', Results.n_maps, ...
                                'mean_recovery', Results.mean_recovery, ...
                                'avg_recovery_per_state', Results.avg_recovery_per_state, ...
                                'recovery_01', recovery_corr(1), ...
                                'recovery_02', recovery_corr(2), ...
                                'recovery_03', recovery_corr(3), ...
                                'best_score', Results.best_criterion_value, ...
                                'runtime_s', Results.runtime);
                            
                            rows = [rows; row]; %#ok<AGROW>
                            
                        catch ME
                            fprintf('ERROR: %s\n', ME.message);
                            fprintf('%s\n', ME.getReport('basic'));
                        end
                        
                        pb.update();
                    end
                end
            end
        end
    end
    pb.done();

    % Compile results table
    fprintf('\n\nCompiling results table...\n');
    if isempty(rows)
        warning('No successful runs!');
        T = table();
        return;
    end
    
    T = struct2table(rows);
    
    % Save summary
    summary_csv = fullfile(res_dir, 'comparison_results_final.csv');
    writetable(T, summary_csv);
    fprintf('Results saved to: %s\n', summary_csv);
    
    % Print summary statistics
    print_comparison_summary_final(T);
    
    % Create comparison plots
    fprintf('\nCreating comparison plots...\n');
    try
        create_comparison_plots_final(T, plots_dir);
    catch ME
        warning('Plotting failed: %s', ME.message);
        fprintf('%s\n', ME.getReport());
    end
    
    fprintf('\n========================================\n');
    fprintf('Comparison pipeline complete!\n');
    fprintf('========================================\n');
end

% ======================== SUMMARY STATISTICS ========================

function print_comparison_summary_final(T)
    fprintf('\n========================================\n');
    fprintf('COMPARISON SUMMARY - FINAL\n');
    fprintf('========================================\n');
    
    methods = unique(T.method);
    criteria = unique(T.criterion);
    
    % Overall accuracy by method and criterion
    fprintf('\nK Selection Accuracy:\n');
    fprintf('%-20s %-20s %10s %10s %10s %10s\n', ...
        'Method', 'Criterion', 'Accuracy', 'Recovery', 'AvgPerSt', 'Runtime');
    fprintf('%s\n', repmat('-', 1, 90));
    
    for m = methods'
        for c = criteria'
            mask = strcmp(T.method, m{1}) & strcmp(T.criterion, c{1});
            if ~any(mask), continue; end
            
            acc = 100 * mean(T.K_correct(mask));
            rec = mean(T.mean_recovery(mask), 'omitnan');
            avg_per_state = mean(T.avg_recovery_per_state(mask), 'omitnan');
            rt = mean(T.runtime_s(mask), 'omitnan');
            
            fprintf('%-20s %-20s %9.1f%% %10.3f %10.3f %9.1fs\n', ...
                m{1}, c{1}, acc, rec, avg_per_state, rt);
        end
    end
    
    % Highlight best performers
    fprintf('\n--- BEST PERFORMERS ---\n');
    [~, idx_acc] = max(T.K_correct);
    fprintf('Best Accuracy: %s + %s (%.1f%%, Recovery=%.3f, AvgPerState=%.3f)\n', ...
        T.method{idx_acc}, T.criterion{idx_acc}, 100*T.K_correct(idx_acc), ...
        T.mean_recovery(idx_acc), T.avg_recovery_per_state(idx_acc));
    
    [~, idx_rec] = max(T.mean_recovery);
    fprintf('Best Recovery: %s + %s (Recovery=%.3f, Acc=%d%%, AvgPerState=%.3f)\n', ...
        T.method{idx_rec}, T.criterion{idx_rec}, T.mean_recovery(idx_rec), ...
        100*T.K_correct(idx_rec), T.avg_recovery_per_state(idx_rec));
    
    [~, idx_avg] = max(T.avg_recovery_per_state);
    fprintf('Best Avg Per State: %s + %s (AvgPerState=%.3f, Acc=%d%%, Recovery=%.3f)\n', ...
        T.method{idx_avg}, T.criterion{idx_avg}, T.avg_recovery_per_state(idx_avg), ...
        100*T.K_correct(idx_avg), T.mean_recovery(idx_avg));
    
    % By SNR
    fprintf('\n\nPerformance by SNR:\n');
    snr_vals = unique(T.SNR_dB);
    for snr = snr_vals'
        fprintf('\nSNR = %+.0f dB:\n', snr);
        fprintf('  %-20s %-20s %10s %10s %10s\n', ...
            'Method', 'Criterion', 'Accuracy', 'Recovery', 'AvgPerSt');
        fprintf('  %s\n', repmat('-', 1, 70));
        
        for m = methods'
            for c = criteria'
                mask = strcmp(T.method, m{1}) & strcmp(T.criterion, c{1}) & T.SNR_dB == snr;
                if ~any(mask), continue; end
                
                acc = 100 * mean(T.K_correct(mask));
                rec = mean(T.mean_recovery(mask), 'omitnan');
                avg_per_state = mean(T.avg_recovery_per_state(mask), 'omitnan');
                
                fprintf('  %-20s %-20s %9.1f%% %10.3f %10.3f\n', ...
                    m{1}, c{1}, acc, rec, avg_per_state);
            end
        end
    end
    
    % Winner analysis
    fprintf('\n\nWINNER ANALYSIS (by condition):\n');
    method_combos = unique(strcat(T.method, '+', T.criterion));
    best_count = zeros(length(method_combos), 1);
    
    snr_k_combos = unique([T.SNR_dB, T.K_true], 'rows');
    for i = 1:size(snr_k_combos, 1)
        snr = snr_k_combos(i, 1);
        k = snr_k_combos(i, 2);
        
        % Find best for this condition
        cond_mask = T.SNR_dB == snr & T.K_true == k;
        if ~any(cond_mask), continue; end
        
        best_acc_in_cond = max(T.K_correct(cond_mask));
        
        % Count wins for each method+criterion
        for j = 1:length(method_combos)
            parts = strsplit(method_combos{j}, '+');
            mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2}) & cond_mask;
            
            if any(mask) && mean(T.K_correct(mask)) >= best_acc_in_cond - 0.01
                best_count(j) = best_count(j) + 1;
            end
        end
    end
    
    [sorted_counts, sort_idx] = sort(best_count, 'descend');
    fprintf('Times each method won (or tied):\n');
    for i = 1:min(10, length(method_combos))
        if sorted_counts(i) > 0
            fprintf('  %s: %d wins\n', method_combos{sort_idx(i)}, sorted_counts(i));
        end
    end
end

% ======================== COMPARISON PLOTS ========================

function create_comparison_plots_final(T, plots_dir)
    
    methods = unique(T.method);
    criteria = unique(T.criterion);
    
    % Create method+criterion labels
    combo_labels = cell(0);
    for m = methods'
        for c = criteria'
            mask = strcmp(T.method, m{1}) & strcmp(T.criterion, c{1});
            if any(mask)
                combo_labels{end+1} = sprintf('%s+%s', m{1}, c{1}); %#ok<AGROW>
            end
        end
    end
    n_combos = length(combo_labels);
    
    % Main comparison figure
    fig = figure('Position', [50, 50, 1800, 1200], 'Visible', 'off');
    
    % Plot 1: Accuracy comparison
    subplot(3, 3, 1);
    acc_vals = zeros(n_combos, 1);
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2});
        if any(mask)
            acc_vals(i) = 100 * mean(T.K_correct(mask));
        end
    end
    bar(acc_vals);
    set(gca, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45);
    ylabel('K Selection Accuracy (%)');
    title('Overall Accuracy Comparison');
    grid on;
    ylim([0 105]);
    
    % Plot 2: Recovery comparison
    subplot(3, 3, 2);
    rec_vals = zeros(n_combos, 1);
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2});
        if any(mask)
            rec_vals(i) = mean(T.mean_recovery(mask), 'omitnan');
        end
    end
    bar(rec_vals);
    set(gca, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45);
    ylabel('Mean Map Recovery');
    title('Map Recovery (Best Match)');
    grid on;
    ylim([0 1]);
    
    % Plot 3: Avg recovery per state
    subplot(3, 3, 3);
    avg_vals = zeros(n_combos, 1);
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2});
        if any(mask)
            avg_vals(i) = mean(T.avg_recovery_per_state(mask), 'omitnan');
        end
    end
    bar(avg_vals);
    set(gca, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45);
    ylabel('Avg Recovery Per State');
    title('Recovery Per Extracted State');
    grid on;
    ylim([0 1]);
    
    % Plot 4: Accuracy vs SNR
    subplot(3, 3, 4);
    snr_vals = unique(T.SNR_dB);
    colors = lines(n_combos);
    hold on;
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        acc_snr = zeros(size(snr_vals));
        for j = 1:numel(snr_vals)
            mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2}) & ...
                   T.SNR_dB == snr_vals(j);
            if any(mask)
                acc_snr(j) = 100 * mean(T.K_correct(mask));
            end
        end
        plot(snr_vals, acc_snr, 'o-', 'LineWidth', 2, 'Color', colors(i, :), ...
            'DisplayName', combo_labels{i});
    end
    xlabel('SNR (dB)');
    ylabel('Accuracy (%)');
    title('Accuracy vs SNR');
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    
    % Plot 5: K error distribution
    subplot(3, 3, 5);
    hold on;
    for i = 1:min(n_combos, 5)  % Only top 5 for clarity
        parts = strsplit(combo_labels{i}, '+');
        mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2});
        if any(mask)
            histogram(T.K_error(mask), 'BinEdges', -0.5:1:10.5, ...
                'DisplayName', combo_labels{i}, 'FaceAlpha', 0.4);
        end
    end
    xlabel('|K_{true} - K_{est}|');
    ylabel('Frequency');
    title('K Estimation Error');
    legend('Location', 'best', 'FontSize', 8);
    grid on;
    
    % Plot 6: Recovery vs K_true
    subplot(3, 3, 6);
    k_vals = unique(T.K_true);
    hold on;
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        rec_k = zeros(size(k_vals));
        for j = 1:numel(k_vals)
            mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2}) & ...
                   T.K_true == k_vals(j);
            if any(mask)
                rec_k(j) = mean(T.avg_recovery_per_state(mask), 'omitnan');
            end
        end
        plot(k_vals, rec_k, 's-', 'LineWidth', 2, 'MarkerSize', 8, ...
            'Color', colors(i, :), 'DisplayName', combo_labels{i});
    end
    xlabel('True K');
    ylabel('Avg Recovery Per State');
    title('Recovery vs True K');
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    
    % Plot 7: Runtime comparison
    subplot(3, 3, 7);
    rt_vals = zeros(n_combos, 1);
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2});
        if any(mask)
            rt_vals(i) = mean(T.runtime_s(mask), 'omitnan');
        end
    end
    bar(rt_vals);
    set(gca, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45);
    ylabel('Runtime (s)');
    title('Computational Cost');
    grid on;
    
    % Plot 8: Scatter - Recovery vs Accuracy
    subplot(3, 3, 8);
    hold on;
    for i = 1:n_combos
        parts = strsplit(combo_labels{i}, '+');
        mask = strcmp(T.method, parts{1}) & strcmp(T.criterion, parts{2});
        if any(mask)
            scatter(T.avg_recovery_per_state(mask), 100*T.K_correct(mask), 100, ...
                colors(i, :), 'filled', 'DisplayName', combo_labels{i});
        end
    end
    xlabel('Avg Recovery Per State');
    ylabel('K Correct (%)');
    title('Recovery vs Accuracy Trade-off');
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    
    % Plot 9: Method comparison summary
    subplot(3, 3, 9);
    axis off;
    % Calculate overall stats
    best_acc_idx = find(acc_vals == max(acc_vals), 1);
    best_rec_idx = find(avg_vals == max(avg_vals), 1);
    fastest_idx = find(rt_vals == min(rt_vals(rt_vals > 0)), 1);
    
    summary_str = sprintf(...
        'OVERALL SUMMARY\n\n' + ...
        'Best Accuracy:\n  %s\n  %.1f%%\n\n' + ...
        'Best Recovery:\n  %s\n  %.3f\n\n' + ...
        'Fastest:\n  %s\n  %.1fs', ...
        combo_labels{best_acc_idx}, acc_vals(best_acc_idx), ...
        combo_labels{best_rec_idx}, avg_vals(best_rec_idx), ...
        combo_labels{fastest_idx}, rt_vals(fastest_idx));
    
    text(0.1, 0.5, summary_str, 'FontSize', 10, ...
        'VerticalAlignment', 'middle', 'Interpreter', 'none');
    
    sgtitle('Microstate Method Comparison - FINAL RESULTS', ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    saveas(fig, fullfile(plots_dir, 'method_comparison_final.png'));
    close(fig);
    
    fprintf('Saved comparison plots to %s\n', plots_dir);
end

% ======================== UTILITIES ========================

function v_pad = pad_vector(v, n)
    v_pad = nan(1, n);
    if ~isempty(v) && ~all(isnan(v))
        v_pad(1:min(length(v), n)) = v(1:min(length(v), n));
    end
end

function pb = progbar(total, label)
    if nargin < 2, label = ''; end
    c = struct('n', 0, 'N', total, 't0', tic, 'label', label);
    pb.update = @() local_update();
    pb.done = @() fprintf('\n');
    function local_update()
        c.n = c.n + 1;
        if mod(c.n, max(1, floor(c.N/50))) == 0 || c.n == c.N
            dt = toc(c.t0);
            eta = dt * (c.N - c.n) / max(c.n, 1);
            fprintf('\r[%s] %d/%d (%.1f%%) - ETA %s', c.label, c.n, c.N, ...
                100*c.n/c.N, dur_str(eta));
        end
    end
end

function s = dur_str(t)
    if t < 60
        s = sprintf('%.0fs', t);
    else
        m = floor(t / 60);
        ssec = mod(t, 60);
        s = sprintf('%dm%.0fs', m, ssec);
    end
end