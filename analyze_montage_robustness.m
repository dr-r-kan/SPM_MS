function analyze_montage_robustness(varargin)
% ANALYZE_MONTAGE_ROBUSTNESS: Comprehensive analysis of montage effects
%
% Analyzes how reduced EEG montages (71 → 20 → 12 leads) affect:
%   - K estimation accuracy
%   - Microstate center identification precision
%   - Recovery metrics stability
%
% USAGE:
%   analyze_montage_robustness()
%   analyze_montage_robustness(results_csv)
%   analyze_montage_robustness(results_csv, 'output_dir', out_dir)
%   analyze_montage_robustness(..., 'K_true', [4 5], 'SNR_dB', [-10 10])
%
% INPUTS:
%   results_csv - Path to comparison_results.csv (optional)
%   'output_dir' - Directory for plots and summary files
%   'K_true' - Filter by true K values
%   'SNR_dB' - Filter by SNR values
%   'method' - Filter by method name
%   'criterion' - Filter by criterion name
%
% OUTPUTS:
%   Generates plots:
%     - K estimation accuracy vs lead count
%     - Mean absolute K error vs lead count
%     - Center correlation vs lead count
%     - Center precision vs lead count
%     - Heatmaps (method × montage performance)
%     - SNR interaction plots
%   Exports summary_montage_robustness.csv

    % Parse inputs
    p = inputParser;
    default_csv = 'E:\OneDrive - University College London\Microstates\Variational Bayesian Microstate Extraction\out_microstate_comparison\results\comparison_results.csv';
    default_out_dir = 'E:\OneDrive - University College London\Microstates\Variational Bayesian Microstate Extraction\out_microstate_comparison\montage_analysis';
    
    addOptional(p, 'results_csv', default_csv, @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', default_out_dir, @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_true', [], @isnumeric);
    addParameter(p, 'SNR_dB', [], @isnumeric);
    addParameter(p, 'method', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'criterion', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    
    results_csv = char(p.Results.results_csv);
    output_dir = char(p.Results.output_dir);
    filter_K = p.Results.K_true;
    filter_SNR = p.Results.SNR_dB;
    filter_method = char(p.Results.method);
    filter_criterion = char(p.Results.criterion);
    
    % Create output directory
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    fprintf('\n========================================\n');
    fprintf('Montage Robustness Analysis\n');
    fprintf('========================================\n');
    fprintf('Results CSV: %s\n', results_csv);
    fprintf('Output directory: %s\n\n', output_dir);
    
    % Load data
    if ~exist(results_csv, 'file')
        error('Results file not found: %s', results_csv);
    end
    
    T = readtable(results_csv);
    fprintf('✓ Loaded %d observations\n', height(T));
    
    % Check for montage columns
    if ~ismember('montage_type', T.Properties.VariableNames)
        error('Results file does not contain montage_type column. Please run pipeline with montage analysis enabled.');
    end
    
    % Apply filters
    if ~isempty(filter_K)
        T = T(ismember(T.K_true, filter_K), :);
        fprintf('  Filtered to K_true = %s: %d obs\n', mat2str(filter_K), height(T));
    end
    
    if ~isempty(filter_SNR)
        T = T(ismember(T.SNR_dB, filter_SNR), :);
        fprintf('  Filtered to SNR_dB = %s: %d obs\n', mat2str(filter_SNR), height(T));
    end
    
    if ~isempty(filter_method)
        T = T(strcmp(T.method, filter_method), :);
        fprintf('  Filtered to method = %s: %d obs\n', filter_method, height(T));
    end
    
    if ~isempty(filter_criterion)
        T = T(strcmp(T.criterion, filter_criterion), :);
        fprintf('  Filtered to criterion = %s: %d obs\n', filter_criterion, height(T));
    end
    
    fprintf('\n');
    
    % Get unique montages and methods
    montages = unique(T.montage_type);
    methods = unique(T.method);
    criteria = unique(T.criterion);
    
    fprintf('Montages found: %s\n', strjoin(montages, ', '));
    fprintf('Methods: %s\n', strjoin(methods, ', '));
    fprintf('Criteria: %s\n', strjoin(criteria, ', '));
    fprintf('\n');
    
    % ===== ANALYSIS 1: K ESTIMATION ACCURACY VS LEAD COUNT =====
    fprintf('Generating K estimation accuracy plots...\n');
    plot_k_accuracy_by_montage(T, output_dir);
    
    % ===== ANALYSIS 2: RECOVERY METRICS VS LEAD COUNT =====
    fprintf('Generating recovery metrics plots...\n');
    plot_recovery_by_montage(T, output_dir);
    
    % ===== ANALYSIS 3: HEATMAPS (METHOD × MONTAGE) =====
    fprintf('Generating heatmaps...\n');
    plot_heatmaps(T, output_dir);
    
    % ===== ANALYSIS 4: SNR INTERACTION PLOTS =====
    fprintf('Generating SNR interaction plots...\n');
    plot_snr_interactions(T, output_dir);
    
    % ===== ANALYSIS 5: SUMMARY STATISTICS =====
    fprintf('Computing summary statistics...\n');
    summary_stats = compute_summary_statistics(T);
    
    % Export summary to CSV
    summary_csv = fullfile(output_dir, 'summary_montage_robustness.csv');
    writetable(summary_stats, summary_csv);
    fprintf('✓ Saved summary statistics: %s\n', summary_csv);
    
    fprintf('\n========================================\n');
    fprintf('Analysis Complete!\n');
    fprintf('Plots saved to: %s\n', output_dir);
    fprintf('========================================\n\n');
end

% ======================== PLOTTING FUNCTIONS ========================

function plot_k_accuracy_by_montage(T, output_dir)
    % K estimation accuracy vs lead count by method
    
    montages = unique(T.montage_type);
    methods = unique(T.method);
    
    % Get lead counts for each montage
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    
    % Sort by lead count
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    figure('Position', [100 100 1200 500]);
    
    % Plot 1: Accuracy by method
    subplot(1, 2, 1);
    hold on;
    colors = lines(length(methods));
    
    for m = 1:length(methods)
        method = methods{m};
        acc = zeros(size(montages));
        err = zeros(size(montages));
        
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, method);
            if any(idx)
                acc(i) = mean(T.K_correct(idx));
                err(i) = std(T.K_correct(idx)) / sqrt(sum(idx));
            end
        end
        
        errorbar(lead_counts, acc * 100, err * 100, 'o-', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(m, :), ...
            'DisplayName', strrep(method, '_', ' '));
    end
    
    xlabel('Number of Leads');
    ylabel('K Estimation Accuracy (%)');
    title('K Estimation Accuracy vs Lead Count');
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    
    % Plot 2: Mean absolute error
    subplot(1, 2, 2);
    hold on;
    
    for m = 1:length(methods)
        method = methods{m};
        mae = zeros(size(montages));
        err = zeros(size(montages));
        
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, method);
            if any(idx)
                mae(i) = mean(T.K_error(idx));
                err(i) = std(T.K_error(idx)) / sqrt(sum(idx));
            end
        end
        
        errorbar(lead_counts, mae, err, 'o-', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(m, :), ...
            'DisplayName', strrep(method, '_', ' '));
    end
    
    xlabel('Number of Leads');
    ylabel('Mean Absolute K Error');
    title('K Error vs Lead Count');
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    
    saveas(gcf, fullfile(output_dir, 'k_accuracy_vs_leads.png'));
    close(gcf);
end

function plot_recovery_by_montage(T, output_dir)
    % Recovery metrics vs lead count
    
    montages = unique(T.montage_type);
    methods = unique(T.method);
    
    % Get lead counts
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    figure('Position', [100 100 1200 900]);
    colors = lines(length(methods));
    
    % Metrics to plot
    metrics = {'mean_recovery_matched', 'mean_recovery_padded', 'sensitivity', 'precision'};
    metric_names = {'Mean Recovery (Matched)', 'Mean Recovery (Padded)', 'Sensitivity', 'Precision'};
    
    for p = 1:4
        subplot(2, 2, p);
        hold on;
        
        for m = 1:length(methods)
            method = methods{m};
            vals = zeros(size(montages));
            errs = zeros(size(montages));
            
            for i = 1:length(montages)
                idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, method);
                if any(idx) && ismember(metrics{p}, T.Properties.VariableNames)
                    vals(i) = mean(T.(metrics{p})(idx));
                    errs(i) = std(T.(metrics{p})(idx)) / sqrt(sum(idx));
                end
            end
            
            errorbar(lead_counts, vals, errs, 'o-', ...
                'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(m, :), ...
                'DisplayName', strrep(method, '_', ' '));
        end
        
        xlabel('Number of Leads');
        ylabel(metric_names{p});
        title([metric_names{p} ' vs Lead Count']);
        legend('Location', 'best');
        grid on;
        set(gca, 'FontSize', 11);
    end
    
    saveas(gcf, fullfile(output_dir, 'recovery_metrics_vs_leads.png'));
    close(gcf);
end

function plot_heatmaps(T, output_dir)
    % Heatmaps showing method × montage performance
    
    montages = unique(T.montage_type);
    methods = unique(T.method);
    
    % Get lead counts for labels
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    % Create montage labels with lead counts
    montage_labels = cell(size(montages));
    for i = 1:length(montages)
        montage_labels{i} = sprintf('%s (%d)', montages{i}, lead_counts(i));
    end
    
    figure('Position', [100 100 1400 500]);
    
    % Heatmap 1: K Accuracy
    subplot(1, 3, 1);
    acc_matrix = zeros(length(methods), length(montages));
    for m = 1:length(methods)
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, methods{m});
            if any(idx)
                acc_matrix(m, i) = mean(T.K_correct(idx)) * 100;
            end
        end
    end
    imagesc(acc_matrix);
    colorbar;
    colormap('hot');
    xlabel('Montage');
    ylabel('Method');
    title('K Estimation Accuracy (%)');
    xticks(1:length(montages));
    xticklabels(montage_labels);
    xtickangle(45);
    yticks(1:length(methods));
    yticklabels(strrep(methods, '_', ' '));
    set(gca, 'FontSize', 10);
    
    % Add text annotations
    for m = 1:length(methods)
        for i = 1:length(montages)
            text(i, m, sprintf('%.1f', acc_matrix(m, i)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
        end
    end
    
    % Heatmap 2: Mean Recovery
    subplot(1, 3, 2);
    rec_matrix = zeros(length(methods), length(montages));
    for m = 1:length(methods)
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, methods{m});
            if any(idx)
                rec_matrix(m, i) = mean(T.mean_recovery_matched(idx));
            end
        end
    end
    imagesc(rec_matrix);
    colorbar;
    colormap('hot');
    xlabel('Montage');
    ylabel('Method');
    title('Mean Recovery (Matched)');
    xticks(1:length(montages));
    xticklabels(montage_labels);
    xtickangle(45);
    yticks(1:length(methods));
    yticklabels(strrep(methods, '_', ' '));
    set(gca, 'FontSize', 10);
    
    % Add text annotations
    for m = 1:length(methods)
        for i = 1:length(montages)
            text(i, m, sprintf('%.2f', rec_matrix(m, i)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
        end
    end
    
    % Heatmap 3: Mean K Error
    subplot(1, 3, 3);
    err_matrix = zeros(length(methods), length(montages));
    for m = 1:length(methods)
        for i = 1:length(montages)
            idx = strcmp(T.montage_type, montages{i}) & strcmp(T.method, methods{m});
            if any(idx)
                err_matrix(m, i) = mean(T.K_error(idx));
            end
        end
    end
    imagesc(err_matrix);
    colorbar;
    colormap('hot');
    xlabel('Montage');
    ylabel('Method');
    title('Mean Absolute K Error');
    xticks(1:length(montages));
    xticklabels(montage_labels);
    xtickangle(45);
    yticks(1:length(methods));
    yticklabels(strrep(methods, '_', ' '));
    set(gca, 'FontSize', 10);
    
    % Add text annotations
    for m = 1:length(methods)
        for i = 1:length(montages)
            text(i, m, sprintf('%.2f', err_matrix(m, i)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
        end
    end
    
    saveas(gcf, fullfile(output_dir, 'heatmaps_method_montage.png'));
    close(gcf);
end

function plot_snr_interactions(T, output_dir)
    % SNR × Montage interaction plots
    
    montages = unique(T.montage_type);
    snr_vals = unique(T.SNR_dB);
    
    % Get lead counts
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T.n_leads(idx);
        end
    end
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    figure('Position', [100 100 1400 500]);
    colors = lines(length(montages));
    
    % Plot 1: K Accuracy vs SNR
    subplot(1, 2, 1);
    hold on;
    
    for i = 1:length(montages)
        montage = montages{i};
        acc = zeros(size(snr_vals));
        
        for s = 1:length(snr_vals)
            idx = strcmp(T.montage_type, montage) & (T.SNR_dB == snr_vals(s));
            if any(idx)
                acc(s) = mean(T.K_correct(idx)) * 100;
            end
        end
        
        plot(snr_vals, acc, 'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
            'Color', colors(i, :), ...
            'DisplayName', sprintf('%s (%d leads)', montage, lead_counts(i)));
    end
    
    xlabel('SNR (dB)');
    ylabel('K Estimation Accuracy (%)');
    title('Montage × SNR Interaction: K Accuracy');
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    
    % Plot 2: Mean Recovery vs SNR
    subplot(1, 2, 2);
    hold on;
    
    for i = 1:length(montages)
        montage = montages{i};
        rec = zeros(size(snr_vals));
        
        for s = 1:length(snr_vals)
            idx = strcmp(T.montage_type, montage) & (T.SNR_dB == snr_vals(s));
            if any(idx)
                rec(s) = mean(T.mean_recovery_matched(idx));
            end
        end
        
        plot(snr_vals, rec, 'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
            'Color', colors(i, :), ...
            'DisplayName', sprintf('%s (%d leads)', montage, lead_counts(i)));
    end
    
    xlabel('SNR (dB)');
    ylabel('Mean Recovery (Matched)');
    title('Montage × SNR Interaction: Recovery');
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    
    saveas(gcf, fullfile(output_dir, 'snr_montage_interactions.png'));
    close(gcf);
end

function summary_stats = compute_summary_statistics(T)
    % Compute summary statistics by montage and method
    
    montages = unique(T.montage_type);
    methods = unique(T.method);
    
    rows = [];
    
    for i = 1:length(montages)
        montage = montages{i};
        
        % Get lead count
        idx = find(strcmp(T.montage_type, montage), 1);
        n_leads = T.n_leads(idx);
        
        for m = 1:length(methods)
            method = methods{m};
            
            idx = strcmp(T.montage_type, montage) & strcmp(T.method, method);
            
            if any(idx)
                row = struct();
                row.montage_type = montage;
                row.n_leads = n_leads;
                row.method = method;
                row.n_obs = sum(idx);
                row.k_accuracy_mean = mean(T.K_correct(idx));
                row.k_accuracy_std = std(T.K_correct(idx));
                row.k_error_mean = mean(T.K_error(idx));
                row.k_error_std = std(T.K_error(idx));
                row.recovery_matched_mean = mean(T.mean_recovery_matched(idx));
                row.recovery_matched_std = std(T.mean_recovery_matched(idx));
                row.sensitivity_mean = mean(T.sensitivity(idx));
                row.sensitivity_std = std(T.sensitivity(idx));
                row.precision_mean = mean(T.precision(idx));
                row.precision_std = std(T.precision(idx));
                
                rows = [rows; row]; %#ok<AGROW>
            end
        end
    end
    
    summary_stats = struct2table(rows);
end
