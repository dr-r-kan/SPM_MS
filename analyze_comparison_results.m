% analyze_comparison_results.m
% Comprehensive analysis pipeline for microstate comparison results.
% Usage:
%   analyze_comparison_results()
%   analyze_comparison_results(results_dir)
%   analyze_comparison_results(results_dir, 'n_boot', 5000, 'n_boot_lmm', 2000, 'n_folds', 5)
%
% Note: call with zero args is supported now.

function analyze_comparison_results(varargin)
    % ---------- Parse inputs ----------
    default_results_dir = 'outputs/simulations/results/';
    default_n_boot = 200;
    default_n_boot_lmm = 200;
    default_n_folds = 2;

    % ---------- Set up plot ----------
    set(0, 'DefaultFigureColor', 'white');
    set(0, 'DefaultTextColor', 'black');
    set(0, 'DefaultAxesColor', 'white');
    set(0, 'DefaultAxesXColor', 'black');
    set(0, 'DefaultAxesYColor', 'black');
    set(0, 'DefaultAxesZColor', 'black');
    set(0, 'DefaultAxesTickDir', 'out');
    set(0, 'DefaultLineLineWidth', 1.5);
    set(0, 'DefaultPatchLineWidth', 1.5);
    set(0, 'DefaultTextInterpreter', 'tex');
    set(0, 'DefaultLegendInterpreter', 'tex');

    p = inputParser;
    addOptional(p, 'results_dir', default_results_dir, @(x) ischar(x) || isstring(x));
    addParameter(p, 'n_boot', default_n_boot, @(x) isnumeric(x) && isscalar(x) && x>0);
    addParameter(p, 'n_boot_lmm', default_n_boot_lmm, @(x) isnumeric(x) && isscalar(x) && x>0);
    addParameter(p, 'n_folds', default_n_folds, @(x) isnumeric(x) && isscalar(x) && x>=2);
    parse(p, varargin{:});

    results_dir = char(p.Results.results_dir);
    n_boot = double(p.Results.n_boot);
    n_boot_lmm = double(p.Results.n_boot_lmm);
    n_folds = max(2, round(p.Results.n_folds));

    % Store in global config so subfunctions can use it
    global ANALYSIS_CONFIG;
    ANALYSIS_CONFIG.n_boot = n_boot;
    ANALYSIS_CONFIG.n_boot_lmm = n_boot_lmm;
    ANALYSIS_CONFIG.n_folds = n_folds;

    fprintf('\n========================================\n');
    fprintf('Bootstrapped Statistical Analysis\n');
    fprintf('========================================\n');
    fprintf('Results directory: %s\n', results_dir);
    fprintf('Bootstrap (aggregates): %d\n', ANALYSIS_CONFIG.n_boot);
    fprintf('Bootstrap (LMM): %d\n', ANALYSIS_CONFIG.n_boot_lmm);
    fprintf('CV folds: %d\n\n', ANALYSIS_CONFIG.n_folds);

    % ---------- Load data ----------
    csv_file = fullfile(results_dir, 'comparison_results.csv');
    if ~exist(csv_file, 'file')
        error('Results file not found: %s', csv_file);
    end

    T = readtable(csv_file);
    fprintf('✓ Loaded %d observations\n\n', height(T));

    % ---------- Backward compatibility: Add montage columns if missing ----------
    if ~ismember('montage_type', T.Properties.VariableNames)
        fprintf('⚠ Legacy results detected: montage_type column not found\n');
        fprintf('  Adding default montage_type = ''full''\n\n');
        T.montage_type = repmat({'full'}, height(T), 1);
    end

    if ~ismember('n_leads', T.Properties.VariableNames)
        fprintf('⚠ Legacy results detected: n_leads column not found\n');
        fprintf('  Adding default n_leads = 71 (full montage)\n\n');
        T.n_leads = repmat(71, height(T), 1);
    end

    if ~ismember('method', T.Properties.VariableNames)
        error('Table missing ''method'' column.');
    end

    raw_methods = cellstr(string(T.method));
    canonical_methods = cellfun(@canonicalize_method, raw_methods, 'UniformOutput', false);
    supported_method_mask = ismember(canonical_methods, {'koenig kmeans', 'spm vb'});
    if ~any(supported_method_mask)
        error(['No supported methods found in comparison results. ', ...
            'Expected Koenig k-means and/or SPM-VB rows.']);
    end
    if any(~supported_method_mask)
        dropped_methods = unique(canonical_methods(~supported_method_mask), 'stable');
        fprintf('Filtering out unsupported methods: %s\n', strjoin(dropped_methods, ', '));
        T = T(supported_method_mask, :);
        canonical_methods = canonical_methods(supported_method_mask);
    end
    T.method = canonical_methods(:);
    fprintf('Retained methods: %s\n\n', strjoin(unique(canonical_methods, 'stable'), ', '));

    % ---------- Derived columns ----------
    % K_error signed
    if ismember('K_estimated', T.Properties.VariableNames) && ismember('K_true', T.Properties.VariableNames)
        try
            T.K_error = double(T.K_estimated) - double(T.K_true);
        catch
            try
                T.K_error = cell2mat(T.K_estimated) - cell2mat(T.K_true);
            catch
                T.K_error = nan(height(T),1);
                warning('Could not compute K_error; filled with NaN.');
            end
        end
    else
        T.K_error = nan(height(T),1);
        warning('K_estimated or K_true not present; added NaN K_error.');
    end

    % Absolute K error
    try
        T.K_abs_error = abs(double(T.K_error));
    catch
        try
            T.K_abs_error = abs(cell2mat(T.K_error));
        catch
            T.K_abs_error = nan(height(T),1);
            warning('Could not compute K_abs_error; filled with NaN.');
        end
    end

    % Mean recovery matched (map-correlation metric)
    if ~ismember('mean_recovery_matched', T.Properties.VariableNames)
        reccols = startsWith(T.Properties.VariableNames, 'recovery_');
        if any(reccols)
            recnames = T.Properties.VariableNames(reccols);
            recmat = nan(height(T), numel(recnames));
            for i = 1:numel(recnames)
                col = T.(recnames{i});
                if iscell(col)
                    for r = 1:height(T)
                        v = col{r};
                        if isnumeric(v) && isscalar(v)
                            recmat(r,i) = v;
                        else
                            recmat(r,i) = NaN;
                        end
                    end
                else
                    recmat(:,i) = double(col);
                end
            end
            T.mean_recovery_matched = mean(recmat, 2, 'omitnan');
            fprintf('Computed mean_recovery_matched from recovery_* columns.\n');
        elseif ismember('recovery_corr', T.Properties.VariableNames)
            tmp = T.recovery_corr;
            mvals = nan(height(T),1);
            for r = 1:height(T)
                v = tmp{r};
                if isnumeric(v)
                    mvals(r) = mean(v);
                elseif ischar(v) || isstring(v)
                    try
                        vv = str2num(v); %#ok<ST2NM>
                        mvals(r) = mean(vv);
                    catch
                        mvals(r) = NaN;
                    end
                else
                    mvals(r) = NaN;
                end
            end
            T.mean_recovery_matched = mvals;
            fprintf('Computed mean_recovery_matched from recovery_corr.\n');
        else
            T.mean_recovery_matched = nan(height(T),1);
            warning('No recovery columns found; mean_recovery_matched filled with NaN.');
        end
    else
        if iscell(T.mean_recovery_matched), T.mean_recovery_matched = cell2mat(T.mean_recovery_matched); end
    end

    % Canonicalize criterion labels to avoid duplicates
    if ismember('criterion', T.Properties.VariableNames)
        critcell = cellstr(string(T.criterion));
        T.criterion_clean = cellfun(@canonicalize_criterion, critcell, 'UniformOutput', false);
    else
        error('Table missing ''criterion'' column.');
    end

    % Ensure subject exists for random effect
    if ~ismember('subject', T.Properties.VariableNames)
        if ismember('eeg_idx', T.Properties.VariableNames)
            T.subject = arrayfun(@(x) sprintf('eeg_%d', x), T.eeg_idx, 'UniformOutput', false);
        elseif ismember('fit_id', T.Properties.VariableNames)
            T.subject = arrayfun(@(x) sprintf('fit_%d', x), T.fit_id, 'UniformOutput', false);
        else
            T.subject = arrayfun(@(i) sprintf('row_%d', i), (1:height(T))', 'UniformOutput', false);
            warning('No subject/eeg_idx/fit_id column; created synthetic subject per row.');
        end
    end

    % Convert to categorical where appropriate
    try, T.method = categorical(T.method); catch, T.method = categorical(cellstr(string(T.method))); end
    T.criterion_clean = categorical(T.criterion_clean);
    T.subject = categorical(T.subject);
    T.method_criterion_combo = categorical(strcat(string(T.method), " | ", string(T.criterion_clean)));

    % ---------- Output setup ----------
    plots_dir = fullfile(fileparts(results_dir), 'analysis_plots');
    if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end
    stats_file = fullfile(plots_dir, 'bootstrap_statistics.txt');
    fid = fopen(stats_file, 'w');

    fprintf(fid, '========================================\n');
    fprintf(fid, 'BOOTSTRAPPED STATISTICAL ANALYSIS\n');
    fprintf(fid, 'Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'User: %s\n', getenv('USERNAME'));
    fprintf(fid, 'Observations: %d\n', height(T));
    fprintf(fid, 'Bootstrap (aggregates): %d\n', ANALYSIS_CONFIG.n_boot);
    fprintf(fid, 'Bootstrap (LMM): %d\n', ANALYSIS_CONFIG.n_boot_lmm);
    fprintf(fid, 'CV folds: %d\n', ANALYSIS_CONFIG.n_folds);
    fprintf(fid, '========================================\n\n');

    design_info = summarize_method_criterion_design(T, fid);
    T_method_effects = subset_to_shared_criteria(T, design_info.shared_criteria);
    T_criterion_effects = subset_to_full_support_methods(T, design_info.full_support_methods);

    % ---------- Outcomes ----------
    % Note: runtime_s is excluded from main comparison plots and handled separately
    outcomes = {'K_correct', 'f1_score', 'sensitivity', 'precision', 'mean_recovery_matched', 'K_abs_error'};
    outcome_labels = {'K_{true} Selection Accuracy', 'F1 Score', 'Sensitivity', 'Precision', 'Mean Matched Correlation', 'Absolute K Error'};
    
    % Full outcomes list (including runtime) for LMM analysis
    outcomes_full = {'K_correct', 'f1_score', 'sensitivity', 'precision', 'runtime_s', 'mean_recovery_matched', 'K_abs_error'};
    outcome_labels_full = {'K_{true} Selection Accuracy', 'F1 Score', 'Sensitivity', 'Precision', 'Runtime (s)', 'Mean Matched Correlation', 'Absolute K Error'};

    % ---------- Analyses ----------
    fprintf('1) METHOD effects\n');    method_results = analyze_factor_effects_with_ci(T_method_effects, 'method', outcomes_full, fid);
    fprintf('2) CRITERION effects\n'); criterion_results = analyze_factor_effects_with_ci_using_clean(T_criterion_effects, 'criterion_clean', outcomes_full, fid);
    fprintf('3) SNR effects\n'); snr_results = analyze_snr_effects(T, outcomes_full, fid);
    fprintf('4) MONTAGE effects\n'); montage_results = analyze_factor_effects_with_ci(T, 'montage_type', outcomes_full, fid);
    fprintf('5) Interaction METHOD × CRITERION\n'); interaction_results = analyze_interaction(T, outcomes_full, fid);

    % Cross-validation
    fprintf('6) Cross-validation (adaptive)\n'); cv_results = analyze_cross_validation_adaptive(T, outcomes_full, ANALYSIS_CONFIG.n_folds, fid);

    % Bootstrapped LMMs (cluster bootstrap by subject) including interaction
    fprintf('7) Bootstrapped Linear Mixed Models (LMM)\n');
    fprintf(fid, '\n========================================\nLMM Analysis (cluster bootstrap by subject)\nRandom intercept: subject\nFixed effects: method_criterion_combo + SNR_dB\n========================================\n\n');
    for i = 1:length(outcomes_full)
        out = outcomes_full{i};
        fprintf('  LMM for %s\n', out);
        try
            analyze_lmm_bootstrap(T, out, ANALYSIS_CONFIG.n_boot_lmm, fid);
        catch ME
            fprintf('LMM failed for %s: %s\n', out, ME.message);
            fprintf(fid, 'LMM failed for %s: %s\n', out, ME.message);
        end
    end

    vb_covariance_outcomes = {'f1_score', 'sensitivity', 'precision'};
    vb_covariance_labels = {'F1 Score', 'Sensitivity', 'Precision'};
    fprintf('8) VB covariance relationships\n');
    vb_covariance_results = analyze_vb_covariance_relationships(T, vb_covariance_outcomes, vb_covariance_labels, fid);

    fclose(fid);
    fprintf('Analysis saved to %s and plots under %s\n', stats_file, plots_dir);

    % ---------- Plots (dynamic grids) ----------
    fprintf('Generating plots...\n');
    % Use outcomes without runtime for these plots
    create_boxplot_comparison(T, outcomes, outcome_labels, plots_dir);
    create_method_effects_plot_with_ci(T_method_effects, outcomes, outcome_labels, method_results, plots_dir);
    create_criterion_effects_plot_with_ci(T_criterion_effects, outcomes, outcome_labels, criterion_results, plots_dir);
    create_snr_effects_plot(T, outcomes, outcome_labels, plots_dir);
    create_interaction_plot(T, outcomes, outcome_labels, plots_dir);
    create_cross_validation_plot(cv_results, outcomes, outcome_labels, plots_dir);
    
    % Separate runtime plot (SNR effects only)
    create_runtime_snr_plot(T, plots_dir);
    
    create_avg_k_error_plot(T, plots_dir);
    create_abs_k_error_plot(T, plots_dir);
    create_backfit_confusion_comparison_plots(T, results_dir, plots_dir);
    create_simulated_backfit_confusion_summary(T, results_dir, plots_dir);
    
    % Method-Criterion comparison boxplots with significance
    create_method_criterion_boxplots(T, plots_dir);
    create_vb_covariance_relationship_plots(T, vb_covariance_outcomes, vb_covariance_labels, vb_covariance_results, plots_dir);
    create_vb_covariance_kcorrect_barplots(vb_covariance_results, plots_dir);
    create_vb_covariance_quintile_topoplots(T, vb_covariance_results, results_dir, plots_dir);

    % ---------- Montage-Criterion Comparison (if montage data available) ----------
    montages = unique(T.montage_type);
    if length(montages) > 1
        fprintf('Generating criterion-montage comparison boxplots...\n');
        create_criterion_montage_boxplots(T, plots_dir);
        fprintf('Generating integrated montage robustness analysis...\n');
        analyze_montage_robustness(csv_file, 'output_dir', fullfile(plots_dir, 'montage_robustness'));
    end

    fprintf('All plots saved in %s\n', plots_dir);
end

% -----------------------
% Subfunctions follow (unchanged logic except use global ANALYSIS_CONFIG where needed)
% -----------------------

function levels = get_factor_levels(T, factor_name)
    if ~isvarname(factor_name) && ~ismember(factor_name, T.Properties.VariableNames)
        levels = {}; return;
    end
    col = T.(factor_name);
    if iscell(col)
        col_cell = cellfun(@(x) char(string(x)), col, 'UniformOutput', false);
    elseif iscategorical(col)
        col_cell = cellstr(col);
    elseif isstring(col)
        col_cell = cellstr(col);
    elseif isnumeric(col)
        col_cell = cellstr(string(col));
    else
        col_cell = cellstr(string(col));
    end
    levels = unique(col_cell, 'stable');
end

function design_info = summarize_method_criterion_design(T, fid)
    methods = get_factor_levels(T, 'method');
    criteria = get_factor_levels(T, 'criterion_clean');
    observed = false(numel(methods), numel(criteria));
    counts = zeros(numel(methods), numel(criteria));
    for m = 1:numel(methods)
        for c = 1:numel(criteria)
            mask = strcmp(cellstr(string(T.method)), methods{m}) & ...
                strcmp(cellstr(string(T.criterion_clean)), criteria{c});
            counts(m, c) = nnz(mask);
            observed(m, c) = counts(m, c) > 0;
        end
    end
    shared_criteria = criteria(all(observed, 1));
    full_support_methods = methods(all(observed, 2));
    design_info = struct('methods', {methods}, 'criteria', {criteria}, ...
        'observed', observed, 'counts', counts, ...
        'shared_criteria', {shared_criteria}, ...
        'full_support_methods', {full_support_methods});

    fprintf(fid, 'Observed method-criterion design:\n');
    for m = 1:numel(methods)
        observed_criteria = criteria(observed(m, :));
        if isempty(observed_criteria)
            observed_label = '<none>';
        else
            observed_label = strjoin(observed_criteria, ', ');
        end
        fprintf(fid, '  %s: %s\n', methods{m}, observed_label);
    end
    if ~isempty(shared_criteria)
        fprintf(fid, 'Shared criteria across all methods: %s\n', strjoin(shared_criteria, ', '));
    else
        fprintf(fid, 'Shared criteria across all methods: <none>\n');
    end
    if ~isempty(full_support_methods)
        fprintf(fid, 'Methods with full criterion support: %s\n\n', strjoin(full_support_methods, ', '));
    else
        fprintf(fid, 'Methods with full criterion support: <none>\n\n');
    end
end

function T_sub = subset_to_shared_criteria(T, shared_criteria)
    if isempty(shared_criteria)
        T_sub = T;
        return;
    end
    mask = ismember(cellstr(string(T.criterion_clean)), shared_criteria);
    T_sub = T(mask, :);
end

function T_sub = subset_to_full_support_methods(T, full_support_methods)
    if isempty(full_support_methods)
        T_sub = T;
        return;
    end
    mask = ismember(cellstr(string(T.method)), full_support_methods);
    T_sub = T(mask, :);
end

function results = analyze_factor_effects_with_ci(T, factor_name, outcomes, fid)
    global ANALYSIS_CONFIG;
    results = struct();
    factor_levels = get_factor_levels(T, factor_name);
    n_boot = ANALYSIS_CONFIG.n_boot;
    fprintf(fid, 'Factor: %s\nLevels: %s\n\n', factor_name, strjoin(factor_levels, ', '));
    for i = 1:length(outcomes)
        outcome = outcomes{i};
        fprintf(fid, '--- %s ---\n', outcome);
        data_by_level = cell(numel(factor_levels),1);
        for j = 1:numel(factor_levels)
            lvl = factor_levels{j};
            colvals = cellstr(string(T.(factor_name)));
            mask = strcmp(colvals, lvl);
            vals = T.(outcome)(mask);
            vals = vals(~isnan(vals));
            data_by_level{j} = vals;
        end
        obs_means = cellfun(@(x) mean(x, 'omitnan'), data_by_level);
        boot_means = nan(n_boot, numel(factor_levels));
        for b = 1:n_boot
            for j = 1:numel(factor_levels)
                d = data_by_level{j};
                if ~isempty(d)
                    samp = datasample(d, numel(d));
                    boot_means(b,j) = mean(samp, 'omitnan');
                end
            end
        end
        ci_lower = prctile(boot_means, 2.5, 1);
        ci_upper = prctile(boot_means, 97.5, 1);
        fprintf(fid, 'Observed Means (95%% CI):\n');
        for j = 1:numel(factor_levels)
            fprintf(fid, '  %s: %.4f [%.4f, %.4f]\n', factor_levels{j}, obs_means(j), ci_lower(j), ci_upper(j));
        end
        all_data = []; groups = [];
        for j = 1:numel(factor_levels)
            all_data = [all_data; data_by_level{j}];
            groups = [groups; repmat(j, numel(data_by_level{j}), 1)];
        end
        if numel(all_data) > 1
            try
                [p_kw, ~, stats_kw] = kruskalwallis(all_data, groups, 'off');
                if isfield(stats_kw, 'chi2stat'), hstat = stats_kw.chi2stat; else hstat = chi2inv(1 - p_kw, numel(factor_levels)-1); end
                fprintf(fid, 'Kruskal-Wallis: H=%.4f p=%.4e\n', hstat, p_kw);
            catch
                fprintf(fid, 'Kruskal-Wallis: failed\n');
            end
        end
        results.(outcome) = struct('means', obs_means, 'ci_lower', ci_lower, 'ci_upper', ci_upper, 'level_names', {factor_levels});
        fprintf(fid, '\n');
    end
end

function posthoc = posthoc_pairwise(T, factor_name, outcomes, fid)
    posthoc = struct();
    factor_levels = get_factor_levels(T, factor_name);
    n_levels = numel(factor_levels);
    if n_levels < 2
        fprintf(fid, 'Not enough levels for post-hoc for %s\n', factor_name); return;
    end
    n_comparisons = nchoosek(n_levels,2);
    alpha_bonf = 0.05 / max(1,n_comparisons);
    fprintf(fid, 'Bonferroni alpha: %.6f (%d comparisons)\n', alpha_bonf, n_comparisons);
    for i = 1:length(outcomes)
        outcome = outcomes{i};
        fprintf(fid, '--- POSTHOC %s ---\n', outcome);
        data_by_level = cell(n_levels,1);
        for j = 1:n_levels
            lvl = factor_levels{j};
            colvals = cellstr(string(T.(factor_name)));
            mask = strcmp(colvals, lvl);
            vals = T.(outcome)(mask); vals = vals(~isnan(vals));
            data_by_level{j} = vals;
        end
        comps = [];
        for j1 = 1:n_levels-1
            for j2 = j1+1:n_levels
                d1 = data_by_level{j1}; d2 = data_by_level{j2};
                if isempty(d1) || isempty(d2), continue; end
                [p, ~, stats] = ranksum(d1, d2);
                if isstruct(stats) && isfield(stats, 'zval'), z = stats.zval; else z = NaN; end
                signif = p < alpha_bonf;
                fprintf(fid, '%s vs %s: z=%.3f p=%.4f %s\n', factor_levels{j1}, factor_levels{j2}, z, p, ifthenelse(signif,'***',''));
                comps = [comps; struct('level1', factor_levels{j1}, 'level2', factor_levels{j2}, 'p_value', p, 'significant', signif)];
            end
        end
        posthoc.(outcome) = comps;
        fprintf(fid, '\n');
    end
end

function results = analyze_factor_effects_with_ci_using_clean(T, factor_name_clean, outcomes, fid)
    results = analyze_factor_effects_with_ci(T, factor_name_clean, outcomes, fid);
end

function posthoc = posthoc_pairwise_using_clean(T, factor_name_clean, outcomes, fid)
    posthoc = posthoc_pairwise(T, factor_name_clean, outcomes, fid);
end

function results = analyze_snr_effects(T, outcomes, fid)
    global ANALYSIS_CONFIG;
    results = struct();
    snr_vals = unique(T.SNR_dB);
    snr_vals = sort(snr_vals);
    n_boot = ANALYSIS_CONFIG.n_boot;
    fprintf(fid, 'SNR levels: %s\n', mat2str(snr_vals));
    for i = 1:length(outcomes)
        outcome = outcomes{i};
        fprintf(fid, '--- %s ---\n', outcome);
        means = nan(numel(snr_vals),1); cil = nan(size(means)); cih = nan(size(means));
        for j = 1:numel(snr_vals)
            snr = snr_vals(j);
            mask = T.SNR_dB == snr;
            data = T.(outcome)(mask); data = data(~isnan(data));
            means(j) = mean(data, 'omitnan');
            boot_means = nan(n_boot,1);
            for b = 1:n_boot
                if isempty(data), boot_means(b)=NaN; else boot_means(b)=mean(datasample(data,numel(data)),'omitnan'); end
            end
            cil(j) = prctile(boot_means,2.5); cih(j) = prctile(boot_means,97.5);
        end
        fprintf(fid, 'Means by SNR (95%% CI):\n'); for j=1:numel(snr_vals), fprintf(fid, ' SNR %+g dB: %.4f [%.4f, %.4f]\n', snr_vals(j), means(j), cil(j), cih(j)); end
        results.(outcome) = struct('snr_vals', snr_vals, 'means', means, 'ci_lower', cil, 'ci_upper', cih);
        fprintf(fid, '\n');
    end
end

function interaction_results = analyze_interaction(T, outcomes, fid)
    interaction_results = struct();
    methods = get_factor_levels(T, 'method');
    criteria = get_factor_levels(T, 'criterion_clean');
    fprintf(fid, 'Interaction METHOD x CRITERION\nMethods: %s\nCriteria: %s\n', strjoin(methods, ', '), strjoin(criteria, ', '));
    for i = 1:length(outcomes)
        outcome = outcomes{i};
        tablevals = nan(numel(methods), numel(criteria));
        for mi = 1:numel(methods)
            for ci = 1:numel(criteria)
                m = methods{mi}; c = criteria{ci};
                mask = strcmp(cellstr(string(T.method)), m) & strcmp(cellstr(string(T.criterion_clean)), c);
                data = T.(outcome)(mask); data = data(~isnan(data));
                if ~isempty(data), tablevals(mi,ci) = mean(data, 'omitnan'); end
            end
        end
        interaction_results.(outcome) = struct('table', tablevals, 'methods', {methods}, 'criteria', {criteria});
        fprintf(fid, 'Interaction table for %s written.\n', outcome);
    end
end

function cv_results = analyze_cross_validation_adaptive(T, outcomes, n_folds, fid)
    cv_results = struct();
    n_obs = height(T);
    fold_size = floor(n_obs / n_folds);
    fprintf(fid, 'Cross-validation %d folds, approx fold size %d\n', n_folds, fold_size);
    rng(0);
    fold_assignment = repmat(1:n_folds, 1, ceil(n_obs/n_folds));
    fold_assignment = fold_assignment(randperm(n_obs));
    fold_assignment = fold_assignment(1:n_obs)';
    for i = 1:length(outcomes)
        outcome = outcomes{i};
        fold_errors = [];
        for f = 1:n_folds
            test_mask = fold_assignment == f;
            test_data = T.(outcome)(test_mask); test_data = test_data(~isnan(test_data));
            train_data = T.(outcome)(~test_mask); train_data = train_data(~isnan(train_data));
            if ~isempty(test_data) && ~isempty(train_data)
                err = abs(mean(test_data, 'omitnan') - mean(train_data, 'omitnan'));
                fold_errors(end+1) = err;
                fprintf(fid, 'Fold %d: train=%.4f test=%.4f err=%.4f\n', f, mean(train_data,'omitnan'), mean(test_data,'omitnan'), err);
            end
        end
        if ~isempty(fold_errors), fprintf(fid, 'CV error mean=%.4f std=%.4f\n', mean(fold_errors), std(fold_errors)); end
        cv_results.(outcome) = struct('fold_errors', fold_errors, 'n_folds', n_folds);
    end
end

function analyze_lmm_bootstrap(T, outcome_var, n_boot, fid)
    if nargin < 3 || isempty(n_boot), n_boot = 1000; end
    if ~exist('fitlme', 'file')
        fprintf(fid, 'fitlme not available; skipping LMM for %s\n', outcome_var); return;
    end
    mask_valid = ~isnan(T.(outcome_var));
    Tsub = T(mask_valid,:);
    subjects = unique(Tsub.subject);
    n_subj = numel(subjects);
    if n_subj < 2
        fprintf(fid, 'Not enough subjects (%d) for LMM on %s; skipping\n', n_subj, outcome_var); return;
    end
    if ~isnumeric(Tsub.SNR_dB), Tsub.SNR_dB = double(Tsub.SNR_dB); end
    formula = sprintf('%s ~ 1 + method_criterion_combo + SNR_dB + (1|subject)', outcome_var);
    % Treat near-zero residual variance as degenerate to cut down on fitlme warnings
    base_var = var(Tsub.(outcome_var), 'omitnan');
    resvar_floor = max(1e-8, 1e-4 * base_var);
    warn_state = warning('off', 'stats:LinearMixedModel:PerfectFit');
    cleanup_warn = onCleanup(@() warning(warn_state));
    lme_opts = statset('Display','off','MaxIter',200,'TolFun',1e-4,'TolX',1e-4);
    try
        lme0 = fitlme(Tsub, formula, 'FitMethod', 'REML', 'Optimizer', 'quasinewton', 'Options', lme_opts, 'StartMethod', 'random');
    catch ME
        fprintf(fid, 'Original LME fit failed for %s: %s\n', outcome_var, ME.message); return;
    end
    % Normality check on standardized residuals; skip bootstrap if residuals look normal
    res = residuals(lme0, 'ResidualType', 'Pearson');
    normal_p = NaN;
    try
        [~, normal_p] = lillietest(res); % Lilliefors if available
    catch
        try
            [~, normal_p] = kstest(res);
        catch
            normal_p = NaN;
        end
    end
    if ~isnan(normal_p) && normal_p > 0.05
        fprintf(fid, 'Residuals appear normal (p=%.3f); skipping bootstrap for %s\n\n', normal_p, outcome_var);
        fprintf('  Residuals normal (p=%.3f); skipped bootstrap for %s\n', normal_p, outcome_var);
        return;
    end
    if lme0.MSE < resvar_floor
        fprintf(fid, 'Skipping LMM for %s: residual variance ~0 (possible perfect fit)\n', outcome_var);
        return;
    end
    coefNames = lme0.Coefficients.Name;
    obsEst = fixedEffects(lme0);
    obsSE = lme0.Coefficients.SE;
    obsP = lme0.Coefficients.pValue;
    nCoefs = numel(obsEst);
    bootEst = nan(n_boot, nCoefs);
    n_success = 0;
    fprintf(fid, 'Bootstrapping LMM for %s (%d iterations), subjects=%d, observations=%d\n', outcome_var, n_boot, n_subj, height(Tsub));
    t_start = tic;
    update_every = max(1, floor(n_boot / 20)); % ~5% updates to console
    fprintf('    LMM bootstrap %s: starting (update every %d iters)\n', outcome_var, update_every);
    for b = 1:n_boot
        sampled = datasample(subjects, n_subj);
        Tb = Tsub([],:);
        for k = 1:n_subj
            Tb = [Tb; Tsub(Tsub.subject == sampled(k), :)];
        end
        if var(Tb.(outcome_var), 'omitnan') < resvar_floor
            continue;
        end
        try
            iter_tic = tic;
            lme_b = fitlme(Tb, formula, 'FitMethod', 'REML', 'Optimizer', 'quasinewton', 'Options', lme_opts, 'StartMethod', 'random');
            if lme_b.MSE < resvar_floor
                continue;
            end
            fe_b = fixedEffects(lme_b);
            if numel(fe_b) == nCoefs
                bootEst(b,:) = fe_b';
            else
                names_b = lme_b.Coefficients.Name;
                estimates_b = lme_b.Coefficients.Estimate;
                mapped = nan(nCoefs,1);
                for ii = 1:nCoefs
                    idx = find(strcmp(names_b, coefNames{ii}), 1);
                    if ~isempty(idx), mapped(ii) = estimates_b(idx); end
                end
                bootEst(b,:) = mapped';
            end
            n_success = n_success + 1;
        catch
            bootEst(b,:) = nan(1,nCoefs);
        end
        if mod(b, update_every) == 0 || b == n_boot
            elapsed = toc(t_start);
            eta = (elapsed / b) * (n_boot - b);
            fprintf('    LMM bootstrap %s: %4d/%4d (%.1f%%%%) elapsed %.1fs ETA %.1fs\n', outcome_var, b, n_boot, 100*b/n_boot, elapsed, eta);
        end
    end
    ciL = prctile(bootEst, 2.5, 1);
    ciH = prctile(bootEst, 97.5, 1);
    bootMean = nanmean(bootEst, 1);
    bootStd = nanstd(bootEst, [], 1);
    bootP = nan(1,nCoefs);
    for ii = 1:nCoefs
        ests = bootEst(:,ii); ests = ests(~isnan(ests));
        if isempty(ests), bootP(ii) = NaN; else prop_pos = mean(ests>=0); prop_neg = mean(ests<=0); bootP(ii) = 2*min(prop_pos, prop_neg); end
    end
    fprintf(fid, '\nObserved fixed-effects for %s:\n', outcome_var);
    fprintf(fid, '%-35s %10s %10s %10s\n', 'Name', 'ObsEst', 'ObsSE', 'ObsP');
    for ii = 1:nCoefs
        fprintf(fid, '%-35s %10.4f %10.4f %10.4g\n', coefNames{ii}, obsEst(ii), obsSE(ii), obsP(ii));
    end
    fprintf(fid, '\nBootstrap summaries (n=%d):\n', n_boot);
    fprintf(fid, '%-35s %10s %10s %10s %10s\n', 'Name', 'BootMean', 'BootStd', 'BootP(emp)', 'CI');
    for ii = 1:nCoefs
        fprintf(fid, '%-35s %10.4f %10.4f %10.4f [%6.4f, %6.4f]\n', coefNames{ii}, bootMean(ii), bootStd(ii), bootP(ii), ciL(ii), ciH(ii));
    end
    fprintf(fid, 'Successful bootstrap fits: %d / %d\n\n', n_success, n_boot);
    fprintf('    LMM bootstrap %s done. Successful fits: %d/%d. Elapsed %.1fs\n', outcome_var, n_success, n_boot, toc(t_start));
end

function results = analyze_vb_covariance_relationships(T, outcomes, outcome_labels, fid)
    global ANALYSIS_CONFIG;
    results = struct();
    cov_specs = get_available_covariance_specs(T);
    if isempty(cov_specs)
        fprintf(fid, '\n========================================\n');
        fprintf(fid, 'VB Covariance Relationships\n');
        fprintf(fid, 'Skipped: no covariance summary columns found in comparison_results.csv\n');
        fprintf(fid, 'Expected one or more of: selected_spm_cov_trace_mean, selected_spm_cov_trace_median, selected_spm_cov_logdet_mean\n');
        fprintf(fid, '========================================\n\n');
        return;
    end

    vb_mask = is_vb_method(T.method);
    T_vb = T(vb_mask, :);
    if height(T_vb) == 0
        fprintf(fid, '\n========================================\n');
        fprintf(fid, 'VB Covariance Relationships\n');
        fprintf(fid, 'Skipped: no VB rows found in method column.\n');
        fprintf(fid, '========================================\n\n');
        return;
    end

    n_boot = ANALYSIS_CONFIG.n_boot;
    results.covariates = cov_specs;
    results.n_rows_vb = height(T_vb);
    results.outcomes = outcomes;
    results.outcome_labels = outcome_labels;

    fprintf(fid, '\n========================================\n');
    fprintf(fid, 'VB Covariance Relationships\n');
    fprintf(fid, 'Method subset: VB only\n');
    fprintf(fid, 'Rows in VB subset: %d\n', height(T_vb));
    fprintf(fid, 'Bootstrap iterations for correlation CI: %d\n', n_boot);
    fprintf(fid, '========================================\n\n');

    for i = 1:numel(cov_specs)
        spec = cov_specs(i);
        covariance_vals = coerce_numeric_column(T_vb.(spec.name));
        metric_result = struct();
        metric_result.name = spec.name;
        metric_result.label = spec.label;
        metric_result.n_nonmissing = sum(~isnan(covariance_vals));
        fprintf(fid, 'Covariance summary: %s (%s)\n', spec.label, spec.name);
        fprintf(fid, '  Non-missing VB rows: %d / %d\n', metric_result.n_nonmissing, height(T_vb));

        for j = 1:numel(outcomes)
            outcome = outcomes{j};
            outcome_vals = coerce_numeric_column(T_vb.(outcome));
            valid_mask = ~isnan(covariance_vals) & ~isnan(outcome_vals);
            xv = outcome_vals(valid_mask);
            yv = covariance_vals(valid_mask);

            outcome_result = struct( ...
                'label', outcome_labels{j}, ...
                'n', numel(xv), ...
                'spearman_rho', NaN, ...
                'spearman_p', NaN, ...
                'ci_lower', NaN, ...
                'ci_upper', NaN, ...
                'pearson_r', NaN, ...
                'pearson_p', NaN, ...
                'slope', NaN, ...
                'intercept', NaN);

            if numel(xv) >= 3 && numel(unique(xv)) > 1 && numel(unique(yv)) > 1
                [rho_s, p_s] = corr(xv, yv, 'Type', 'Spearman', 'Rows', 'complete');
                [rho_p, p_p] = corr(xv, yv, 'Type', 'Pearson', 'Rows', 'complete');
                pfit = polyfit(xv, yv, 1);
                boot_rho = nan(n_boot, 1);
                for b = 1:n_boot
                    idx = randi(numel(xv), numel(xv), 1);
                    xb = xv(idx);
                    yb = yv(idx);
                    if numel(unique(xb)) > 1 && numel(unique(yb)) > 1
                        boot_rho(b) = corr(xb, yb, 'Type', 'Spearman', 'Rows', 'complete');
                    end
                end
                outcome_result.spearman_rho = rho_s;
                outcome_result.spearman_p = p_s;
                outcome_result.ci_lower = prctile(boot_rho, 2.5);
                outcome_result.ci_upper = prctile(boot_rho, 97.5);
                outcome_result.pearson_r = rho_p;
                outcome_result.pearson_p = p_p;
                outcome_result.slope = pfit(1);
                outcome_result.intercept = pfit(2);
            end

            metric_result.(outcome) = outcome_result;
            fprintf(fid, '  %-24s n=%4d  Spearman rho=%7.4f [%7.4f, %7.4f] p=%8.4g  Pearson r=%7.4f p=%8.4g\n', ...
                outcome, outcome_result.n, outcome_result.spearman_rho, outcome_result.ci_lower, ...
                outcome_result.ci_upper, outcome_result.spearman_p, outcome_result.pearson_r, outcome_result.pearson_p);
        end

        metric_result.k_correct_groups = compute_vb_covariance_kcorrect_groups(T_vb, covariance_vals, ANALYSIS_CONFIG.n_boot);
        group_stats = metric_result.k_correct_groups;
        fprintf(fid, '  K-correct grouping:\n');
        fprintf(fid, '    K correct     n=%4d  mean=%9.4f [%9.4f, %9.4f]\n', ...
            group_stats.n_correct, group_stats.mean_correct, group_stats.ci_correct(1), group_stats.ci_correct(2));
        fprintf(fid, '    K incorrect   n=%4d  mean=%9.4f [%9.4f, %9.4f]\n', ...
            group_stats.n_incorrect, group_stats.mean_incorrect, group_stats.ci_incorrect(1), group_stats.ci_incorrect(2));
        fprintf(fid, '    Mean diff (correct - incorrect) = %9.4f [%9.4f, %9.4f]\n', ...
            group_stats.mean_difference, group_stats.ci_difference(1), group_stats.ci_difference(2));
        fprintf(fid, '\n');
        results.(spec.name) = metric_result;
    end
end

function group_stats = compute_vb_covariance_kcorrect_groups(T_vb, covariance_vals, n_boot)
    if ~ismember('K_correct', T_vb.Properties.VariableNames)
        group_stats = default_kcorrect_group_stats();
        return;
    end

    k_correct_vals = coerce_numeric_column(T_vb.K_correct);
    correct_mask = ~isnan(covariance_vals) & (k_correct_vals >= 0.5);
    incorrect_mask = ~isnan(covariance_vals) & (k_correct_vals < 0.5);
    correct_vals = covariance_vals(correct_mask);
    incorrect_vals = covariance_vals(incorrect_mask);

    group_stats = default_kcorrect_group_stats();
    group_stats.n_correct = numel(correct_vals);
    group_stats.n_incorrect = numel(incorrect_vals);
    group_stats.mean_correct = mean(correct_vals, 'omitnan');
    group_stats.mean_incorrect = mean(incorrect_vals, 'omitnan');
    group_stats.ci_correct = bootstrap_mean_ci(correct_vals, n_boot);
    group_stats.ci_incorrect = bootstrap_mean_ci(incorrect_vals, n_boot);

    if ~isempty(correct_vals) && ~isempty(incorrect_vals)
        group_stats.mean_difference = group_stats.mean_correct - group_stats.mean_incorrect;
        boot_diff = nan(n_boot, 1);
        for b = 1:n_boot
            sample_correct = datasample(correct_vals, numel(correct_vals));
            sample_incorrect = datasample(incorrect_vals, numel(incorrect_vals));
            boot_diff(b) = mean(sample_correct, 'omitnan') - mean(sample_incorrect, 'omitnan');
        end
        group_stats.ci_difference = [prctile(boot_diff, 2.5), prctile(boot_diff, 97.5)];
    end
end

function group_stats = default_kcorrect_group_stats()
    group_stats = struct( ...
        'n_correct', 0, ...
        'n_incorrect', 0, ...
        'mean_correct', NaN, ...
        'mean_incorrect', NaN, ...
        'ci_correct', [NaN NaN], ...
        'ci_incorrect', [NaN NaN], ...
        'mean_difference', NaN, ...
        'ci_difference', [NaN NaN]);
end

function ci = bootstrap_mean_ci(values, n_boot)
    if isempty(values)
        ci = [NaN NaN];
        return;
    end

    boot_means = nan(n_boot, 1);
    for b = 1:n_boot
        sample = datasample(values, numel(values));
        boot_means(b) = mean(sample, 'omitnan');
    end
    ci = [prctile(boot_means, 2.5), prctile(boot_means, 97.5)];
end

function cov_specs = get_available_covariance_specs(T)
    candidates = { ...
        'selected_spm_cov_trace_mean', 'Mean covariance trace'; ...
        'selected_spm_cov_trace_median', 'Median covariance trace'; ...
        'selected_spm_cov_logdet_mean', 'Mean covariance logdet'};
    cov_specs = struct('name', {}, 'label', {});
    for i = 1:size(candidates, 1)
        if ismember(candidates{i,1}, T.Properties.VariableNames)
            cov_specs(end+1) = struct('name', candidates{i,1}, 'label', candidates{i,2}); %#ok<AGROW>
        end
    end
end

function values = coerce_numeric_column(col)
    if isnumeric(col)
        values = double(col);
        return;
    end
    if islogical(col)
        values = double(col);
        return;
    end
    if iscell(col)
        values = nan(numel(col), 1);
        for i = 1:numel(col)
            value = col{i};
            if isnumeric(value) && isscalar(value)
                values(i) = double(value);
            elseif islogical(value) && isscalar(value)
                values(i) = double(value);
            elseif isstring(value) || ischar(value)
                parsed = str2double(string(value));
                if ~isnan(parsed)
                    values(i) = parsed;
                end
            end
        end
        return;
    end
    if iscategorical(col) || isstring(col) || ischar(col)
        values = str2double(string(col));
        return;
    end
    try
        values = double(col);
    catch
        values = nan(numel(col), 1);
    end
end

function mask = is_vb_method(method_col)
    method_vals = lower(strtrim(cellstr(string(method_col))));
    mask = contains(method_vals, 'vb');
end

function [nrows, ncols] = choose_subplot_grid(n)
    if n <= 3, nrows = 1; ncols = n;
    elseif n == 4, nrows = 2; ncols = 2;
    elseif n <= 6, nrows = 2; ncols = 3;
    elseif n <= 9, nrows = 3; ncols = 3;
    else, ncols = 4; nrows = ceil(n / ncols);
    end
end

function create_boxplot_comparison(T, outcomes, outcome_labels, plots_dir)
    methods_col = cellstr(string(T.method));
    criteria_col = cellstr(string(T.criterion_clean));
    pairs = [methods_col, criteria_col];
    [~, uid] = unique(string(pairs),'rows'); pairs = pairs(sort(uid),:);
    n = numel(outcomes);
    [nrows, ncols] = choose_subplot_grid(n);
    fig = figure('Visible','off','Position',[100 100 1600 1000], 'Color', 'white');
    
    for i = 1:n
        subplot(nrows, ncols, i);
        outcome = outcomes{i};
        data_groups = {}; group_labels = {};
        for p = 1:size(pairs,1)
            m = pairs{p,1}; c = pairs{p,2};
            mask = strcmp(cellstr(string(T.method)), m) & strcmp(cellstr(string(T.criterion_clean)), c);
            vals = T.(outcome)(mask); vals = vals(~isnan(vals));
            if ~isempty(vals)
                data_groups{end+1} = vals;
                group_labels{end+1} = sprintf('%s\n%s', strrep(m,'_',' '), strrep(c,'_',' '));
            end
        end
        if ~isempty(data_groups)
            allvals = vertcat(data_groups{:});
            group_idx = [];
            for g = 1:numel(data_groups), group_idx = [group_idx; repmat(g, numel(data_groups{g}),1)]; end
            boxplot(allvals, group_idx, 'Labels', group_labels);
            set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', 'ZColor', 'black');
            set(gca,'XTickLabelRotation',45,'TickLabelInterpreter','none');
            ylabel(outcome_labels{i}, 'FontSize', 11, 'FontWeight', 'bold'); 
            title(outcome_labels{i}, 'FontSize', 11); 
            grid on;
            set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
        end
    end
    sgtitle('Method × Criterion Comparison (all combos)', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir,'boxplot_comparison.png'));
    close(fig);
end

function create_method_effects_plot_with_ci(T, outcomes, outcome_labels, method_results, plots_dir)
    global ANALYSIS_CONFIG;
    methods = get_factor_levels(T,'method');
    n_boot = ANALYSIS_CONFIG.n_boot;
    n = numel(outcomes);
    [nrows, ncols] = choose_subplot_grid(n);
    fig = figure('Visible','off','Position',[100 100 1400 900], 'Color', 'white');
    
    for i = 1:n
        subplot(nrows, ncols, i);
        outcome = outcomes{i};
        means = nan(numel(methods),1); cil = means; cih = means;
        for m = 1:numel(methods)
            mname = methods{m};
            mask = strcmp(cellstr(string(T.method)), mname);
            data = T.(outcome)(mask); data = data(~isnan(data));
            means(m) = mean(data,'omitnan');
            boots = nan(n_boot,1);
            for b = 1:n_boot
                if isempty(data), boots(b) = NaN; else boots(b) = mean(datasample(data,numel(data)),'omitnan'); end
            end
            cil(m) = prctile(boots,2.5); cih(m) = prctile(boots,97.5);
        end
        bar(1:numel(methods), means, 'FaceColor',[0.2 0.6 0.8], 'EdgeColor', 'black', 'LineWidth', 1.5);
        hold on;
        errlow = means - cil; errhigh = cih - means;
        errorbar(1:numel(methods), means, errlow, errhigh, 'k.','LineWidth', 2, 'CapSize', 8);
        set(gca,'XTick',1:numel(methods),'XTickLabel',cellfun(@display_method_label,methods,'UniformOutput',false),'XTickLabelRotation',45);
        set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
        ylabel(outcome_labels{i}, 'FontSize', 11, 'FontWeight', 'bold'); 
        title(outcome_labels{i}, 'FontSize', 11); 
        grid on;
        set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    end
    sgtitle('Method effects (shared criteria only)', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir,'method_effects_with_ci.png'));
    close(fig);
end

function create_criterion_effects_plot_with_ci(T, outcomes, outcome_labels, criterion_results, plots_dir)
    global ANALYSIS_CONFIG;
    clean_levels = get_factor_levels(T,'criterion_clean');
    n_boot = ANALYSIS_CONFIG.n_boot;
    n = numel(outcomes);
    [nrows, ncols] = choose_subplot_grid(n);
    fig = figure('Visible','off','Position',[100 100 1400 900], 'Color', 'white');
    
    for i = 1:n
        subplot(nrows, ncols, i);
        outcome = outcomes{i};
        means = nan(numel(clean_levels),1); cil = means; cih = means;
        for c = 1:numel(clean_levels)
            clev = clean_levels{c};
            mask = strcmp(cellstr(string(T.criterion_clean)), clev);
            data = T.(outcome)(mask); data = data(~isnan(data));
            means(c) = mean(data,'omitnan');
            boots = nan(n_boot,1);
            for b = 1:n_boot
                if isempty(data), boots(b) = NaN; else boots(b) = mean(datasample(data,numel(data)),'omitnan'); end
            end
            cil(c) = prctile(boots,2.5); cih(c) = prctile(boots,97.5);
        end
        bar(1:numel(clean_levels), means, 'FaceColor',[0.8 0.2 0.2], 'EdgeColor', 'black', 'LineWidth', 1.5);
        hold on;
        errlow = means - cil; errhigh = cih - means;
        errorbar(1:numel(clean_levels), means, errlow, errhigh, 'k.','LineWidth', 2, 'CapSize', 8);
        set(gca,'XTick',1:numel(clean_levels),'XTickLabel',clean_levels,'XTickLabelRotation',45);
        set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
        ylabel(outcome_labels{i}, 'FontSize', 11, 'FontWeight', 'bold'); 
        title(outcome_labels{i}, 'FontSize', 11); 
        grid on;
        set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    end
    sgtitle('Criterion effects (full-support methods only)', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir,'criterion_effects_with_ci.png'));
    close(fig);
end

function create_snr_effects_plot(T, outcomes, outcome_labels, plots_dir)
    methods = get_factor_levels(T,'method');
    n = numel(outcomes);
    [nrows, ncols] = choose_subplot_grid(n);
    fig = figure('Visible','off','Position',[100 100 1400 900], 'Color', 'white');
    snr_vals = sort(unique(T.SNR_dB));
    colors = lines(numel(methods));
    
    for i = 1:n
        subplot(nrows, ncols, i);
        outcome = outcomes{i};
        hold on;
        for m = 1:numel(methods)
            mname = methods{m};
            means = nan(size(snr_vals));
            for s = 1:numel(snr_vals)
                mask = strcmp(cellstr(string(T.method)), mname) & T.SNR_dB==snr_vals(s);
                data = T.(outcome)(mask); data = data(~isnan(data));
                if ~isempty(data), means(s) = mean(data,'omitnan'); end
            end
            plot(snr_vals, means,'-o','Color',colors(m,:),'LineWidth', 2.5, 'MarkerSize', 8, ...
                 'DisplayName',display_method_label(mname));
        end
        set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
        xlabel('SNR (dB)', 'FontSize', 11, 'FontWeight', 'bold'); 
        ylabel(outcome_labels{i}, 'FontSize', 11, 'FontWeight', 'bold'); 
        title(outcome_labels{i}, 'FontSize', 11); 
        legend('Location','best', 'Interpreter', 'none', 'EdgeColor', 'black'); 
        grid on;
        set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    end
    sgtitle('SNR effects by method (aggregated over criteria)', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir,'snr_effects.png'));
    close(fig);
end

function create_interaction_plot(T, outcomes, outcome_labels, plots_dir)
    methods = get_factor_levels(T,'method'); 
    clean_levels = get_factor_levels(T,'criterion_clean');
    n = numel(outcomes);
    [nrows, ncols] = choose_subplot_grid(n);
    fig = figure('Visible','off','Position',[100 100 1600 1000], 'Color', 'white');
    colors = lines(numel(methods));
    
    for i = 1:n
        subplot(nrows, ncols, i);
        outcome = outcomes{i};
        tablevals = nan(numel(methods), numel(clean_levels));
        for m = 1:numel(methods)
            for c = 1:numel(clean_levels)
                mask = strcmp(cellstr(string(T.method)), methods{m}) & strcmp(cellstr(string(T.criterion_clean)), clean_levels{c});
                data = T.(outcome)(mask); data = data(~isnan(data));
                if ~isempty(data), tablevals(m,c) = mean(data,'omitnan'); end
            end
        end
        hold on;
        for m = 1:numel(methods)
            plot(1:numel(clean_levels), tablevals(m,:), 'o-','Color',colors(m,:),'LineWidth', 2.5, ...
                 'MarkerSize', 8, 'DisplayName',display_method_label(methods{m}));
        end
        set(gca,'XTick',1:numel(clean_levels),'XTickLabel',clean_levels,'XTickLabelRotation',45);
        set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
        xlabel('Criterion', 'FontSize', 11, 'FontWeight', 'bold'); 
        ylabel(outcome_labels{i}, 'FontSize', 11, 'FontWeight', 'bold'); 
        title(outcome_labels{i}, 'FontSize', 11); 
        legend('Location','best', 'Interpreter', 'none', 'EdgeColor', 'black'); 
        grid on;
        set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    end
    sgtitle('Method × Criterion Interaction', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir,'interaction_plot.png'));
    close(fig);
end

function create_cross_validation_plot(cv_results, outcomes, outcome_labels, plots_dir)
    n = numel(outcomes);
    [nrows, ncols] = choose_subplot_grid(n);
    fig = figure('Visible','off','Position',[100 100 1400 900], 'Color', 'white');
    
    for i = 1:n
        subplot(nrows, ncols, i);
        outcome = outcomes{i};
        if isfield(cv_results, outcome)
            fe = cv_results.(outcome).fold_errors;
            nf = cv_results.(outcome).n_folds;
            if ~isempty(fe)
                bar(1:numel(fe), fe, 'FaceColor',[0.2 0.8 0.2], 'EdgeColor', 'black', 'LineWidth', 1.5);
                hold on; 
                yline(mean(fe),'--r','LineWidth', 2.5, 'DisplayName',sprintf('Mean=%.4f',mean(fe)));
                set(gca,'XTick',1:numel(fe),'XTickLabel',arrayfun(@(k)sprintf('Fold %d',k),1:numel(fe),'UniformOutput',false));
                set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
                ylabel('CV Error', 'FontSize', 11, 'FontWeight', 'bold'); 
                title(sprintf('%s (CV, %d folds)', outcome_labels{i}, nf), 'FontSize', 11); 
                legend('Location','best', 'EdgeColor', 'black'); 
                grid on;
                set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
            end
        end
    end
    sgtitle('Cross-validation Results', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir,'cross_validation.png'));
    close(fig);
end

function create_vb_covariance_relationship_plots(T, outcomes, outcome_labels, vb_covariance_results, plots_dir)
    if isempty(fieldnames(vb_covariance_results)) || ~isfield(vb_covariance_results, 'covariates')
        return;
    end

    vb_mask = is_vb_method(T.method);
    T_vb = T(vb_mask, :);
    if height(T_vb) == 0
        return;
    end

    cov_specs = vb_covariance_results.covariates;
    [nrows, ncols] = choose_subplot_grid(numel(outcomes));

    for i = 1:numel(cov_specs)
        spec = cov_specs(i);
        covariance_vals = coerce_numeric_column(T_vb.(spec.name));
        fig = figure('Visible', 'off', 'Position', [100 100 1500 950], 'Color', 'white');

        for j = 1:numel(outcomes)
            outcome = outcomes{j};
            subplot(nrows, ncols, j);
            outcome_vals = coerce_numeric_column(T_vb.(outcome));
            valid_mask = ~isnan(covariance_vals) & ~isnan(outcome_vals);
            xv = outcome_vals(valid_mask);
            yv = covariance_vals(valid_mask);

            if numel(xv) < 3
                text(0.5, 0.5, 'Insufficient data', 'HorizontalAlignment', 'center', 'FontSize', 12);
                axis off;
                title(outcome_labels{j}, 'FontSize', 11);
                continue;
            end

            if ismember('SNR_dB', T_vb.Properties.VariableNames)
                snr_vals = coerce_numeric_column(T_vb.SNR_dB);
                sv = snr_vals(valid_mask);
                scatter(xv, yv, 36, sv, 'filled', 'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', [0.15 0.15 0.15]);
                cb = colorbar;
                ylabel(cb, 'SNR (dB)', 'FontSize', 9);
            else
                scatter(xv, yv, 36, [0.2 0.45 0.7], 'filled', 'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', [0.15 0.15 0.15]);
            end
            hold on;

            if numel(unique(xv)) > 1
                xfit = linspace(min(xv), max(xv), 100);
                stats = vb_covariance_results.(spec.name).(outcome);
                yfit = stats.slope * xfit + stats.intercept;
                plot(xfit, yfit, '-', 'Color', [0.85 0.2 0.2], 'LineWidth', 2);
                title(sprintf('%s\n\\rho_s=%.2f, p=%.3g', outcome_labels{j}, stats.spearman_rho, stats.spearman_p), 'FontSize', 11);
            else
                title(outcome_labels{j}, 'FontSize', 11);
            end

            xlabel(outcome_labels{j}, 'FontSize', 10, 'FontWeight', 'bold');
            ylabel(spec.label, 'FontSize', 10, 'FontWeight', 'bold');
            grid on;
            set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
            set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
        end

        sgtitle(sprintf('VB covariance relationships: %s', spec.label), 'FontSize', 13, 'FontWeight', 'bold');
        saveas(fig, fullfile(plots_dir, sprintf('vb_covariance_relationships_%s.png', spec.name)));
        close(fig);
    end
end

function create_vb_covariance_kcorrect_barplots(vb_covariance_results, plots_dir)
    if isempty(fieldnames(vb_covariance_results)) || ~isfield(vb_covariance_results, 'covariates')
        return;
    end

    cov_specs = vb_covariance_results.covariates;
    [nrows, ncols] = choose_subplot_grid(numel(cov_specs));
    fig = figure('Visible', 'off', 'Position', [120 120 1400 800], 'Color', 'white');
    group_labels = {'K correct', 'K incorrect'};
    face_colors = [0.25 0.55 0.85; 0.85 0.35 0.35];

    for i = 1:numel(cov_specs)
        spec = cov_specs(i);
        subplot(nrows, ncols, i);
        stats = vb_covariance_results.(spec.name).k_correct_groups;
        means = [stats.mean_correct, stats.mean_incorrect];
        ci_low = [stats.ci_correct(1), stats.ci_incorrect(1)];
        ci_high = [stats.ci_correct(2), stats.ci_incorrect(2)];
        err_low = means - ci_low;
        err_high = ci_high - means;

        b = bar(1:2, means, 'FaceColor', 'flat', 'EdgeColor', 'black', 'LineWidth', 1.5);
        b.CData = face_colors;
        hold on;
        errorbar(1:2, means, err_low, err_high, 'k.', 'LineWidth', 1.8, 'CapSize', 10);

        ylim_vals = [ci_low, ci_high];
        ylim_vals = ylim_vals(~isnan(ylim_vals));
        if ~isempty(ylim_vals)
            span = max(ylim_vals) - min(ylim_vals);
            if span <= 0
                span = max(1, abs(max(ylim_vals)) * 0.1);
            end
            ylim([min(ylim_vals) - 0.15 * span, max(ylim_vals) + 0.2 * span]);
        end

        set(gca, 'XTick', 1:2, 'XTickLabel', group_labels, 'XTickLabelRotation', 20);
        ylabel(spec.label, 'FontSize', 10, 'FontWeight', 'bold');
        title(sprintf('%s\nDelta=%.3f [%0.3f, %0.3f]', ...
            spec.label, stats.mean_difference, stats.ci_difference(1), stats.ci_difference(2)), 'FontSize', 11);
        text(1, means(1), sprintf(' n=%d', stats.n_correct), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 9);
        text(2, means(2), sprintf(' n=%d', stats.n_incorrect), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 9);
        grid on;
        set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
        set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    end

    sgtitle('VB covariance by K selection accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir, 'vb_covariance_by_k_correct.png'));
    close(fig);
end

function create_vb_covariance_quintile_topoplots(T, vb_covariance_results, results_dir, plots_dir)
    if isempty(fieldnames(vb_covariance_results)) || ~isfield(vb_covariance_results, 'covariates') || ...
            isempty(vb_covariance_results.covariates)
        return;
    end
    if ~ismember('json_file', T.Properties.VariableNames) || exist('topoplot', 'file') ~= 2
        return;
    end

    util = microstate_utilities();
    cfg = util.load_config();
    template_file = char(cfg.paths.template_file);
    if isempty(template_file) || ~isfile(template_file)
        fprintf('Skipping VB covariance quintile topoplots: template file not found.\n');
        return;
    end

    cov_spec = vb_covariance_results.covariates(1);
    vb_mask = is_vb_method(T.method);
    T_vb = T(vb_mask, :);
    if height(T_vb) == 0 || ~ismember(cov_spec.name, T_vb.Properties.VariableNames)
        return;
    end

    covariance_vals = coerce_numeric_column(T_vb.(cov_spec.name));
    json_paths = cellstr(string(T_vb.json_file));
    output_dir = fileparts(char(results_dir));
    json_dir = fullfile(output_dir, 'microstates_json');
    valid_json_mask = false(size(json_paths));
    resolved_json_paths = cell(size(json_paths));
    for i = 1:numel(json_paths)
        resolved_json_paths{i} = resolve_json_path_local(json_paths{i}, output_dir, json_dir);
        valid_json_mask(i) = ~isempty(resolved_json_paths{i}) && isfile(resolved_json_paths{i});
    end

    valid_mask = ~isnan(covariance_vals) & valid_json_mask;
    T_vb = T_vb(valid_mask, :);
    covariance_vals = covariance_vals(valid_mask);
    resolved_json_paths = resolved_json_paths(valid_mask);
    if numel(covariance_vals) < 5
        fprintf('Skipping VB covariance quintile topoplots: fewer than five valid SPM-VB rows with covariance and JSON output.\n');
        return;
    end

    [~, template_labels, template_channel_labels, template_chanlocs] = load_metamaps_templates(template_file, 'K', 7);
    if numel(template_chanlocs) < 4 || isempty(template_channel_labels)
        fprintf('Skipping VB covariance quintile topoplots: template scalp geometry is unavailable.\n');
        return;
    end

    quintile_idx = rank_based_quintile_groups(covariance_vals, 5);
    n_quintiles = max(quintile_idx);
    n_labels = numel(template_labels);
    avg_maps_by_quintile = cell(n_quintiles, 1);
    label_counts = zeros(n_quintiles, n_labels);
    row_counts = zeros(n_quintiles, 1);
    quintile_ranges = nan(n_quintiles, 2);

    for q = 1:n_quintiles
        mask_q = quintile_idx == q;
        quintile_ranges(q, :) = [min(covariance_vals(mask_q)), max(covariance_vals(mask_q))];
        sum_maps = zeros(n_labels, numel(template_channel_labels));
        count_maps = zeros(n_labels, numel(template_channel_labels));

        row_indices = find(mask_q);
        for i = 1:numel(row_indices)
            row_idx = row_indices(i);
            json_file = resolved_json_paths{row_idx};
            try
                data = jsondecode(fileread(json_file));
                [maps, channel_labels] = extract_json_state_maps_local(data, 'estimated_microstates');
                alignment = align_microstates_to_template(maps, template_file, ...
                    'estimated_channel_labels', channel_labels, 'template_K', 7);
                projected_maps = project_maps_to_template_channels_local( ...
                    alignment.aligned_maps, channel_labels, template_channel_labels);
                added_row = false;
                for s = 1:numel(alignment.labels)
                    label_idx = find(strcmp(alignment.labels{s}, template_labels), 1, 'first');
                    if isempty(label_idx)
                        continue;
                    end
                    vals = projected_maps(s, :);
                    finite_mask = isfinite(vals);
                    if ~any(finite_mask)
                        continue;
                    end
                    sum_maps(label_idx, finite_mask) = sum_maps(label_idx, finite_mask) + vals(finite_mask);
                    count_maps(label_idx, finite_mask) = count_maps(label_idx, finite_mask) + 1;
                    label_counts(q, label_idx) = label_counts(q, label_idx) + 1;
                    added_row = true;
                end
                if added_row
                    row_counts(q) = row_counts(q) + 1;
                end
            catch ME
                fprintf('Skipping VB covariance quintile topoplot contribution from %s: %s\n', json_file, ME.message);
            end
        end

        avg_maps = nan(size(sum_maps));
        finite_mask = count_maps > 0;
        avg_maps(finite_mask) = sum_maps(finite_mask) ./ count_maps(finite_mask);
        avg_maps_by_quintile{q} = normalize_map_rows_omitnan_local(avg_maps);
    end

    finite_blocks = cellfun(@(x) x(isfinite(x)), avg_maps_by_quintile, 'UniformOutput', false);
    finite_blocks = finite_blocks(~cellfun(@isempty, finite_blocks));
    if isempty(finite_blocks)
        fprintf('Skipping VB covariance quintile topoplots: no usable aligned maps were found.\n');
        return;
    end

    clim = max(abs(cat(1, finite_blocks{:})));
    if ~isfinite(clim) || clim <= eps
        clim = 1;
    end

    fig = figure('Visible', 'off', 'Position', [80 80 1850 1550], 'Color', 'white');
    tl = tiledlayout(fig, n_quintiles, n_labels, 'TileSpacing', 'compact', 'Padding', 'compact');
    colormap(fig, 'jet');

    for q = 1:n_quintiles
        maps_this = avg_maps_by_quintile{q};
        for k = 1:n_labels
            ax = nexttile(tl, (q - 1) * n_labels + k);
            vals = maps_this(k, :);
            if all(~isfinite(vals))
                axis(ax, 'off');
                text(ax, 0.5, 0.55, 'No data', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', [0.45 0.45 0.45]);
            else
                topoplot(vals, template_chanlocs, 'electrodes', 'off', 'numcontour', 6, 'maplimits', [-clim clim]);
                axis(ax, 'off');
            end
            if q == 1
                title(ax, template_labels{k}, 'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');
            end
            text(ax, 0.5, -0.08, sprintf('n=%d', label_counts(q, k)), ...
                'Units', 'normalized', 'HorizontalAlignment', 'center', ...
                'FontSize', 9, 'Color', [0.25 0.25 0.25], 'FontWeight', 'bold');
            if k == 1
                text(ax, -0.30, 0.5, sprintf(['Q%d\n%s\n[%.3g, %.3g]\n(%d rows)'], ...
                    q, covariance_quintile_label(q, n_quintiles), quintile_ranges(q, 1), quintile_ranges(q, 2), row_counts(q)), ...
                    'Units', 'normalized', 'Rotation', 90, 'HorizontalAlignment', 'center', ...
                    'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'none');
            end
        end
    end

    sgtitle(tl, sprintf('SPM-VB microstate topographies by covariance quintile: %s', cov_spec.label), ...
        'FontSize', 15, 'FontWeight', 'bold');
    saveas(fig, fullfile(plots_dir, sprintf('vb_covariance_quintile_topoplots_%s.png', cov_spec.name)));
    close(fig);
end

function create_backfit_confusion_comparison_plots(T, results_dir, plots_dir)
    if ~istable(T) || height(T) == 0 || ~ismember('backfit_diagnostic_file', T.Properties.VariableNames)
        return;
    end

    file_vals = cellstr(string(T.backfit_diagnostic_file));
    valid_file = ~cellfun(@isempty, file_vals) & cellfun(@isfile, file_vals);
    if ~any(valid_file)
        return;
    end

    underfit_mask = false(height(T), 1);
    if ismember('K_true', T.Properties.VariableNames) && ismember('K_estimated', T.Properties.VariableNames)
        underfit_mask = double(T.K_true) > double(T.K_estimated);
    end

    method_vals = lower(strtrim(cellstr(string(T.method))));
    kmeans_mask = contains(method_vals, 'kmeans') & ~contains(method_vals, 'spm');
    vb_mask = contains(method_vals, 'vb');
    keep_mask = valid_file & underfit_mask & (kmeans_mask | vb_mask);
    if ~any(keep_mask)
        return;
    end

    T_plot = T(keep_mask, :);
    util = microstate_utilities();
    cfg = util.load_config();
    template_file = '';
    if isfield(cfg, 'paths') && isfield(cfg.paths, 'template_file')
        template_file = util.resolve_path(char(cfg.paths.template_file), util.project_root());
    end

    template_maps = [];
    template_labels = {};
    template_chanlocs = [];
    try
        if ~isempty(template_file) && isfile(template_file)
            [template_maps, template_labels] = load_metamaps_templates(template_file, 'K', 7);
            [template_maps, template_chanlocs] = prepare_template_topoplot_data(template_file, template_maps);
        end
    catch ME
        warning('Could not load canonical template topographies for confusion plots: %s', ME.message);
    end

    output_dir = fullfile(plots_dir, 'backfit_confusion_comparisons');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    group_keys = strcat( ...
        string(T_plot.criterion_clean), "|", ...
        string(T_plot.K_true), "|", ...
        string(T_plot.K_estimated));
    [unique_keys, ~, key_idx] = unique(group_keys, 'stable');

    manifest_rows = cell(numel(unique_keys), 1);
    row_count = 0;
    for g = 1:numel(unique_keys)
        group_rows = T_plot(key_idx == g, :);
        if height(group_rows) == 0
            continue;
        end

        group_methods = lower(strtrim(cellstr(string(group_rows.method))));
        rows_kmeans = group_rows(contains(group_methods, 'kmeans') & ~contains(group_methods, 'spm'), :);
        rows_vb = group_rows(contains(group_methods, 'vb'), :);
        if height(rows_kmeans) == 0 && height(rows_vb) == 0
            continue;
        end

        [counts_kmeans, labels_kmeans, n_runs_kmeans] = aggregate_backfit_confusions(rows_kmeans.backfit_diagnostic_file);
        [counts_vb, labels_vb, n_runs_vb] = aggregate_backfit_confusions(rows_vb.backfit_diagnostic_file);
        if isempty(counts_kmeans) && isempty(counts_vb)
            continue;
        end

        label_order = canonical_confusion_label_order(template_labels, labels_kmeans, labels_vb);
        if ~isempty(counts_kmeans)
            counts_kmeans = reorder_confusion_matrix(counts_kmeans, labels_kmeans, label_order);
            rownorm_kmeans = normalize_confusion_rows(counts_kmeans);
        else
            rownorm_kmeans = [];
        end
        if ~isempty(counts_vb)
            counts_vb = reorder_confusion_matrix(counts_vb, labels_vb, label_order);
            rownorm_vb = normalize_confusion_rows(counts_vb);
        else
            rownorm_vb = [];
        end

        criterion = char(string(group_rows.criterion_clean(1)));
        K_true = double(group_rows.K_true(1));
        K_estimated = double(group_rows.K_estimated(1));
        file_stub = sprintf('confusion_compare__%s__Ktrue%d__Kest%d', ...
            sanitize_stub(criterion), K_true, K_estimated);
        output_file = fullfile(output_dir, [file_stub '.png']);

        create_method_confusion_pair_plot( ...
            rownorm_kmeans, rownorm_vb, counts_kmeans, counts_vb, label_order, ...
            template_maps, template_labels, template_chanlocs, ...
            output_file, criterion, K_true, K_estimated, n_runs_kmeans, n_runs_vb);

        row_count = row_count + 1;
        manifest_rows{row_count} = table( ...
            string(criterion), K_true, K_estimated, K_true - K_estimated, ...
            n_runs_kmeans, n_runs_vb, string(output_file), ...
            'VariableNames', {'criterion', 'K_true', 'K_estimated', 'K_gap', ...
            'n_runs_kmeans', 'n_runs_vb', 'comparison_plot'});
    end

    manifest_rows = manifest_rows(1:row_count);
    if row_count > 0
        manifest = vertcat(manifest_rows{:});
        writetable(manifest, fullfile(output_dir, 'backfit_confusion_comparison_manifest.csv'));
    end
end

function create_simulated_backfit_confusion_summary(T, results_dir, plots_dir)
    if ~istable(T) || height(T) == 0 || ~ismember('backfit_diagnostic_file', T.Properties.VariableNames)
        return;
    end

    results_csv = fullfile(results_dir, 'comparison_results.csv');
    if ~isfile(results_csv)
        return;
    end

    diag_files = cellstr(string(T.backfit_diagnostic_file));
    valid_diag_mask = ~cellfun(@isempty, diag_files) & cellfun(@isfile, diag_files);
    if ~any(valid_diag_mask)
        fprintf('Skipping simulated backfit confusion summary: no valid diagnostic files were found.\n');
        return;
    end

    output_file = fullfile(plots_dir, 'backfit_confusion_summary.png');
    try
        plot_simulated_backfit_confusion_summary( ...
            'results_csv', results_csv, ...
            'output_file', output_file, ...
            'visible', false);
    catch ME
        fprintf('Skipping simulated backfit confusion summary: %s\n', ME.message);
    end
end

function [counts_total, labels, n_runs] = aggregate_backfit_confusions(files)
    counts_total = [];
    labels = {};
    n_runs = 0;
    if isempty(files)
        return;
    end

    file_list = cellstr(string(files));
    for i = 1:numel(file_list)
        file_i = char(string(file_list{i}));
        if isempty(file_i) || ~isfile(file_i)
            continue;
        end
        S = load(file_i, 'BackfitDiagnostics');
        if ~isfield(S, 'BackfitDiagnostics') || ~isfield(S.BackfitDiagnostics, 'ok') || ~S.BackfitDiagnostics.ok
            continue;
        end
        diag_i = S.BackfitDiagnostics;
        labels_i = cellstr(string(diag_i.template_labels(:)));
        counts_i = double(diag_i.confusion_counts);
        if isempty(counts_total)
            labels = labels_i;
            counts_total = counts_i;
        else
            [labels, counts_total] = merge_confusion_matrices(labels, counts_total, labels_i, counts_i);
        end
        n_runs = n_runs + 1;
    end
end

function [labels_out, values_out] = merge_confusion_matrices(labels_a, values_a, labels_b, values_b)
    labels_out = canonical_confusion_label_order({}, labels_a, labels_b);
    values_out = zeros(numel(labels_out), numel(labels_out));
    values_out = values_out + reorder_confusion_matrix(values_a, labels_a, labels_out);
    values_out = values_out + reorder_confusion_matrix(values_b, labels_b, labels_out);
end

function label_order = canonical_confusion_label_order(template_labels, labels_a, labels_b)
    label_order = {};
    if ~isempty(template_labels)
        label_order = cellstr(string(template_labels(:)));
    end
    extras = [cellstr(string(labels_a(:))); cellstr(string(labels_b(:)))];
    for i = 1:numel(extras)
        if ~any(strcmp(label_order, extras{i}))
            label_order{end+1, 1} = extras{i}; %#ok<AGROW>
        end
    end
    label_order = label_order(:);
end

function values_out = reorder_confusion_matrix(values_in, labels_in, label_order)
    n = numel(label_order);
    values_out = zeros(n, n);
    if isempty(values_in) || isempty(labels_in)
        return;
    end

    label_in = cellstr(string(labels_in(:)));
    for r = 1:n
        src_r = find(strcmp(label_in, label_order{r}), 1, 'first');
        if isempty(src_r)
            continue;
        end
        for c = 1:n
            src_c = find(strcmp(label_in, label_order{c}), 1, 'first');
            if isempty(src_c)
                continue;
            end
            values_out(r, c) = values_in(src_r, src_c);
        end
    end
end

function rownorm = normalize_confusion_rows(values)
    rownorm = values;
    if isempty(values)
        return;
    end
    for r = 1:size(values, 1)
        row = values(r, :);
        finite_mask = isfinite(row);
        if ~any(finite_mask)
            continue;
        end
        row_sum = sum(row(finite_mask));
        if row_sum > 0
            rownorm(r, finite_mask) = row(finite_mask) ./ row_sum;
        end
    end
end

function create_method_confusion_pair_plot( ...
        rownorm_kmeans, rownorm_vb, counts_kmeans, counts_vb, label_order, ...
        template_maps, template_labels, template_chanlocs, ...
        output_file, criterion, K_true, K_estimated, n_runs_kmeans, n_runs_vb)
    n_labels = numel(label_order);
    fig = figure('Visible', 'off', 'Position', [60 60 1820 980], 'Color', 'white');
    sgtitle(sprintf('Microstate misrecognition | %s | K_{true}=%d, K_{est}=%d', ...
        strrep(criterion, '_', ' '), K_true, K_estimated), 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'tex');

    x_row = 0.03;
    w_row = 0.12;
    gap = 0.02;
    x_left = x_row + w_row + gap;
    w_heat = 0.29;
    x_right = x_left + w_heat + 0.12;
    y_heat = 0.16;
    h_heat = 0.58;
    y_top = y_heat + h_heat + 0.035;
    h_top = 0.11;
    clim = [0 1];

    left_title = sprintf('kmeans (n=%d)', n_runs_kmeans);
    right_title = sprintf('VB (n=%d)', n_runs_vb);
    annotation(fig, 'textbox', [x_left, y_top + h_top + 0.01, w_heat, 0.03], 'String', left_title, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    annotation(fig, 'textbox', [x_right, y_top + h_top + 0.01, w_heat, 0.03], 'String', right_title, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');

    for i = 1:n_labels
        y_ax = y_heat + (n_labels - i) * (h_heat / n_labels);
        ax_row = axes('Parent', fig, 'Position', [x_row, y_ax, w_row, h_heat / n_labels]);
        render_template_topography(ax_row, label_order{i}, template_maps, template_labels, template_chanlocs);
    end

    for i = 1:n_labels
        x_left_ax = x_left + (i - 1) * (w_heat / n_labels);
        ax_top_left = axes('Parent', fig, 'Position', [x_left_ax, y_top, w_heat / n_labels, h_top]);
        render_template_topography(ax_top_left, label_order{i}, template_maps, template_labels, template_chanlocs);

        x_right_ax = x_right + (i - 1) * (w_heat / n_labels);
        ax_top_right = axes('Parent', fig, 'Position', [x_right_ax, y_top, w_heat / n_labels, h_top]);
        render_template_topography(ax_top_right, label_order{i}, template_maps, template_labels, template_chanlocs);
    end

    ax_left = axes('Parent', fig, 'Position', [x_left, y_heat, w_heat, h_heat]);
    render_confusion_heatmap(ax_left, rownorm_kmeans, label_order, counts_kmeans, clim);
    ax_right = axes('Parent', fig, 'Position', [x_right, y_heat, w_heat, h_heat]);
    render_confusion_heatmap(ax_right, rownorm_vb, label_order, counts_vb, clim);

    annotation(fig, 'textbox', [x_row, y_heat + h_heat + 0.005, w_row, 0.028], 'String', 'True label', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    annotation(fig, 'textbox', [x_left, y_heat - 0.07, w_heat, 0.03], 'String', 'Estimated label', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    annotation(fig, 'textbox', [x_right, y_heat - 0.07, w_heat, 0.03], 'String', 'Estimated label', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');

    cb = colorbar(ax_right, 'eastoutside');
    cb.Position = [0.92 0.22 0.015 0.46];
    ylabel(cb, 'Row-normalized confusion', 'FontSize', 10);
    saveas(fig, output_file);
    close(fig);
end

function render_confusion_heatmap(ax, values, label_order, counts, clim)
    axes(ax);
    if isempty(values) || all(~isfinite(values), 'all')
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
        axis(ax, 'off');
        return;
    end

    imagesc(ax, values, clim);
    axis(ax, 'square');
    colormap(ax, parula(256));
    set(ax, 'XTick', 1:numel(label_order), 'XTickLabel', repmat({''}, 1, numel(label_order)), ...
        'YTick', 1:numel(label_order), 'YTickLabel', repmat({''}, 1, numel(label_order)), ...
        'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    grid(ax, 'on');
    ax.GridColor = [1 1 1];
    ax.GridAlpha = 0.4;
    ax.LineWidth = 1;
    ax.XTick = 1:numel(label_order);
    ax.YTick = 1:numel(label_order);
    ax.XMinorTick = 'off';
    ax.YMinorTick = 'off';

    if isempty(counts)
        counts = nan(size(values));
    end
    for r = 1:size(values, 1)
        for c = 1:size(values, 2)
            v = values(r, c);
            if ~isfinite(v)
                continue;
            end
            text_color = [0 0 0];
            if v >= 0.6
                text_color = [1 1 1];
            end
            if r <= size(counts, 1) && c <= size(counts, 2) && isfinite(counts(r, c))
                cell_txt = sprintf('%.2f\n(n=%.0f)', v, counts(r, c));
            else
                cell_txt = sprintf('%.2f', v);
            end
            text(ax, c, r, cell_txt, 'HorizontalAlignment', 'center', 'Color', text_color, ...
                'FontSize', 8, 'FontWeight', 'bold');
        end
    end
end

function render_template_topography(ax, label, template_maps, template_labels, template_chanlocs)
    axes(ax);
    cla(ax);
    axis(ax, 'off');
    set(ax, 'Color', 'white');

    idx = [];
    if ~isempty(template_labels)
        idx = find(strcmp(cellstr(string(template_labels(:))), char(string(label))), 1, 'first');
    end
    if isempty(idx) || isempty(template_maps)
        text(0.5, 0.5, char(string(label)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        return;
    end

    [vals, ok] = select_template_topography_values(template_maps, idx, numel(template_chanlocs));
    if exist('topoplot', 'file') == 2 && ~isempty(template_chanlocs) && ok
        try
            topoplot(vals, template_chanlocs, 'electrodes', 'off', 'numcontour', 6, 'maplimits', 'absmax');
            axis(ax, 'off');
            colormap(ax, jet(256));
        catch ME
            warning('Template topography for %s fell back to text because topoplot failed: %s', char(string(label)), ME.message);
            text(0.5, 0.5, char(string(label)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        end
    else
        text(0.5, 0.5, char(string(label)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    end
    title(ax, char(string(label)), 'FontSize', 9, 'FontWeight', 'bold');
end

function [template_maps_out, chanlocs_out] = prepare_template_topoplot_data(template_file, template_maps_in)
    template_maps_out = template_maps_in;
    chanlocs_out = [];
    if isempty(template_maps_in) || isempty(template_file) || ~isfile(template_file) || exist('pop_loadset', 'file') ~= 2
        return;
    end

    EEG = pop_loadset('filename', char(template_file));
    if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
        return;
    end

    n_channels = size(template_maps_in, 2);
    if numel(EEG.chanlocs) < n_channels
        warning('Template chanloc count (%d) is smaller than map width (%d); skipping template topoplots.', numel(EEG.chanlocs), n_channels);
        return;
    end

    util = microstate_utilities();
    [chanlocs_out, keep] = util.prepare_metamaps_chanlocs(EEG.chanlocs, n_channels);
    if numel(chanlocs_out) < 4
        warning('Fewer than four canonical template channels have usable topoplot geometry; skipping template topoplots.');
        return;
    end

    template_maps_out = template_maps_in(:, keep);
end

function [vals, ok] = select_template_topography_values(template_maps, idx, n_chanlocs)
    vals = [];
    ok = false;
    if isempty(template_maps) || isempty(idx) || n_chanlocs < 1
        return;
    end

    if idx <= size(template_maps, 1)
        candidate = double(template_maps(idx, :));
        if numel(candidate) == n_chanlocs
            vals = candidate(:)';
            ok = true;
            return;
        end
    end

    if idx <= size(template_maps, 2)
        candidate = double(template_maps(:, idx));
        if numel(candidate) == n_chanlocs
            vals = candidate(:)';
            ok = true;
        end
    end
end

function create_vb_covariance_correlation_heatmap(vb_covariance_results, outcomes, outcome_labels, plots_dir)
    if isempty(fieldnames(vb_covariance_results)) || ~isfield(vb_covariance_results, 'covariates')
        return;
    end

    cov_specs = vb_covariance_results.covariates;
    rho_mat = nan(numel(cov_specs), numel(outcomes));
    p_mat = nan(numel(cov_specs), numel(outcomes));

    for i = 1:numel(cov_specs)
        metric_name = cov_specs(i).name;
        for j = 1:numel(outcomes)
            outcome = outcomes{j};
            if isfield(vb_covariance_results.(metric_name), outcome)
                rho_mat(i, j) = vb_covariance_results.(metric_name).(outcome).spearman_rho;
                p_mat(i, j) = vb_covariance_results.(metric_name).(outcome).spearman_p;
            end
        end
    end

    if all(isnan(rho_mat), 'all')
        return;
    end

    fig = figure('Visible', 'off', 'Position', [120 120 1200 380], 'Color', 'white');
    imagesc(rho_mat, [-1 1]);
    colormap(redbluecmap());
    cb = colorbar;
    ylabel(cb, 'Spearman \rho', 'FontSize', 10);
    set(gca, 'XTick', 1:numel(outcomes), 'XTickLabel', outcome_labels, 'XTickLabelRotation', 35, ...
        'YTick', 1:numel(cov_specs), 'YTickLabel', {cov_specs.label}, ...
        'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    title('VB covariance vs success metrics', 'FontSize', 13, 'FontWeight', 'bold');

    for i = 1:size(rho_mat, 1)
        for j = 1:size(rho_mat, 2)
            if ~isnan(rho_mat(i, j))
                if p_mat(i, j) < 0.001
                    sig = '***';
                elseif p_mat(i, j) < 0.01
                    sig = '**';
                elseif p_mat(i, j) < 0.05
                    sig = '*';
                else
                    sig = '';
                end
                text(j, i, sprintf('%.2f%s', rho_mat(i, j), sig), ...
                    'HorizontalAlignment', 'center', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end

    saveas(fig, fullfile(plots_dir, 'vb_covariance_correlation_heatmap.png'));
    close(fig);
end

function create_runtime_snr_plot(T, plots_dir)
    % Create a dedicated plot for runtime vs SNR effects
    % Shows runtime across SNR levels, with separate lines for each method and criterion combination
    methods = get_factor_levels(T, 'method');
    clean_levels = get_factor_levels(T, 'criterion_clean');
    snr_vals = sort(unique(T.SNR_dB));
    
    fig = figure('Visible','off','Position',[100 100 1200 700], 'Color', 'white');
    hold on;
    
    % Create combinations of methods and criteria for different line styles
    colors = hsv(numel(methods));
    line_styles = {'-', '--', '-.', ':'};
    marker_styles = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h'};
    
    plot_count = 0;
    legend_entries = {};
    
    for m = 1:numel(methods)
        for c = 1:numel(clean_levels)
            mname = methods{m};
            clev = clean_levels{c};
            means = nan(size(snr_vals));
            n_vals = zeros(size(snr_vals));
            
            % Calculate means for each SNR level
            for s = 1:numel(snr_vals)
                mask = strcmp(cellstr(string(T.method)), mname) & ...
                       strcmp(cellstr(string(T.criterion_clean)), clev) & ...
                       T.SNR_dB == snr_vals(s);
                data = T.runtime_s(mask);
                data = data(~isnan(data));
                if ~isempty(data)
                    means(s) = mean(data, 'omitnan');
                    n_vals(s) = numel(data);
                end
            end
            
            % Only plot if we have data for this combination
            if ~all(isnan(means))
                plot_count = plot_count + 1;
                line_style = line_styles{mod(c-1, numel(line_styles)) + 1};
                marker = marker_styles{mod(m-1, numel(marker_styles)) + 1};
                label = sprintf('%s - %s', display_method_label(mname), strrep(clev,'_',' '));
                
                plot(snr_vals, means, line_style, 'Color', colors(m,:), 'LineWidth', 2.5, ...
                     'Marker', marker, 'MarkerSize', 8, 'DisplayName', label);
                
                legend_entries{end+1} = label;
            end
        end
    end
    
    set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Runtime (seconds)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    title('Runtime vs SNR by Method and Criterion', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'black');
    
    if plot_count > 0
        legend('Location', 'best', 'Interpreter', 'none', 'EdgeColor', 'black', 'FontSize', 10);
    end
    
    grid on;
    
    saveas(fig, fullfile(plots_dir, 'runtime_snr_effect.png'));
    close(fig);
end

function create_avg_k_error_plot(T, plots_dir)
    if ~ismember('K_error', T.Properties.VariableNames)
        warning('No K_error to plot.'); return;
    end
    methods = get_factor_levels(T,'method'); 
    clean_levels = get_factor_levels(T,'criterion_clean');
    nM = numel(methods); nC = numel(clean_levels);
    mean_err = nan(nM,nC); count_n = zeros(nM,nC);
    for m = 1:nM
        for c = 1:nC
            mask = strcmp(cellstr(string(T.method)), methods{m}) & strcmp(cellstr(string(T.criterion_clean)), clean_levels{c});
            v = T.K_error(mask); v = v(~isnan(v));
            if ~isempty(v), mean_err(m,c) = mean(v,'omitnan'); count_n(m,c) = numel(v); end
        end
    end
    fig = figure('Visible','off','Position',[200 200 1200 800], 'Color', 'white');
    imagesc(mean_err); colormap(redbluecmap()); h = colorbar; 
    set(h, 'XColor', 'black', 'YColor', 'black');
    axis tight;
    set(gca,'XTick',1:nC,'XTickLabel',clean_levels,'XTickLabelRotation',45,'YTick',1:nM,...
            'YTickLabel',cellfun(@display_method_label,methods,'UniformOutput',false), ...
            'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    title('Average Signed K Error (K_{est} - K_{true})', 'FontSize', 12, 'FontWeight', 'bold');
    for m = 1:nM
        for c = 1:nC
            if ~isnan(mean_err(m,c)), text(c,m,sprintf('%.2f\n(n=%d)',mean_err(m,c),count_n(m,c)),...
                'HorizontalAlignment','center','Color','white', 'FontSize', 10, 'FontWeight', 'bold'); end
        end
    end
    saveas(fig, fullfile(plots_dir,'avg_k_error_heatmap.png'));
    close(fig);
end

function create_abs_k_error_plot(T, plots_dir)
    global ANALYSIS_CONFIG;
    if ~ismember('K_abs_error', T.Properties.VariableNames)
        warning('No K_abs_error column found; skipping absolute K error bar plot.'); return;
    end
    methods = get_factor_levels(T, 'method');
    n_methods = numel(methods);
    n_boot = ANALYSIS_CONFIG.n_boot;
    means = nan(n_methods,1); ci_low = nan(n_methods,1); ci_high = nan(n_methods,1); counts = zeros(n_methods,1);
    for m = 1:n_methods
        mname = methods{m};
        mask = strcmp(cellstr(string(T.method)), mname);
        vals = T.K_abs_error(mask); vals = vals(~isnan(vals));
        counts(m) = numel(vals);
        if isempty(vals), means(m)=NaN; ci_low(m)=NaN; ci_high(m)=NaN; continue; end
        means(m) = mean(vals,'omitnan');
        boot_means = nan(n_boot,1);
        for b = 1:n_boot
            sample = datasample(vals, numel(vals));
            boot_means(b) = mean(sample,'omitnan');
        end
        ci_low(m) = prctile(boot_means,2.5); ci_high(m) = prctile(boot_means,97.5);
    end
    fig = figure('Visible','off','Position',[200 200 1000 600], 'Color', 'white');
    bar(1:n_methods, means, 'FaceColor',[0.4 0.6 0.8], 'EdgeColor', 'black', 'LineWidth', 1.5); 
    hold on;
    errlow = means - ci_low; errhigh = ci_high - means;
    errorbar(1:n_methods, means, errlow, errhigh, 'k.', 'LineWidth', 2, 'CapSize', 12);
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', cellfun(@display_method_label, methods, 'UniformOutput', false), ...
        'XTickLabelRotation', 45, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    ylabel('Mean Absolute K Error', 'FontSize', 11, 'FontWeight', 'bold'); 
    title('Absolute K Error by Method (mean ± 95% CI)', 'FontSize', 12, 'FontWeight', 'bold'); 
    grid on;
    set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    for m = 1:n_methods
        if ~isnan(means(m)), text(m, means(m) + max(errhigh)*0.05, sprintf('n=%d', counts(m)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9); end
    end
    saveas(fig, fullfile(plots_dir, 'abs_k_error_by_method.png'));
    close(fig);
end

function cmap = redbluecmap()
    n = 256;
    r = [(0:(n/2-1))/(n/2-1), ones(1,n/2)];
    g = [(0:(n/2-1))/(n/2-1), fliplr((0:(n/2-1))/(n/2-1))];
    b = [ones(1,n/2), fliplr((0:(n/2-1))/(n/2-1))];
    cmap = [r(:), g(:), b(:)];
end

function s_out = canonicalize_criterion(s_in)
    if isempty(s_in), s_out=''; return; end
    s = char(string(s_in));
    s = lower(s);
    s = strrep(s, '_', ' ');
    s = regexprep(s, '\s+', ' ');
    s = strtrim(s);
    
    % Bare "elbow" refers to the free-energy elbow in the simulation outputs.
    if strcmp(s, 'elbow')
        s = 'free energy elbow';
    end
    
    if any(strcmp(s, {'gfp', 'global field power', 'global explained variance'}))
        s = 'gev';
    end
    if contains(s,'elbow sil') || contains(s, 'free energy elbow sil')
        s = 'elbow sil combined';
    elseif strcmp(s,'free energy elbow only')
        s = 'free energy elbow';
    elseif strcmp(s,'silhouette only')
        s = 'silhouette';
    elseif any(strcmp(s, {'covariance raw', 'covariance min'}))
        s = 'covariance';
    elseif any(strcmp(s, {'calinski harabasz', 'ch'}))
        s = 'calinski harabasz score';
    end
    s = regexprep(s, '(\b\w+\b)(\s+\1)+', '$1');
    s_out = s;
end

function m = canonicalize_method(m_in)
    if isempty(m_in), m = ''; return; end
    m = char(string(m_in));
    m = lower(m);
    m = strrep(m, '_', ' ');
    m = regexprep(m, '\s+', ' ');
    m = strtrim(m);
    if contains(m, 'spm vb')
        m = 'spm vb';
    elseif contains(m, 'spm kmeans')
        m = 'spm kmeans';
    elseif contains(m, 'kmeans koenig') || contains(m, 'koenig kmeans')
        m = 'koenig kmeans';
    elseif contains(m, 'kmeans standard') || contains(m, 'standard kmeans')
        m = 'standard kmeans';
    end
end

function label = display_method_label(m_in)
    switch canonicalize_method(m_in)
        case 'spm vb'
            label = 'SPM-VB';
        case 'koenig kmeans'
            label = 'Koenig k-means';
        case 'standard kmeans'
            label = 'Standard k-means';
        otherwise
            label = canonicalize_method(m_in);
    end
end

function quintile_idx = rank_based_quintile_groups(values, n_groups)
    quintile_idx = nan(size(values));
    if isempty(values)
        return;
    end
    [~, sort_idx] = sort(values(:), 'ascend');
    n = numel(sort_idx);
    rank_groups = ceil((1:n) * n_groups / n);
    rank_groups(rank_groups < 1) = 1;
    rank_groups(rank_groups > n_groups) = n_groups;
    quintile_idx(sort_idx) = rank_groups;
    quintile_idx = reshape(quintile_idx, size(values));
end

function label = covariance_quintile_label(idx, n_groups)
    if idx == 1
        label = 'lowest covariance';
    elseif idx == n_groups
        label = 'highest covariance';
    else
        label = 'mid covariance';
    end
end

function json_file = resolve_json_path_local(raw_path, output_dir, json_dir)
    raw_path = char(string(raw_path));
    if isempty(strtrim(raw_path))
        json_file = '';
        return;
    end
    if isfile(raw_path)
        json_file = raw_path;
        return;
    end
    candidate = fullfile(output_dir, raw_path);
    if isfile(candidate)
        json_file = candidate;
        return;
    end
    candidate = fullfile(json_dir, raw_path);
    if isfile(candidate)
        json_file = candidate;
        return;
    end
    [~, base, ext] = fileparts(raw_path);
    candidate = fullfile(json_dir, [base ext]);
    if isfile(candidate)
        json_file = candidate;
    else
        json_file = raw_path;
    end
end

function [maps, channel_labels] = extract_json_state_maps_local(data, field_name)
    if ~isfield(data, field_name) || isempty(data.(field_name))
        error('JSON payload is missing %s.', field_name);
    end
    state_struct = data.(field_name);
    state_names = fieldnames(state_struct);
    state_order = zeros(numel(state_names), 1);
    for i = 1:numel(state_names)
        tok = regexp(state_names{i}, '^state_(\d+)$', 'tokens', 'once');
        if isempty(tok)
            state_order(i) = i;
        else
            state_order(i) = str2double(tok{1});
        end
    end
    [~, sort_idx] = sort(state_order);
    state_names = state_names(sort_idx);

    channel_labels = cellstr(string(data.channel_info.labels));
    if isfield(data.channel_info, 'labels_sanitized')
        channel_keys = cellstr(string(data.channel_info.labels_sanitized));
    else
        channel_keys = sanitize_channel_labels_plot_local(channel_labels);
    end

    n_states = numel(state_names);
    n_channels = numel(channel_labels);
    maps = nan(n_states, n_channels);
    for s = 1:n_states
        state_data = state_struct.(state_names{s});
        for c = 1:n_channels
            if isfield(state_data, channel_keys{c})
                maps(s, c) = double(state_data.(channel_keys{c}));
            end
        end
    end
end

function projected_maps = project_maps_to_template_channels_local(maps, source_channel_labels, template_channel_labels)
    projected_maps = nan(size(maps, 1), numel(template_channel_labels));
    source_labels_l = cellfun(@(s) lower(strtrim(char(s))), cellstr(source_channel_labels(:)), 'UniformOutput', false);
    template_labels_l = cellfun(@(s) lower(strtrim(char(s))), cellstr(template_channel_labels(:)), 'UniformOutput', false);
    [lia, locb] = ismember(template_labels_l, source_labels_l);
    for c = 1:numel(template_channel_labels)
        if lia(c)
            projected_maps(:, c) = maps(:, locb(c));
        end
    end
end

function maps_out = normalize_map_rows_omitnan_local(maps_in)
    maps_out = maps_in;
    for r = 1:size(maps_in, 1)
        row = maps_in(r, :);
        finite_mask = isfinite(row);
        if nnz(finite_mask) < 2
            continue;
        end
        vals = row(finite_mask);
        vals = vals - mean(vals);
        denom = norm(vals);
        if denom > eps
            maps_out(r, finite_mask) = vals ./ denom;
        else
            maps_out(r, finite_mask) = vals;
        end
    end
end

function sanitized = sanitize_channel_labels_plot_local(ch_labels)
    sanitized = cell(size(ch_labels));
    for i = 1:length(ch_labels)
        label = char(ch_labels{i});
        label = regexprep(label, '[-/\\\s\.\,\(\)\[\]\{\}]', '_');
        label = regexprep(label, '^_+|_+$', '');
        if isempty(label) || isempty(regexp(label(1), '[A-Za-z]', 'once'))
            label = ['Ch' label];
        end
        sanitized{i} = matlab.lang.makeValidName(label);
    end
end

function s = sanitize_stub(s)
    s = lower(char(string(s)));
    s = regexprep(s, '[^a-z0-9]+', '_');
    s = regexprep(s, '^_+|_+$', '');
end

function r = ifthenelse(cond, true_val, false_val)
    if cond, r = true_val; else r = false_val; end
end

function create_criterion_montage_boxplots(T, plots_dir)
    % Creates boxplots of F1 and absolute K error for each criterion
    % at all montage levels, for SPM and k-means methods, with significance brackets
    
    global ANALYSIS_CONFIG;
    if isempty(ANALYSIS_CONFIG)
        ANALYSIS_CONFIG.n_boot = 10000;
    end
    
    % Filter for the supported methods only
    method_col = cellstr(string(T.method));
    method_canon = cellfun(@canonicalize_method, method_col, 'UniformOutput', false);
    method_filter = ismember(method_canon, {'koenig kmeans', 'spm vb'});
    T_filtered = T(method_filter, :);
    
    if height(T_filtered) == 0
        fprintf('⚠ No SPM or k-means data found\n');
        return;
    end
    
    % Get unique criteria and montages
    criteria = unique(cellstr(string(T_filtered.criterion_clean)));
    montages = unique(T_filtered.montage_type);
    
    fprintf('  Found %d criteria: %s\n', length(criteria), strjoin(criteria, ', '));
    fprintf('  Found %d montages: %s\n', length(montages), strjoin(cellstr(montages), ', '));
    
    % Get lead counts for labeling
    lead_counts = zeros(size(montages));
    for i = 1:length(montages)
        idx = find(strcmp(T_filtered.montage_type, montages{i}), 1);
        if ~isempty(idx)
            lead_counts(i) = T_filtered.n_leads(idx);
        end
    end
    
    % Sort by lead count
    [lead_counts, sort_idx] = sort(lead_counts);
    montages = montages(sort_idx);
    
    % Create montage labels with lead counts
    montage_labels = cell(size(montages));
    for i = 1:length(montages)
        montage_labels{i} = sprintf('%s(%dch)', montages{i}, lead_counts(i));
    end
    
    % Identify methods
    method_canon_filtered = method_canon(method_filter);
    unique_methods = unique(method_canon_filtered, 'stable');

    spm_vb_idx = find(strcmp(unique_methods, 'spm vb'), 1);
    kmeans_idx = find(strcmp(unique_methods, 'koenig kmeans'), 1);

    if isempty(spm_vb_idx) || isempty(kmeans_idx)
        fprintf('⚠ Need both SPM-VB and Koenig k-means data\n');
        fprintf('   Available methods: %s\n', strjoin(unique_methods, ', '));
        return;
    end

    spm_vb_method = unique_methods{spm_vb_idx};
    kmeans_method = unique_methods{kmeans_idx};

    fprintf('  Using methods: %s vs %s\n', spm_vb_method, kmeans_method);
    
    % ========== FIGURE 1: Absolute K Error by Criterion and Montage ==========
    for method_idx = 1:2
        if method_idx == 1
            method = spm_vb_method;
            method_label = display_method_label(method);
        else
            method = kmeans_method;
            method_label = display_method_label(method);
        end
        
        fig = figure('Position', [100 100 1400 600]);
        sgtitle(sprintf('Absolute K Error by Criterion and Montage - %s', method_label), ...
                'FontSize', 16, 'FontWeight', 'bold');
        
        for c = 1:length(criteria)
            crit = criteria{c};
            
            subplot(2, ceil(length(criteria)/2), c);
            hold on;
            box on;
            
            % Set title first so it appears even if no data
            crit_display = strrep(crit, '_', ' ');
            title(crit_display, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
            
            % Prepare data for this criterion
            data_groups = [];
            group_labels = {};
            montage_data = cell(length(montages), 1);
            
            for m = 1:length(montages)
                mont = montages{m};
                idx = strcmp(cellstr(string(T_filtered.method)), method) & ...
                      strcmp(cellstr(string(T_filtered.criterion_clean)), crit) & ...
                      strcmp(T_filtered.montage_type, mont);
                
                if any(idx)
                    vals = T_filtered.K_abs_error(idx);
                    vals = vals(~isnan(vals));
                    montage_data{m} = vals;
                    
                    if ~isempty(vals)
                        data_groups = [data_groups; vals]; %#ok<AGROW>
                        group_labels = [group_labels; repmat(montage_labels(m), length(vals), 1)]; %#ok<AGROW>
                    end
                end
            end
            
            if ~isempty(data_groups)
                h = boxplot(data_groups, group_labels, 'Colors', [0.2 0.4 0.8], 'LabelOrientation', 'inline');
                ylabel('Absolute K Error', 'FontSize', 11, 'FontWeight', 'bold');
                xlabel('');
                grid on;
                ylim_curr = ylim;
                ylim([0, ylim_curr(2)]);
                set(gca, 'XTickLabelRotation', 0);
                
                % Add significance brackets (compare adjacent montages)
                if length(montages) > 1
                    y_max = max(data_groups);
                    bracket_height = y_max * 0.05;
                    current_y = y_max * 1.05;
                    
                    for m = 1:(length(montages)-1)
                        data1 = montage_data{m};
                        data2 = montage_data{m+1};
                        
                        if ~isempty(data1) && ~isempty(data2) && length(data1) > 1 && length(data2) > 1
                            % Bootstrap test
                            n_boot = ANALYSIS_CONFIG.n_boot;
                            diff_boot = zeros(n_boot, 1);
                            for b = 1:n_boot
                                boot1 = mean(datasample(data1, length(data1)));
                                boot2 = mean(datasample(data2, length(data2)));
                                diff_boot(b) = boot1 - boot2;
                            end
                            
                            % Two-tailed test
                            p_val = 2 * min(mean(diff_boot >= 0), mean(diff_boot <= 0));
                            
                            % Determine significance
                            if p_val < 0.001
                                sig_label = '***';
                            elseif p_val < 0.01
                                sig_label = '**';
                            elseif p_val < 0.05
                                sig_label = '*';
                            else
                                sig_label = '';
                            end
                            
                            % Draw bracket if significant
                            if ~isempty(sig_label)
                                pos1 = m;
                                pos2 = m + 1;
                                
                                plot([pos1, pos1], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1);
                                plot([pos1, pos2], [current_y + bracket_height, current_y + bracket_height], 'k-', 'LineWidth', 1);
                                plot([pos2, pos2], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1);
                                text((pos1 + pos2)/2, current_y + bracket_height * 1.3, sig_label, ...
                                     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                                
                                current_y = current_y + bracket_height * 3;
                            end
                        end
                    end
                    
                    % Adjust y-axis to fit brackets
                    ylim_new = ylim;
                    if current_y > ylim_new(2)
                        ylim([ylim_new(1), current_y + bracket_height]);
                    end
                end
            else
                text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center', ...
                     'FontSize', 14, 'Color', [0.5 0.5 0.5]);
                axis off;
            end
        end
        
        saveas(fig, fullfile(plots_dir, sprintf('abs_k_error_criterion_montage_%s.png', ...
               strrep(lower(method_label), ' ', '_'))));
        close(fig);
    end
    
    % ========== FIGURE 2: F1 Score by Criterion and Montage ==========
    for method_idx = 1:2
        if method_idx == 1
            method = spm_vb_method;
            method_label = display_method_label(method);
        else
            method = kmeans_method;
            method_label = display_method_label(method);
        end
        
        fig = figure('Position', [100 100 1400 600]);
        sgtitle(sprintf('F1 Score by Criterion and Montage - %s', method_label), ...
                'FontSize', 16, 'FontWeight', 'bold');
        
        for c = 1:length(criteria)
            crit = criteria{c};
            
            subplot(2, ceil(length(criteria)/2), c);
            hold on;
            box on;
            
            % Set title first so it appears even if no data
            crit_display = strrep(crit, '_', ' ');
            title(crit_display, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
            
            % Prepare data for this criterion
            data_groups = [];
            group_labels = {};
            montage_data = cell(length(montages), 1);
            
            for m = 1:length(montages)
                mont = montages{m};
                idx = strcmp(cellstr(string(T_filtered.method)), method) & ...
                      strcmp(cellstr(string(T_filtered.criterion_clean)), crit) & ...
                      strcmp(T_filtered.montage_type, mont);
                
                if any(idx)
                    vals = T_filtered.f1_score(idx);
                    vals = vals(~isnan(vals));
                    montage_data{m} = vals;
                    
                    if ~isempty(vals)
                        data_groups = [data_groups; vals]; %#ok<AGROW>
                        group_labels = [group_labels; repmat(montage_labels(m), length(vals), 1)]; %#ok<AGROW>
                    end
                end
            end
            
            if ~isempty(data_groups)
                h = boxplot(data_groups, group_labels, 'Colors', [0.8 0.4 0.2], 'LabelOrientation', 'inline');
                ylabel('F1 Score', 'FontSize', 11, 'FontWeight', 'bold');
                xlabel('');
                grid on;
                ylim([0, 1]);
                set(gca, 'XTickLabelRotation', 0);
                
                % Add significance brackets (compare adjacent montages)
                if length(montages) > 1
                    bracket_height = 0.04;
                    current_y = 0.85;
                    
                    for m = 1:(length(montages)-1)
                        data1 = montage_data{m};
                        data2 = montage_data{m+1};
                        
                        if ~isempty(data1) && ~isempty(data2) && length(data1) > 1 && length(data2) > 1
                            % Bootstrap test
                            n_boot = ANALYSIS_CONFIG.n_boot;
                            diff_boot = zeros(n_boot, 1);
                            for b = 1:n_boot
                                boot1 = mean(datasample(data1, length(data1)));
                                boot2 = mean(datasample(data2, length(data2)));
                                diff_boot(b) = boot1 - boot2;
                            end
                            
                            % Two-tailed test
                            p_val = 2 * min(mean(diff_boot >= 0), mean(diff_boot <= 0));
                            
                            % Determine significance
                            if p_val < 0.001
                                sig_label = '***';
                            elseif p_val < 0.01
                                sig_label = '**';
                            elseif p_val < 0.05
                                sig_label = '*';
                            else
                                sig_label = '';
                            end
                            
                            % Draw bracket if significant
                            if ~isempty(sig_label)
                                pos1 = m;
                                pos2 = m + 1;
                                
                                plot([pos1, pos1], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1);
                                plot([pos1, pos2], [current_y + bracket_height, current_y + bracket_height], 'k-', 'LineWidth', 1);
                                plot([pos2, pos2], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1);
                                text((pos1 + pos2)/2, current_y + bracket_height * 1.3, sig_label, ...
                                     'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                                
                                current_y = current_y + bracket_height * 2.5;
                            end
                        end
                    end
                    
                    % Adjust y-axis to fit brackets
                    if current_y > 0.9
                        ylim([0, min(1.0, current_y + bracket_height * 1.5)]);
                    end
                end
            else
                text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center', ...
                     'FontSize', 14, 'Color', [0.5 0.5 0.5]);
                axis off;
            end
        end
        
        saveas(fig, fullfile(plots_dir, sprintf('f1_score_criterion_montage_%s.png', ...
               strrep(lower(method_label), ' ', '_'))));
        close(fig);
    end
    
    fprintf('✓ Criterion-montage boxplots saved\n');
end


function create_method_criterion_boxplots(T, plots_dir)
    % Creates box-and-whisker plots comparing absolute K error and F1 score
    % across different criteria for SPM and k-means methods, with significance brackets
    
    global ANALYSIS_CONFIG;
    if isempty(ANALYSIS_CONFIG)
        ANALYSIS_CONFIG.n_boot = 10000;
    end
    
    % Filter for SPM and k-means methods only
    method_col = cellstr(string(T.method));
    method_filter = contains(method_col, 'spm', 'IgnoreCase', true) | ...
                    contains(method_col, 'kmeans', 'IgnoreCase', true);
    T_filtered = T(method_filter, :);
    
    if height(T_filtered) == 0
        fprintf('⚠ No SPM or k-means data found for method-criterion boxplots\n');
        fprintf('   Available methods: %s\n', strjoin(unique(method_col), ', '));
        return;
    end
    
    fprintf('Creating method-criterion comparison boxplots...\n');
    
    % Get unique criteria
    criteria = unique(cellstr(string(T_filtered.criterion_clean)));
    
    % Identify actual method names in data (defensive in case one is missing)
    unique_methods = unique(cellstr(string(T_filtered.method)));
    spm_idx = find(contains(unique_methods, 'spm', 'IgnoreCase', true), 1);
    kmeans_idx = find(contains(unique_methods, 'kmeans', 'IgnoreCase', true), 1);
    if isempty(spm_idx) || isempty(kmeans_idx)
        fprintf('ƒsÿ Expected both SPM and k-means methods, found: %s\n', strjoin(unique_methods, ', '));
        return;
    end
    spm_method = unique_methods{spm_idx};
    kmeans_method = unique_methods{kmeans_idx};
    
    methods = {spm_method, kmeans_method};
    method_labels = {display_method_label(spm_method), display_method_label(kmeans_method)};
    
    % Create figure with two subplots
    fig = figure('Position', [100 100 1400 600]);
    
    % ========== SUBPLOT 1: Absolute K Error ==========
    subplot(1, 2, 1);
    hold on;
    
    % Prepare data and group labels
    data_k_error = [];
    group_labels_k = {};
    group_positions = [];
    tick_labels = {};
    pos = 0;
    
    % Organize data by criterion, then method within each criterion
    for c = 1:length(criteria)
        crit = criteria{c};
        for m = 1:length(methods)
            meth = methods{m};
            idx = strcmp(cellstr(string(T_filtered.method)), meth) & ...
                  strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
            
            if any(idx)
                pos = pos + 1;
                vals = T_filtered.K_abs_error(idx);
                vals = vals(~isnan(vals));
                
                data_k_error = [data_k_error; vals]; %#ok<AGROW>
                group_labels_k = [group_labels_k; repmat({sprintf('%s_%s', crit, meth)}, length(vals), 1)]; %#ok<AGROW>
                group_positions(end+1) = pos; %#ok<AGROW>
                
                if m == 1
                    tick_labels{end+1} = sprintf('%s\n%s', crit, method_labels{m}); %#ok<AGROW>
                else
                    tick_labels{end+1} = method_labels{m}; %#ok<AGROW>
                end
            end
        end
        % Add gap between criteria
        pos = pos + 0.5;
    end
    
    % Create boxplot
    positions = group_positions;
    boxplot(data_k_error, group_labels_k, 'Positions', positions, ...
            'Colors', repmat([0.2 0.4 0.8; 0.8 0.4 0.2], ceil(length(positions)/2), 1), ...
            'Widths', 0.6);
    
    % Customize appearance
    set(gca, 'XTick', positions, 'XTickLabel', tick_labels, 'FontSize', 10);
    ylabel('Absolute K Error', 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Criterion and Method', 'FontSize', 13, 'FontWeight', 'bold');
    title('Absolute K Error by Criterion and Method', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim_curr = ylim;
    ylim([0, ylim_curr(2)]);
    
    % Add significance brackets (compare methods within each criterion)
    y_max = max(data_k_error);
    bracket_height = y_max * 0.05;
    current_y = y_max * 1.1;
    
    sig_comparisons = {};
    pos_idx = 1;
    for c = 1:length(criteria)
        crit = criteria{c};
        
        % Get data for both methods
        data_spm = [];
        data_kmeans = [];
        
        idx_spm = strcmp(cellstr(string(T_filtered.method)), spm_method) & ...
                  strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
        idx_kmeans = strcmp(cellstr(string(T_filtered.method)), kmeans_method) & ...
                     strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
        
        if any(idx_spm)
            data_spm = T_filtered.K_abs_error(idx_spm);
            data_spm = data_spm(~isnan(data_spm));
        end
        if any(idx_kmeans)
            data_kmeans = T_filtered.K_abs_error(idx_kmeans);
            data_kmeans = data_kmeans(~isnan(data_kmeans));
        end
        
        % Perform bootstrap test if both methods have data
        if ~isempty(data_spm) && ~isempty(data_kmeans)
            % Bootstrap test
            n_boot = ANALYSIS_CONFIG.n_boot;
            diff_boot = zeros(n_boot, 1);
            for b = 1:n_boot
                boot_spm = mean(datasample(data_spm, length(data_spm)));
                boot_kmeans = mean(datasample(data_kmeans, length(data_kmeans)));
                diff_boot(b) = boot_spm - boot_kmeans;
            end
            
            % Two-tailed test
            p_val = 2 * min(mean(diff_boot >= 0), mean(diff_boot <= 0));
            
            % Determine significance
            if p_val < 0.001
                sig_label = '***';
            elseif p_val < 0.01
                sig_label = '**';
            elseif p_val < 0.05
                sig_label = '*';
            else
                sig_label = 'ns';
            end
            
            % Draw bracket if significant
            if ~strcmp(sig_label, 'ns')
                pos1 = positions(pos_idx);
                pos2 = positions(pos_idx + 1);
                
                plot([pos1, pos1], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1.5);
                plot([pos1, pos2], [current_y + bracket_height, current_y + bracket_height], 'k-', 'LineWidth', 1.5);
                plot([pos2, pos2], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1.5);
                text((pos1 + pos2)/2, current_y + bracket_height * 1.5, sig_label, ...
                     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
                
                current_y = current_y + bracket_height * 4;
            end
            
            sig_comparisons{end+1} = sprintf('%s: p=%.4f (%s)', crit, p_val, sig_label); %#ok<AGROW>
            pos_idx = pos_idx + 2;
        else
            pos_idx = pos_idx + sum([any(idx_spm), any(idx_kmeans)]);
        end
    end
    
    % Adjust y-axis to fit brackets
    ylim_new = ylim;
    ylim([ylim_new(1), current_y + bracket_height]);
    
    % ========== SUBPLOT 2: F1 Score ==========
    subplot(1, 2, 2);
    hold on;
    
    % Prepare data and group labels
    data_f1 = [];
    group_labels_f1 = {};
    group_positions = [];
    tick_labels = {};
    pos = 0;
    
    % Organize data by criterion, then method within each criterion
    for c = 1:length(criteria)
        crit = criteria{c};
        for m = 1:length(methods)
            meth = methods{m};
            idx = strcmp(cellstr(string(T_filtered.method)), meth) & ...
                  strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
            
            if any(idx)
                pos = pos + 1;
                vals = T_filtered.f1_score(idx);
                vals = vals(~isnan(vals));
                
                data_f1 = [data_f1; vals]; %#ok<AGROW>
                group_labels_f1 = [group_labels_f1; repmat({sprintf('%s_%s', crit, meth)}, length(vals), 1)]; %#ok<AGROW>
                group_positions(end+1) = pos; %#ok<AGROW>
                
                if m == 1
                    tick_labels{end+1} = sprintf('%s\n%s', crit, method_labels{m}); %#ok<AGROW>
                else
                    tick_labels{end+1} = method_labels{m}; %#ok<AGROW>
                end
            end
        end
        % Add gap between criteria
        pos = pos + 0.5;
    end
    
    % Create boxplot
    positions = group_positions;
    boxplot(data_f1, group_labels_f1, 'Positions', positions, ...
            'Colors', repmat([0.2 0.4 0.8; 0.8 0.4 0.2], ceil(length(positions)/2), 1), ...
            'Widths', 0.6);
    
    % Customize appearance
    set(gca, 'XTick', positions, 'XTickLabel', tick_labels, 'FontSize', 10);
    ylabel('F1 Score', 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Criterion and Method', 'FontSize', 13, 'FontWeight', 'bold');
    title('F1 Score by Criterion and Method', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0, 1]);
    
    % Add significance brackets (compare methods within each criterion)
    y_max = 1.0;
    bracket_height = 0.03;
    current_y = 0.85;
    
    sig_comparisons_f1 = {};
    pos_idx = 1;
    for c = 1:length(criteria)
        crit = criteria{c};
        
        % Get data for both methods
        data_spm = [];
        data_kmeans = [];
        
        idx_spm = strcmp(cellstr(string(T_filtered.method)), spm_method) & ...
                  strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
        idx_kmeans = strcmp(cellstr(string(T_filtered.method)), kmeans_method) & ...
                     strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
        
        if any(idx_spm)
            data_spm = T_filtered.f1_score(idx_spm);
            data_spm = data_spm(~isnan(data_spm));
        end
        if any(idx_kmeans)
            data_kmeans = T_filtered.f1_score(idx_kmeans);
            data_kmeans = data_kmeans(~isnan(data_kmeans));
        end
        
        % Perform bootstrap test if both methods have data
        if ~isempty(data_spm) && ~isempty(data_kmeans)
            % Bootstrap test
            n_boot = ANALYSIS_CONFIG.n_boot;
            diff_boot = zeros(n_boot, 1);
            for b = 1:n_boot
                boot_spm = mean(datasample(data_spm, length(data_spm)));
                boot_kmeans = mean(datasample(data_kmeans, length(data_kmeans)));
                diff_boot(b) = boot_spm - boot_kmeans;
            end
            
            % Two-tailed test
            p_val = 2 * min(mean(diff_boot >= 0), mean(diff_boot <= 0));
            
            % Determine significance
            if p_val < 0.001
                sig_label = '***';
            elseif p_val < 0.01
                sig_label = '**';
            elseif p_val < 0.05
                sig_label = '*';
            else
                sig_label = 'ns';
            end
            
            % Draw bracket if significant
            if ~strcmp(sig_label, 'ns')
                pos1 = positions(pos_idx);
                pos2 = positions(pos_idx + 1);
                
                plot([pos1, pos1], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1.5);
                plot([pos1, pos2], [current_y + bracket_height, current_y + bracket_height], 'k-', 'LineWidth', 1.5);
                plot([pos2, pos2], [current_y, current_y + bracket_height], 'k-', 'LineWidth', 1.5);
                text((pos1 + pos2)/2, current_y + bracket_height * 1.5, sig_label, ...
                     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
                
                current_y = current_y + bracket_height * 4;
            end
            
            sig_comparisons_f1{end+1} = sprintf('%s: p=%.4f (%s)', crit, p_val, sig_label); %#ok<AGROW>
            pos_idx = pos_idx + 2;
        else
            pos_idx = pos_idx + sum([any(idx_spm), any(idx_kmeans)]);
        end
    end
    
    % Adjust y-axis to fit brackets
    if current_y > 0.85
        ylim([0, min(1.0, current_y + bracket_height * 2)]);
    end
    
    % Save figure
    saveas(fig, fullfile(plots_dir, 'method_criterion_comparison_boxplots.png'));
    close(fig);
    
    fprintf('✓ Method-criterion comparison boxplots saved\n');
    fprintf('  Absolute K Error significance tests:\n');
    for i = 1:length(sig_comparisons)
        fprintf('    %s\n', sig_comparisons{i});
    end
    fprintf('  F1 Score significance tests:\n');
    for i = 1:length(sig_comparisons_f1)
        fprintf('    %s\n', sig_comparisons_f1{i});
    end
end
