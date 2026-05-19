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
    default_results_dir = 'Output/results';
    default_n_boot = 2000;
    default_n_boot_lmm = 2000;
    default_n_folds = 20;

    % ---------- Set up publication-style plot ----------
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

    % ---------- Outcomes ----------
    % Note: runtime_s is excluded from main comparison plots and handled separately
    outcomes = {'K_correct', 'f1_score', 'sensitivity', 'precision', 'mean_recovery_matched', 'K_abs_error'};
    outcome_labels = {'K_{true} Selection Accuracy', 'F1 Score', 'Sensitivity', 'Precision', 'Mean Matched Correlation', 'Absolute K Error'};
    
    % Full outcomes list (including runtime) for LMM analysis
    outcomes_full = {'K_correct', 'f1_score', 'sensitivity', 'precision', 'runtime_s', 'mean_recovery_matched', 'K_abs_error'};
    outcome_labels_full = {'K_{true} Selection Accuracy', 'F1 Score', 'Sensitivity', 'Precision', 'Runtime (s)', 'Mean Matched Correlation', 'Absolute K Error'};

    % ---------- Analyses ----------
    fprintf('1) METHOD effects\n');    method_results = analyze_factor_effects_with_ci(T, 'method', outcomes_full, fid);
    fprintf('2) CRITERION effects\n'); criterion_results = analyze_factor_effects_with_ci_using_clean(T, 'criterion_clean', outcomes_full, fid);
    fprintf('3) SNR effects\n'); snr_results = analyze_snr_effects(T, outcomes_full, fid);
    fprintf('4) MONTAGE effects\n'); montage_results = analyze_factor_effects_with_ci(T, 'montage_type', outcomes_full, fid);
    fprintf('5) Interaction METHOD × CRITERION\n'); interaction_results = analyze_interaction(T, outcomes_full, fid);

    % Cross-validation
    fprintf('6) Cross-validation (adaptive)\n'); cv_results = analyze_cross_validation_adaptive(T, outcomes_full, ANALYSIS_CONFIG.n_folds, fid);

    % Bootstrapped LMMs (cluster bootstrap by subject) including interaction
    fprintf('7) Bootstrapped Linear Mixed Models (LMM)\n');
    fprintf(fid, '\n========================================\nLMM Analysis (cluster bootstrap by subject)\nRandom intercept: subject\nFixed effects: method * criterion_clean + SNR_dB\n========================================\n\n');
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

    fclose(fid);
    fprintf('Analysis saved to %s and plots under %s\n', stats_file, plots_dir);

    % ---------- Plots (dynamic grids) ----------
    fprintf('Generating plots...\n');
    % Use outcomes without runtime for these plots
    create_boxplot_comparison(T, outcomes, outcome_labels, plots_dir);
    create_method_effects_plot_with_ci(T, outcomes, outcome_labels, method_results, plots_dir);
    create_criterion_effects_plot_with_ci(T, outcomes, outcome_labels, criterion_results, plots_dir);
    create_snr_effects_plot(T, outcomes, outcome_labels, plots_dir);
    create_interaction_plot(T, outcomes, outcome_labels, plots_dir);
    create_cross_validation_plot(cv_results, outcomes, outcome_labels, plots_dir);
    
    % Separate runtime plot (SNR effects only)
    create_runtime_snr_plot(T, plots_dir);
    
    create_avg_k_error_plot(T, plots_dir);
    create_abs_k_error_plot(T, plots_dir);
    
    % Method-Criterion comparison boxplots with significance
    create_method_criterion_boxplots(T, plots_dir);

    % ---------- Montage-Criterion Comparison (if montage data available) ----------
    montages = unique(T.montage_type);
    if length(montages) > 1
        fprintf('Generating criterion-montage comparison boxplots...\n');
        create_criterion_montage_boxplots(T, plots_dir);
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
    formula = sprintf('%s ~ 1 + method*criterion_clean + SNR_dB + (1|subject)', outcome_var);
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
        set(gca,'XTick',1:numel(methods),'XTickLabel',cellfun(@(s)strrep(s,'_',' '),methods,'UniformOutput',false),'XTickLabelRotation',45);
        set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
        ylabel(outcome_labels{i}, 'FontSize', 11, 'FontWeight', 'bold'); 
        title(outcome_labels{i}, 'FontSize', 11); 
        grid on;
        set(gca, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.5);
    end
    sgtitle('Method effects (aggregated over criteria)', 'FontSize', 13, 'FontWeight', 'bold');
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
    sgtitle('Criterion effects', 'FontSize', 13, 'FontWeight', 'bold');
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
                 'DisplayName',strrep(mname,'_',' '));
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
                 'MarkerSize', 8, 'DisplayName',strrep(methods{m},'_',' '));
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
                label = sprintf('%s - %s', strrep(mname,'_',' '), strrep(clev,'_',' '));
                
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
            'YTickLabel',cellfun(@(s)strrep(s,'_',' '),methods,'UniformOutput',false), ...
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
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', cellfun(@(s)strrep(s,'_',' '), methods, 'UniformOutput', false), ...
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
    
    % Replace "elbow" with "free energy elbow"
    if contains(s, 'elbow') && ~contains(s, 'free energy')
        s = strrep(s, 'elbow', 'free energy elbow');
    end
    
    if contains(s,'elbow sil') || contains(s, 'free energy elbow sil'), s = 'elbow sil combined';
    elseif strcmp(s,'free energy elbow only'), s = 'free energy elbow';
    elseif strcmp(s,'silhouette only'), s = 'silhouette'; end
    s = regexprep(s, '(\b\w+\b)(\s+\1)+', '$1');
    s_out = s;
end

function canonicalize_method(m_in)
    if isempty(m_in), return; end
    m = char(string(m_in));
    m = lower(m);
    m = strrep(m, '_', ' ');
    m = regexprep(m, '\s+', ' ');
    m = strtrim(m);
    % Replace "kmeans koenig" with "kmeans standard"
    if contains(m, 'kmeans koenig')
        m = strrep(m, 'kmeans koenig', 'kmeans standard');
    end
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
    
    % Filter for SPM and k-means methods only
    method_col = cellstr(string(T.method));
    method_filter = contains(method_col, 'spm', 'IgnoreCase', true) | ...
                    contains(method_col, 'kmeans', 'IgnoreCase', true);
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
    unique_methods = unique(cellstr(string(T_filtered.method)));
    
    % Look for spm-k-means (combined method)
    spm_kmeans_idx = find(contains(unique_methods, 'spm', 'IgnoreCase', true) & contains(unique_methods, 'kmeans', 'IgnoreCase', true), 1);
    
    % Look for standard kmeans
    kmeans_idx = find(contains(unique_methods, 'kmeans', 'IgnoreCase', true) & ~contains(unique_methods, 'spm', 'IgnoreCase', true), 1);
    
    if isempty(spm_kmeans_idx) || isempty(kmeans_idx)
        fprintf('⚠ Need both SPM-K-Means and K-Means data\n');
        fprintf('   Available methods: %s\n', strjoin(unique_methods, ', '));
        return;
    end
    
    spm_kmeans_method = unique_methods{spm_kmeans_idx};
    kmeans_method = unique_methods{kmeans_idx};
    
    fprintf('  Using methods: %s vs %s\n', spm_kmeans_method, kmeans_method);
    
    % ========== FIGURE 1: Absolute K Error by Criterion and Montage ==========
    for method_idx = 1:2
        if method_idx == 1
            method = spm_kmeans_method;
            method_label = 'SPM-K-Means';
        else
            method = kmeans_method;
            method_label = 'K-Means Standard';
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
            method = spm_kmeans_method;
            method_label = 'SPM-K-Means';
        else
            method = kmeans_method;
            method_label = 'K-Means Standard';
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
    method_labels = {'SPM VB', 'K-Means'};
    
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
        
        idx_spm = strcmp(cellstr(string(T_filtered.method)), 'spm vb') & ...
                  strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
        idx_kmeans = strcmp(cellstr(string(T_filtered.method)), 'kmeans standard') & ...
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
        
        idx_spm = strcmp(cellstr(string(T_filtered.method)), 'spm vb') & ...
                  strcmp(cellstr(string(T_filtered.criterion_clean)), crit);
        idx_kmeans = strcmp(cellstr(string(T_filtered.method)), 'kmeans standard') & ...
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
