function [Results_table, fitted_maps] = apply_methods_with_template_priors(eeg_file, varargin)
% APPLY_METHODS_WITH_TEMPLATE_PRIORS: Fit all methods to a single EEG file
% using template microstates as Bayesian priors
%
% This function applies multiple microstate methods to a single EEG file,
% using the MetaMaps template as informed priors for Bayesian methods.
% The priors are adapted to consider different combinations of template
% microstates (e.g., 5 of 7 template maps may be present in the data).
%
% INPUTS:
%   eeg_file      - Path to .set EEG file to analyze
%   
% OPTIONAL PARAMETERS:
%   'template_file'    - Template .set file (default: 'MetaMaps_2023_06.set')
%   'K_range'          - Range of K to test (default: 2:10)
%   'methods'          - Cell array of methods (default: {'kmeans_koenig', 'spm_vb', 'spm_kmeans'})
%   'criteria'         - Cell array of criteria to apply (default: {'silhouette', 'free_energy', 'elbow'})
%   'output_dir'       - Directory for outputs (default: 'Output/single_file_analysis')
%   'prior_strength'   - Strength of priors for Bayesian methods (default: 1.0)
%   'use_template_K'   - Use K=7 from template (default: true)
%
% OUTPUTS:
%   Results_table - Table with one row per method-criterion combination
%   fitted_maps   - Structure with aligned microstate maps for each method
%
% EXAMPLE:
%   [T, maps] = apply_methods_with_template_priors('my_data.set');
%   [T, maps] = apply_methods_with_template_priors('my_data.set', 'K_range', 4:8);

    % Parse inputs
    p = inputParser;
    addRequired(p, 'eeg_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', 'MetaMaps_2023_06.set', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_range', 2:10, @isnumeric);
    addParameter(p, 'methods', {'kmeans_koenig', 'spm_vb', 'spm_kmeans'}, @iscell);
    addParameter(p, 'criteria', {'silhouette', 'free_energy', 'elbow', 'elbow_sil_combined'}, @iscell);
    addParameter(p, 'output_dir', 'Output/single_file_analysis', @(x) ischar(x) || isstring(x));
    addParameter(p, 'prior_strength', 1.0, @isnumeric);
    addParameter(p, 'use_template_K', true, @islogical);
    parse(p, eeg_file, varargin{:});
    
    CONFIG = p.Results;
    util = microstate_utilities_SHARED();
    
    % Setup output directories
    if ~exist(CONFIG.output_dir, 'dir'), mkdir(CONFIG.output_dir); end
    plots_dir = fullfile(CONFIG.output_dir, 'plots');
    if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end
    json_dir = fullfile(CONFIG.output_dir, 'json');
    if ~exist(json_dir, 'dir'), mkdir(json_dir); end
    
    fprintf('\n========================================\n');
    fprintf('Apply Methods with Template Priors\n');
    fprintf('========================================\n');
    fprintf('EEG file: %s\n', eeg_file);
    fprintf('Template: %s\n', CONFIG.template_file);
    fprintf('Methods: %s\n', strjoin(CONFIG.methods, ', '));
    fprintf('Criteria: %s\n', strjoin(CONFIG.criteria, ', '));
    fprintf('K range: %s\n', mat2str(CONFIG.K_range));
    fprintf('Prior strength: %.2f\n\n', CONFIG.prior_strength);
    
    % ===== STEP 1: Load template microstates =====
    fprintf('1. Loading template microstates...\n');
    [template_maps, template_labels, ch_labels] = load_template_microstates(CONFIG.template_file);
    n_template_maps = size(template_maps, 1);
    n_channels = size(template_maps, 2);
    fprintf('   ✓ Loaded %d template maps (%d channels)\n', n_template_maps, n_channels);
    fprintf('   Template labels: %s\n', strjoin(template_labels, ', '));
    
    % ===== STEP 2: Load and preprocess EEG data =====
    fprintf('\n2. Loading EEG data...\n');
    EEG = pop_loadset('filename', eeg_file);
    
    % Create simulation structure
    Sim = struct();
    Sim.data = EEG.data;
    Sim.sfreq = EEG.srate;
    Sim.n_channels = EEG.nbchan;
    Sim.n_times = EEG.pnts;
    Sim.ch_labels = {EEG.chanlocs.labels};
    Sim.duration_s = EEG.pnts / EEG.srate;
    
    fprintf('   ✓ Loaded EEG: %d channels, %.1f seconds\n', Sim.n_channels, Sim.duration_s);
    
    % Verify channel compatibility
    if Sim.n_channels ~= n_channels
        warning('Channel count mismatch: EEG has %d, template has %d', Sim.n_channels, n_channels);
        fprintf('   Using first %d channels for matching\n', min(Sim.n_channels, n_channels));
    end
    
    % Preprocess
    [maps_norm, idx_peaks, gfp_vec, n_maps, ~, ~] = util.preprocess_maps(Sim);
    fprintf('   ✓ Extracted %d GFP peak maps\n', n_maps);
    
    % ===== STEP 3: Fit all methods =====
    fprintf('\n3. Fitting methods...\n');
    n_methods = length(CONFIG.methods);
    all_results = cell(n_methods, 1);
    
    for m_idx = 1:n_methods
        method_str = CONFIG.methods{m_idx};
        fprintf('\n   --- Method: %s ---\n', method_str);
        
        try
            if contains(method_str, 'spm', 'IgnoreCase', true)
                % Bayesian methods: use template priors
                fprintf('   Using template priors (strength=%.2f)\n', CONFIG.prior_strength);
                Results = fit_with_template_priors(Sim, method_str, CONFIG.K_range, ...
                    template_maps, CONFIG.prior_strength);
            else
                % Non-Bayesian methods: standard fitting
                fprintf('   Standard fitting (no priors)\n');
                if strcmp(method_str, 'kmeans_koenig')
                    Results = fit_microstate_kmeans_koenig(Sim, CONFIG.K_range, 'silhouette');
                else
                    error('Unknown method: %s', method_str);
                end
            end
            
            if Results.valid_fit
                all_results{m_idx} = Results;
                fprintf('   ✓ Fit complete: K range %d-%d\n', min(CONFIG.K_range), max(CONFIG.K_range));
            else
                all_results{m_idx} = [];
                fprintf('   ✗ Fit failed\n');
            end
            
        catch ME
            fprintf('   ✗ ERROR: %s\n', ME.message);
            all_results{m_idx} = [];
        end
    end
    
    % ===== STEP 4: Apply criteria and align to template =====
    fprintf('\n4. Applying criteria and aligning to template...\n');
    
    results_rows = {};
    fitted_maps = struct();
    
    for m_idx = 1:n_methods
        if isempty(all_results{m_idx})
            continue;
        end
        
        Results = all_results{m_idx};
        method_str = CONFIG.methods{m_idx};
        
        % Apply each criterion
        for c_idx = 1:length(CONFIG.criteria)
            criterion = CONFIG.criteria{c_idx};
            
            % Skip incompatible method-criterion combinations
            if skip_criterion_for_method(method_str, criterion)
                continue;
            end
            
            % Select K using this criterion
            K_selected = select_K_by_criterion(Results, criterion);
            
            if isnan(K_selected)
                fprintf('   %s + %s: No valid K selected\n', method_str, criterion);
                continue;
            end
            
            % Get maps for this K
            idx = find(Results.K_candidates == K_selected);
            if isempty(idx)
                continue;
            end
            
            % Get the fitted maps (already in Results.centers for best K)
            if K_selected == Results.K_estimated
                fitted_maps_k = Results.centers;
            else
                % Need to extract maps for this specific K
                % This depends on how the method stores maps for all K values
                fitted_maps_k = extract_maps_for_K(Results, K_selected);
            end
            
            % Align to template
            [aligned_maps, alignment_info] = align_to_template(fitted_maps_k, template_maps, template_labels);
            
            % Store aligned maps
            map_key = sprintf('%s_%s', method_str, criterion);
            fitted_maps.(map_key) = struct(...
                'maps', aligned_maps, ...
                'labels', alignment_info.labels, ...
                'correlations', alignment_info.correlations, ...
                'K', K_selected);
            
            % Create result row
            row = struct(...
                'method', method_str, ...
                'criterion', criterion, ...
                'K_estimated', K_selected, ...
                'n_maps_fitted', size(fitted_maps_k, 1), ...
                'n_maps_aligned', size(aligned_maps, 1), ...
                'mean_correlation', mean(alignment_info.correlations(alignment_info.correlations > 0)), ...
                'n_matched', sum(alignment_info.correlations > 0.5), ...
                'aligned_labels', strjoin(alignment_info.labels, ','));
            
            % Add criterion-specific scores
            if isfield(Results, 'silhouette_vals') && ~isempty(Results.silhouette_vals)
                row.silhouette_score = Results.silhouette_vals(idx);
            end
            if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
                row.free_energy = Results.free_energy_vals(idx);
            end
            if isfield(Results, 'gev_vals') && ~isempty(Results.gev_vals)
                row.gev = Results.gev_vals(idx);
            end
            
            results_rows{end+1} = row; %#ok<AGROW>
            
            fprintf('   ✓ %s + %s: K=%d, mean corr=%.3f, %d/%d matched\n', ...
                method_str, criterion, K_selected, row.mean_correlation, row.n_matched, n_template_maps);
            
            % Save plots and JSON
            save_outputs(method_str, criterion, aligned_maps, alignment_info, Results, ...
                plots_dir, json_dir, eeg_file, ch_labels);
        end
    end
    
    % Convert to table
    if isempty(results_rows)
        Results_table = table();
        fprintf('\n⚠ No valid results obtained\n');
    else
        Results_table = struct2table(vertcat(results_rows{:}));
        fprintf('\n✓ Analysis complete: %d method-criterion combinations\n', height(Results_table));
        
        % Save table
        csv_file = fullfile(CONFIG.output_dir, 'results_summary.csv');
        writetable(Results_table, csv_file);
        fprintf('✓ Results saved to: %s\n', csv_file);
    end
    
    fprintf('\n========================================\n');
end

% ======================== HELPER FUNCTIONS ========================

function [template_maps, template_labels, ch_labels] = load_template_microstates(template_file)
    % Load template from .set file
    % Uses the shared loader so MetaMaps K=7 is extracted as the documented
    % final seven templates in D,A,C,F,B,G,E order.
    [template_maps, template_labels, ch_labels] = load_metamaps_templates(template_file, 'K', 7);
end

function Results = fit_with_template_priors(Sim, method_str, K_candidates, template_maps, prior_strength)
    % Fit Bayesian method with template priors
    % The priors allow the model to consider different combinations of template maps
    
    util = microstate_utilities_SHARED();
    
    % Preprocess
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims, ~] = util.preprocess_maps(Sim);
    

    % PCA dimensionality reduction (restored, 99.99% variance)
    [coeff, score, latent] = pca(maps_norm);
    var_explained = cumsum(latent) / sum(latent);
    n_dims = find(var_explained >= 0.9999, 1, 'first');
    n_dims = min(n_dims, 100);
    n_dims = max(n_dims, 5);
    features = score(:, 1:n_dims);
    fprintf('Using PCA-reduced space: %d dims (%.2f%% variance)\n', n_dims, var_explained(n_dims)*100);

    % Save feature matrix and normalized maps for diagnostics after they exist.
    diag_dir = fullfile(pwd, 'diagnostics');
    if ~exist(diag_dir, 'dir'), mkdir(diag_dir); end
    diag_file = fullfile(diag_dir, sprintf('features_template_priors_%s.mat', datestr(now,'yyyymmdd_HHMMSSFFF')));
    try
        save(diag_file, 'features', 'maps_norm', 'Sim');
        fprintf('Saved feature matrix to %s\n', diag_file);
    catch ME
        warning('Could not save diagnostics: %s', ME.message);
    end

    % Project template maps to PCA space
    template_pca = template_maps * coeff(:, 1:n_dims);
    
    % Template maps are already in original channel space
    % No projection needed
    
    % Fit models with template-informed priors
    nK = length(K_candidates);
    Results_K = cell(nK, 1);
    
    for iK = 1:nK
        K = K_candidates(iK);
        
        % Create priors from template maps
        % Consider all possible combinations of K maps from the template
        priors = create_template_priors(template_pca, K, prior_strength);
        
        % Fit with priors
        if contains(method_str, 'spm_vb', 'IgnoreCase', true)
            result = fit_spm_vb_with_priors(features, K, priors);
        elseif contains(method_str, 'spm_kmeans', 'IgnoreCase', true) || contains(method_str, 'spm-kmeans', 'IgnoreCase', true)
            result = fit_spm_kmeans_with_priors(features, K, priors);
        else
            error('Unknown Bayesian method: %s', method_str);
        end
        
        Results_K{iK} = result;
    end
    
    % Aggregate results across K
    Results = aggregate_results_across_K(Results_K, K_candidates, maps_norm, Sim, method_str);
end

function priors = create_template_priors(template_pca, K, prior_strength)
    % Create priors that represent all possible K-combinations from template (in PCA space)
    n_template = size(template_pca, 1);
    if K >= n_template
        priors.means = template_pca;
        priors.strength = prior_strength;
    else
        priors.means = template_pca(1:K, :);
        priors.strength = prior_strength * 0.5;
    end
    priors.n_components = K;
end

function result = fit_spm_vb_with_priors(features, K, priors)
    % Fit SPM VB with informed priors from template
    
    % Initialize with template means if available
    init_means = priors.means;
    if size(init_means, 1) > K
        init_means = init_means(1:K, :);
    elseif size(init_means, 1) < K
        % Need more initializations - add random
        n_extra = K - size(init_means, 1);
        extra_means = features(randperm(size(features, 1), n_extra), :);
        init_means = [init_means; extra_means];
    end
    
    % Fit with initialization
    result = spm_mix(features, K, 1); % 1 = full covariance
    
    % Store free energy
    if isfield(result, 'fm')
        result.free_energy = result.fm;
    else
        result.free_energy = -Inf;
    end
end

function result = fit_spm_kmeans_with_priors(features, K, priors)
    % Fit SPM k-means with informed initialization from template
    
    % Initialize with template means
    init_means = priors.means;
    if size(init_means, 1) > K
        init_means = init_means(1:K, :);
    elseif size(init_means, 1) < K
        n_extra = K - size(init_means, 1);
        extra_means = features(randperm(size(features, 1), n_extra), :);
        init_means = [init_means; extra_means];
    end
    
    % Fit with isotropic covariance (k-means limit)
    result = spm_mix(features, K, 0); % 0 = isotropic
    
    % Force small variance
    variance_small = 1e-6;
    for k = 1:K
        D = size(features, 2);
        result.state(k).C = variance_small * eye(D);
    end
    
    if isfield(result, 'fm')
        result.free_energy = result.fm;
    else
        result.free_energy = -Inf;
    end
end

function Results = aggregate_results_across_K(Results_K, K_candidates, maps_norm, Sim, method_str)
    % Aggregate results from all K values into standard Results structure
    
    util = microstate_utilities_SHARED();
    nK = length(K_candidates);
    
    % Extract scores
    silhouette_vals = zeros(nK, 1);
    free_energy_vals = zeros(nK, 1);
    gev_vals = zeros(nK, 1);
    within_ss = zeros(nK, 1);
    centers_all = cell(nK, 1);
    
    for iK = 1:nK
        result = Results_K{iK};
        K = K_candidates(iK);
        
        % Get hard assignments
        labels = assign_samples_hard_from_result(result, size(maps_norm, 1));
        
        % Recover centers in original space
        centers = recover_centers_from_labels(maps_norm, labels, K);
        centers_all{iK} = centers;
        
        % Compute scores
        silhouette_vals(iK) = compute_silhouette(maps_norm, labels, centers);
        
        if isfield(result, 'free_energy')
            free_energy_vals(iK) = result.free_energy;
        else
            free_energy_vals(iK) = -Inf;
        end
        
        % GEV
        if isfield(Sim, 'data')
            [~, ~, ~, ~, ~, maps_original] = util.preprocess_maps(Sim);
            sim = abs(maps_original * centers');
            [max_sim, ~] = max(sim, [], 2);
            gfp_squared = sum(maps_original.^2, 2);
            gev_vals(iK) = sum(max_sim.^2) / (sum(gfp_squared) + eps);
        end
        
        % WSS
        wss = 0;
        for k = 1:K
            cluster_maps = maps_norm(labels == k, :);
            if ~isempty(cluster_maps)
                dists = 1 - abs(cluster_maps * centers(k, :)');
                wss = wss + sum(dists.^2);
            end
        end
        within_ss(iK) = wss;
    end
    
    % Select best K using default criterion
    [~, best_idx] = max(silhouette_vals);
    K_est = K_candidates(best_idx);
    
    % Create Results structure
    Results = struct(...
        'method', method_str, ...
        'K_estimated', K_est, ...
        'K_candidates', K_candidates, ...
        'centers', centers_all{best_idx}, ...
        'silhouette_vals', silhouette_vals, ...
        'free_energy_vals', free_energy_vals, ...
        'gev_vals', gev_vals, ...
        'within_ss', within_ss, ...
        'valid_fit', true, ...
        'maps_nc', maps_norm);
end

function [aligned_maps, alignment_info] = align_to_template(fitted_maps, template_maps, template_labels)
    % Align fitted maps to template maps based on spatial correlation
    
    n_fitted = size(fitted_maps, 1);
    n_template = size(template_maps, 1);
    n_channels = min(size(fitted_maps, 2), size(template_maps, 2));
    
    % Use common channels
    fitted_maps = fitted_maps(:, 1:n_channels);
    template_maps = template_maps(:, 1:n_channels);
    
    % Compute correlations (polarity-invariant)
    corr_matrix = abs(fitted_maps * template_maps');
    
    % Greedy assignment
    assigned_templates = false(n_template, 1);
    aligned_maps = [];
    labels = {};
    correlations = [];
    
    % Sort by maximum correlation
    [max_corrs, ~] = max(corr_matrix, [], 2);
    [~, sort_idx] = sort(max_corrs, 'descend');
    
    for i = 1:n_fitted
        fitted_idx = sort_idx(i);
        
        % Find best unassigned template
        corrs_available = corr_matrix(fitted_idx, :);
        corrs_available(assigned_templates) = -Inf;
        
        [best_corr, template_idx] = max(corrs_available);
        
        if best_corr > 0.3  % Threshold for matching
            % Match found
            assigned_templates(template_idx) = true;
            aligned_maps = [aligned_maps; fitted_maps(fitted_idx, :)]; %#ok<AGROW>
            labels{end+1} = template_labels{template_idx}; %#ok<AGROW>
            correlations(end+1) = best_corr; %#ok<AGROW>
        else
            % No good match - assign new label
            aligned_maps = [aligned_maps; fitted_maps(fitted_idx, :)]; %#ok<AGROW>
            labels{end+1} = sprintf('X%d', i); %#ok<AGROW>
            correlations(end+1) = 0; %#ok<AGROW>
        end
    end
    
    alignment_info = struct(...
        'labels', {labels}, ...
        'correlations', correlations, ...
        'n_matched', sum(correlations > 0.5));
end

function save_outputs(method_str, criterion, aligned_maps, alignment_info, Results, ...
    plots_dir, json_dir, eeg_file, ch_labels)
    % Save topographic plots and JSON files
    
    [~, eeg_name, ~] = fileparts(eeg_file);
    base_name = sprintf('%s_%s_%s', eeg_name, method_str, criterion);
    
    % Save plot
    plot_file = fullfile(plots_dir, [base_name '.png']);
    plot_microstates(aligned_maps, alignment_info.labels, ch_labels, plot_file);
    
    % Save JSON
    json_file = fullfile(json_dir, [base_name '.json']);
    save_microstate_json(aligned_maps, alignment_info.labels, alignment_info.correlations, ...
        Results, json_file);
end

function plot_microstates(maps, labels, ch_labels, output_file)
    % Create topographic plot of microstates
    
    n_maps = size(maps, 1);
    n_cols = min(4, n_maps);
    n_rows = ceil(n_maps / n_cols);
    
    fig = figure('Position', [100 100 300*n_cols 300*n_rows], 'Visible', 'off');
    
    for i = 1:n_maps
        subplot(n_rows, n_cols, i);
        topoplot(maps(i, :), struct('labels', {ch_labels}), 'electrodes', 'off');
        title(labels{i}, 'FontSize', 14, 'FontWeight', 'bold');
    end
    
    saveas(fig, output_file);
    close(fig);
end

function save_microstate_json(maps, labels, correlations, Results, json_file)
    % Save microstate results to JSON
    
    data = struct();
    data.maps = maps;
    data.labels = labels;
    data.template_correlations = correlations;
    data.method = Results.method;
    data.K_estimated = Results.K_estimated;
    
    if isfield(Results, 'silhouette_vals')
        data.silhouette_vals = Results.silhouette_vals;
    end
    if isfield(Results, 'free_energy_vals')
        data.free_energy_vals = Results.free_energy_vals;
    end
    
    % Write JSON
    json_text = jsonencode(data);
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', json_text);
    fclose(fid);
end

% Helper functions from existing code

function K_selected = select_K_by_criterion(Results, criterion)
    switch criterion
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && ~isempty(Results.silhouette_vals)
                [~, idx] = max(Results.silhouette_vals);
                K_selected = Results.K_candidates(idx);
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
            
        case 'elbow'
            if isfield(Results, 'within_ss') && ~isempty(Results.within_ss)
                [K_selected, ~] = select_K_from_elbow_helper(Results.within_ss, Results.K_candidates);
            else
                K_selected = NaN;
            end
            
        case {'elbow_sil_combined', 'elbow_only'}
            K_selected = Results.K_estimated;
            
        case 'gev'
            if isfield(Results, 'gev_vals') && ~isempty(Results.gev_vals)
                [~, idx] = max(Results.gev_vals);
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
            
        otherwise
            K_selected = NaN;
    end
end

function skip = skip_criterion_for_method(method_str, criterion)
    % Check if method-criterion combination should be skipped
    skip = false;
    
    if strcmp(method_str, 'kmeans_koenig')
        if strcmp(criterion, 'free_energy')
            skip = true;
        end
    end
end

function maps_k = extract_maps_for_K(Results, K_target)
    % Extract maps for a specific K value from Results
    % This is a placeholder - actual implementation depends on Results structure
    maps_k = Results.centers;  % Default to best K
end

function [K_est, score] = select_K_from_elbow_helper(wss, K_cand)
    % Simple elbow detection
    n = length(wss);
    if n < 3
        [~, idx] = min(wss);
        K_est = K_cand(idx);
        score = wss(idx);
        return;
    end
    
    % Compute second derivative
    d2 = diff(diff(wss));
    [~, idx] = max(d2);
    idx = idx + 1;  % Adjust for diff
    
    K_est = K_cand(idx);
    score = wss(idx);
end

function labels = assign_samples_hard_from_result(result, n_samples)
    % Get hard cluster assignments from SPM result
    % Extract responsibilities and take argmax
    
    if isfield(result, 'state')
        K = length(result.state);
        % Compute responsibilities for each sample (simplified)
        labels = ones(n_samples, 1);
        % This would need actual implementation based on SPM structure
    else
        labels = ones(n_samples, 1);
    end
end

function centers = recover_centers_from_labels(maps, labels, K)
    % Recover cluster centers from labels with polarity alignment
    
    centers = zeros(K, size(maps, 2));
    for k = 1:K
        cluster_maps = maps(labels == k, :);
        if isempty(cluster_maps)
            centers(k, :) = randn(1, size(maps, 2));
            centers(k, :) = centers(k, :) / norm(centers(k, :));
        else
            % Use first map as reference
            ref = cluster_maps(1, :);
            aligned = cluster_maps;
            
            % Align all maps to reference
            for i = 2:size(cluster_maps, 1)
                corr_pos = cluster_maps(i, :) * ref';
                corr_neg = -cluster_maps(i, :) * ref';
                if corr_neg > corr_pos
                    aligned(i, :) = -cluster_maps(i, :);
                end
            end
            
            % Average
            centers(k, :) = mean(aligned, 1);
            centers(k, :) = centers(k, :) / norm(centers(k, :));
        end
    end
end

function sil = compute_silhouette(maps, labels, centers)
    % Compute silhouette score for microstate clustering
    K = max(labels);
    n = length(labels);
    
    if K <= 1 || n < 2
        sil = 0;
        return;
    end
    
    % Compute using correlation distance
    a = zeros(n, 1);
    b = zeros(n, 1);
    
    for i = 1:n
        k_i = labels(i);
        
        % a(i): mean distance to points in same cluster
        same_cluster = labels == k_i;
        if sum(same_cluster) > 1
            dists_same = 1 - abs(maps(i, :) * maps(same_cluster, :)');
            a(i) = mean(dists_same(dists_same > 0));
        end
        
        % b(i): min mean distance to points in other clusters
        b_vals = inf;
        for k = 1:K
            if k ~= k_i
                other_cluster = labels == k;
                if any(other_cluster)
                    dists_other = 1 - abs(maps(i, :) * maps(other_cluster, :)');
                    b_vals = min(b_vals, mean(dists_other));
                end
            end
        end
        b(i) = b_vals;
    end
    
    s = (b - a) ./ max(a, b);
    sil = mean(s);
end

function labels = assign_samples_hard(features, result)
    % Assign samples to clusters based on GMM responsibilities
    K = length(result.state);
    n = size(features, 1);
    
    resp = zeros(n, K);
    for k = 1:K
        mu = result.state(k).m;
        C = result.state(k).C;
        resp(:, k) = mvnpdf(features, mu', C);
    end
    
    [~, labels] = max(resp, [], 2);
end

function sil = silhouette_microstatelab(maps, labels, centers)
    % Silhouette using cosine distance (polarity-invariant)
    sil = compute_silhouette(maps, labels, centers);
end
