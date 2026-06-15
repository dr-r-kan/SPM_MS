function [Tprep, feature_names] = prepare_spm_vb_meta_selector_dataset(source, varargin)
%PREPARE_SPM_VB_META_SELECTOR_DATASET Build a per-K design matrix for meta-selection.
%
% Input can be a table or a path to k_candidate_metrics.csv produced by
% simulated_ms_retrieval_experiment.m. The output preserves one row per
% candidate K, with consistent feature columns for training or inference.

    p = inputParser;
    addRequired(p, 'source');
    addParameter(p, 'method', 'spm_vb', @(x) ischar(x) || isstring(x));
    parse(p, source, varargin{:});

    if istable(source)
        T = source;
    else
        source_csv = char(string(source));
        if ~isfile(source_csv)
            error('Meta-selector dataset source not found: %s', source_csv);
        end
        T = readtable(source_csv, 'TextType', 'string');
    end

    required_vars = {'fit_group_id', 'K_candidate', 'is_true_k', 'method'};
    missing_vars = required_vars(~ismember(required_vars, T.Properties.VariableNames));
    if ~isempty(missing_vars)
        error('Dataset is missing required columns: %s', strjoin(missing_vars, ', '));
    end

    method_name = lower(strtrim(char(string(p.Results.method))));
    method_mask = strtrim(lower(string(T.method))) == method_name;
    Tprep = T(method_mask, :);
    if height(Tprep) == 0
        error('No rows found for method %s in the candidate-metrics dataset.', method_name);
    end

    Tprep.fit_group_id = string(Tprep.fit_group_id);
    Tprep.K_candidate = double(Tprep.K_candidate);
    Tprep.is_true_k = double(Tprep.is_true_k);

    Tprep = ensure_feature_column(Tprep, 'free_energy_norm', {'free_energy_norm', 'free_energy'});
    Tprep = ensure_feature_column(Tprep, 'silhouette_norm', {'silhouette_norm', 'silhouette'});
    Tprep = ensure_feature_column(Tprep, 'gev_norm', {'gev_norm', 'gev'});
    Tprep = ensure_feature_column(Tprep, 'covariance_tightness_norm', {'covariance_tightness_norm', 'covariance_primary'});
    Tprep = ensure_feature_column(Tprep, 'free_energy_elbow_norm', {'free_energy_elbow_norm', 'score_free_energy_elbow'});
    Tprep = ensure_feature_column(Tprep, 'covariance_elbow_norm', {'covariance_elbow_norm', 'score_covariance_elbow'});

    if ~ismember('edge_distance_norm', Tprep.Properties.VariableNames)
        Tprep.edge_distance_norm = compute_groupwise_edge_distance_norm(Tprep.fit_group_id, Tprep);
    else
        Tprep.edge_distance_norm = double(Tprep.edge_distance_norm);
    end
    if ~ismember('is_edge_candidate', Tprep.Properties.VariableNames)
        Tprep.is_edge_candidate = double(Tprep.edge_distance_norm <= eps);
    else
        Tprep.is_edge_candidate = double(Tprep.is_edge_candidate);
    end

    feature_names = { ...
        'free_energy_norm', ...
        'silhouette_norm', ...
        'gev_norm', ...
        'covariance_tightness_norm', ...
        'free_energy_elbow_norm', ...
        'covariance_elbow_norm', ...
        'edge_distance_norm', ...
        'is_edge_candidate'};

    for i = 1:numel(feature_names)
        f = feature_names{i};
        values = double(Tprep.(f));
        values(~isfinite(values)) = 0;
        Tprep.(f) = values;
    end
end

function T = ensure_feature_column(T, target_name, source_names)
    if ismember(target_name, T.Properties.VariableNames)
        T.(target_name) = double(T.(target_name));
        return;
    end

    source_name = '';
    for i = 1:numel(source_names)
        if ismember(source_names{i}, T.Properties.VariableNames)
            source_name = source_names{i};
            break;
        end
    end
    if isempty(source_name)
        T.(target_name) = zeros(height(T), 1);
        return;
    end

    raw_values = double(T.(source_name));
    if strcmp(target_name, 'covariance_tightness_norm') && strcmp(source_name, 'covariance_primary')
        raw_values = invert_groupwise_metric(T.fit_group_id, raw_values);
    end
    T.(target_name) = groupwise_normalize_01(T.fit_group_id, raw_values);
end

function values_norm = compute_groupwise_edge_distance_norm(group_ids, T)
    values_norm = zeros(height(T), 1);
    groups = unique(group_ids, 'stable');
    for g = 1:numel(groups)
        mask = group_ids == groups(g);
        if ismember('edge_distance', T.Properties.VariableNames)
            vals = double(T.edge_distance(mask));
        else
            k_idx = (1:nnz(mask))';
            vals = min(k_idx - 1, nnz(mask) - k_idx);
        end
        max_val = max(vals);
        if isfinite(max_val) && max_val > eps
            values_norm(mask) = vals ./ max_val;
        else
            values_norm(mask) = 0;
        end
    end
end

function values_norm = groupwise_normalize_01(group_ids, values)
    values = double(values(:));
    values_norm = zeros(size(values));
    groups = unique(group_ids, 'stable');
    for g = 1:numel(groups)
        mask = group_ids == groups(g);
        vals = values(mask);
        finite_mask = isfinite(vals);
        if ~any(finite_mask)
            values_norm(mask) = 0;
            continue;
        end
        vals_finite = vals(finite_mask);
        vmin = min(vals_finite);
        vmax = max(vals_finite);
        tmp = zeros(size(vals));
        if isfinite(vmin) && isfinite(vmax) && abs(vmax - vmin) > eps
            tmp(finite_mask) = (vals_finite - vmin) ./ (vmax - vmin);
        end
        values_norm(mask) = tmp;
    end
end

function values_out = invert_groupwise_metric(group_ids, values_in)
    values_out = nan(size(values_in));
    groups = unique(group_ids, 'stable');
    for g = 1:numel(groups)
        mask = group_ids == groups(g);
        vals = values_in(mask);
        finite_mask = isfinite(vals);
        if ~any(finite_mask)
            continue;
        end
        vals_out = nan(size(vals));
        vals_out(finite_mask) = max(vals(finite_mask)) - vals(finite_mask);
        values_out(mask) = vals_out;
    end
end
