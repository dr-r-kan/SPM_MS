function [summary, selected_table, candidate_table] = apply_spm_vb_meta_selector(source, model_file, varargin)
%APPLY_SPM_VB_META_SELECTOR Apply a trained meta-selector to per-K simulation metrics.
%
% Example:
%   [summary, selected_table] = apply_spm_vb_meta_selector( ...
%       'outputs/sim_test/results/k_candidate_metrics.csv', ...
%       'outputs/sim_train/results/spm_vb_meta_selector.mat');

    p = inputParser;
    addRequired(p, 'source');
    addRequired(p, 'model_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_prefix', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'verbose', true, @islogical);
    parse(p, source, model_file, varargin{:});
    cfg = p.Results;

    model_file = char(string(model_file));
    if ~isfile(model_file)
        error('Meta-selector model file not found: %s', model_file);
    end
    S = load(model_file, 'model');
    if ~isfield(S, 'model') || ~isstruct(S.model)
        error('Model file does not contain a valid model struct: %s', model_file);
    end
    model = S.model;

    [Tprep, feature_names] = prepare_spm_vb_meta_selector_dataset(source, 'method', 'spm_vb');
    if ~isequal(cellstr(string(model.feature_names(:))), cellstr(string(feature_names(:))))
        error('Model feature list does not match the prepared dataset feature list.');
    end

    X = zeros(height(Tprep), numel(feature_names));
    for i = 1:numel(feature_names)
        X(:, i) = double(Tprep.(feature_names{i}));
    end
    candidate_table = Tprep;
    candidate_table.meta_probability = predict_logistic_ridge(X, model.beta);

    groups = unique(string(candidate_table.fit_group_id), 'stable');
    selected_rows = cell(numel(groups), 1);
    correct = false(numel(groups), 1);
    for i = 1:numel(groups)
        mask = string(candidate_table.fit_group_id) == groups(i);
        Ti = candidate_table(mask, :);
        [best_prob, idx] = max(Ti.meta_probability);
        chosen = Ti(idx, :);
        correct(i) = logical(chosen.is_true_k);
        selected_rows{i} = table( ...
            groups(i), chosen.K_true(1), chosen.K_candidate(1), best_prob, double(chosen.is_true_k(1)), ...
            'VariableNames', {'fit_group_id', 'K_true', 'K_selected', 'meta_probability', 'K_correct'});
    end
    selected_table = vertcat(selected_rows{:});

    summary = struct( ...
        'n_runs', numel(groups), ...
        'run_accuracy', mean(correct), ...
        'mean_selected_probability', mean(selected_table.meta_probability, 'omitnan'), ...
        'source', char(string(source)), ...
        'model_file', model_file);

    output_prefix = char(string(cfg.output_prefix));
    if isempty(output_prefix)
        if istable(source)
            output_prefix = '';
        else
            [folder, base, ~] = fileparts(char(string(source)));
            output_prefix = fullfile(folder, [base '_meta_selector']);
        end
    end

    if ~isempty(output_prefix)
        out_dir = fileparts(output_prefix);
        if ~isempty(out_dir) && ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        writetable(candidate_table, [output_prefix '_candidate_predictions.csv']);
        writetable(selected_table, [output_prefix '_selected_k.csv']);
        summary_table = struct2table(summary, 'AsArray', true);
        writetable(summary_table, [output_prefix '_summary.csv']);
    end

    if cfg.verbose
        fprintf('Applied SPM-VB meta-selector to %d runs.\n', summary.n_runs);
        fprintf('Run-level accuracy: %.3f\n', summary.run_accuracy);
        if ~isempty(output_prefix)
            fprintf('Saved outputs with prefix: %s\n', output_prefix);
        end
    end
end

function prob = predict_logistic_ridge(X, beta)
    eta = [ones(size(X, 1), 1), double(X)] * double(beta(:));
    eta = max(min(eta, 35), -35);
    prob = 1 ./ (1 + exp(-eta));
end
