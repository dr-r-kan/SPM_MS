function model = train_spm_vb_meta_selector(train_source, varargin)
%TRAIN_SPM_VB_META_SELECTOR Fit a grouped logistic meta-selector for SPM-VB K choice.
%
% Example:
%   model = train_spm_vb_meta_selector( ...
%       'outputs/sim_train/results/k_candidate_metrics.csv', ...
%       'output_model_file', 'outputs/sim_train/results/spm_vb_meta_selector.mat');

    p = inputParser;
    addRequired(p, 'train_source');
    addParameter(p, 'output_model_file', 'spm_vb_meta_selector.mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'holdout_fraction', 0.2, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 1);
    addParameter(p, 'ridge_lambda', 1.0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'max_iter', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'tol', 1e-6, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'seed', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'verbose', true, @islogical);
    parse(p, train_source, varargin{:});
    cfg = p.Results;

    [Tprep, feature_names] = prepare_spm_vb_meta_selector_dataset(train_source, 'method', 'spm_vb');
    group_ids = string(Tprep.fit_group_id);
    y = double(Tprep.is_true_k(:));
    X = zeros(height(Tprep), numel(feature_names));
    for i = 1:numel(feature_names)
        X(:, i) = double(Tprep.(feature_names{i}));
    end

    groups = unique(group_ids, 'stable');
    rng(double(cfg.seed), 'twister');
    shuffled = groups(randperm(numel(groups)));
    n_holdout = round(cfg.holdout_fraction * numel(groups));
    if n_holdout >= numel(groups)
        n_holdout = max(0, numel(groups) - 1);
    end
    holdout_groups = shuffled(1:n_holdout);
    train_groups = shuffled((n_holdout + 1):end);
    if isempty(train_groups)
        train_groups = groups;
        holdout_groups = strings(0, 1);
    end

    train_mask = ismember(group_ids, train_groups);
    holdout_mask = ismember(group_ids, holdout_groups);

    beta = fit_logistic_ridge_irls(X(train_mask, :), y(train_mask), cfg.ridge_lambda, cfg.max_iter, cfg.tol);
    train_pred = predict_logistic_ridge(X(train_mask, :), beta);
    train_eval = evaluate_group_predictions(Tprep(train_mask, :), train_pred);

    holdout_eval = struct();
    holdout_predictions = table();
    if any(holdout_mask)
        holdout_pred = predict_logistic_ridge(X(holdout_mask, :), beta);
        [holdout_eval, holdout_predictions] = evaluate_group_predictions(Tprep(holdout_mask, :), holdout_pred);
    end

    model = struct();
    model.model_type = 'spm_vb_meta_selector_logistic_ridge';
    model.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    model.feature_names = feature_names;
    model.beta = beta;
    model.ridge_lambda = cfg.ridge_lambda;
    model.training_source = char(string(train_source));
    model.holdout_fraction = cfg.holdout_fraction;
    model.seed = cfg.seed;
    model.train_evaluation = train_eval;
    model.holdout_evaluation = holdout_eval;

    output_model_file = char(string(cfg.output_model_file));
    output_dir = fileparts(output_model_file);
    if ~isempty(output_dir) && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    save(output_model_file, 'model', '-v7.3');

    coef_table = table(string(['intercept', feature_names]), beta, ...
        'VariableNames', {'feature', 'coefficient'});
    coef_csv = replace_suffix_local(output_model_file, '_coefficients.csv');
    writetable(coef_table, coef_csv);

    if any(holdout_mask)
        holdout_csv = replace_suffix_local(output_model_file, '_holdout_predictions.csv');
        writetable(holdout_predictions, holdout_csv);
    end

    if cfg.verbose
        fprintf('Trained SPM-VB meta-selector on %d runs (%d candidate rows).\n', ...
            numel(train_groups), nnz(train_mask));
        fprintf('Saved model: %s\n', output_model_file);
        fprintf('Saved coefficients: %s\n', coef_csv);
        fprintf('Train run-level accuracy: %.3f\n', train_eval.run_accuracy);
        if any(holdout_mask)
            fprintf('Holdout run-level accuracy: %.3f (%d runs)\n', ...
                holdout_eval.run_accuracy, holdout_eval.n_runs);
        end
    end
end

function beta = fit_logistic_ridge_irls(X, y, lambda, max_iter, tol)
    X = double(X);
    y = double(y(:));
    n = size(X, 1);
    p = size(X, 2);
    X1 = [ones(n, 1), X];
    beta = zeros(p + 1, 1);
    penalty = diag([0; ones(p, 1)]);

    for iter = 1:max_iter
        eta = max(min(X1 * beta, 35), -35);
        mu = 1 ./ (1 + exp(-eta));
        W = max(mu .* (1 - mu), 1e-6);
        z = eta + (y - mu) ./ W;
        Xw = X1 .* W;
        A = X1' * Xw + lambda * penalty;
        b = X1' * (W .* z);
        beta_new = A \ b;
        if norm(beta_new - beta, 2) <= tol * (1 + norm(beta, 2))
            beta = beta_new;
            return;
        end
        beta = beta_new;
    end
end

function prob = predict_logistic_ridge(X, beta)
    X = double(X);
    eta = [ones(size(X, 1), 1), X] * beta;
    eta = max(min(eta, 35), -35);
    prob = 1 ./ (1 + exp(-eta));
end

function [summary, predictions] = evaluate_group_predictions(T, prob)
    T = T;
    T.meta_probability = prob(:);
    groups = unique(string(T.fit_group_id), 'stable');
    pred_rows = cell(numel(groups), 1);
    correct = false(numel(groups), 1);
    for i = 1:numel(groups)
        mask = string(T.fit_group_id) == groups(i);
        Ti = T(mask, :);
        [best_prob, idx] = max(Ti.meta_probability);
        chosen = Ti(idx, :);
        correct(i) = logical(chosen.is_true_k);
        pred_rows{i} = table( ...
            groups(i), chosen.K_true(1), chosen.K_candidate(1), best_prob, double(chosen.is_true_k(1)), ...
            'VariableNames', {'fit_group_id', 'K_true', 'K_selected', 'meta_probability', 'K_correct'});
    end
    predictions = vertcat(pred_rows{:});
    summary = struct( ...
        'n_runs', numel(groups), ...
        'run_accuracy', mean(correct), ...
        'mean_selected_probability', mean(predictions.meta_probability, 'omitnan'));
end

function out = replace_suffix_local(path_in, suffix)
    [folder, base, ~] = fileparts(char(string(path_in)));
    out = fullfile(folder, [base suffix]);
end
