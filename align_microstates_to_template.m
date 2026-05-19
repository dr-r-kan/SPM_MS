function alignment = align_microstates_to_template(estimated_maps, template_file, varargin)
% ALIGN_MICROSTATES_TO_TEMPLATE Match estimated microstates to MetaMaps templates.
%
% Matching is polarity-invariant and channel-label aware when labels are
% provided.  The returned maps keep the estimated-state order, with polarity
% flipped where needed to match the assigned template.

    p = inputParser;
    addRequired(p, 'estimated_maps', @isnumeric);
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'estimated_channel_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'template_K', 7, @isnumeric);
    addParameter(p, 'strong_threshold', 0.5, @isnumeric);
    parse(p, estimated_maps, template_file, varargin{:});

    util = microstate_utilities_SHARED();
    estimated_maps = util.normalize_maps(double(estimated_maps));
    [template_maps, template_labels, template_channel_labels] = ...
        load_metamaps_templates(template_file, 'K', p.Results.template_K);

    est_labels = cellstr(p.Results.estimated_channel_labels);
    tmpl_labels = cellstr(template_channel_labels);
    n_est_channels = size(estimated_maps, 2);
    n_template_channels = size(template_maps, 2);

    if ~isempty(est_labels) && numel(est_labels) >= n_est_channels && ~isempty(tmpl_labels)
        [common_labels, idx_est, idx_template] = intersect( ...
            lower(strtrim(est_labels(1:n_est_channels))), ...
            lower(strtrim(tmpl_labels(1:n_template_channels))), 'stable');
        if numel(common_labels) >= 4
            estimated_common = estimated_maps(:, idx_est);
            template_common = template_maps(:, idx_template);
            channel_match_mode = 'labels';
        else
            [estimated_common, template_common, channel_match_mode, idx_est, idx_template] = ...
                use_first_common_channels(estimated_maps, template_maps);
            common_labels = est_labels(idx_est);
        end
    else
        [estimated_common, template_common, channel_match_mode, idx_est, idx_template] = ...
            use_first_common_channels(estimated_maps, template_maps);
        common_labels = arrayfun(@(i) sprintf('Ch%03d', i), idx_est, 'UniformOutput', false);
    end

    estimated_common = util.normalize_maps(estimated_common);
    template_common = util.normalize_maps(template_common);
    signed_corr = estimated_common * template_common';
    corr_matrix = abs(signed_corr);

    n_est = size(estimated_common, 1);
    n_template = size(template_common, 1);
    template_idx = nan(n_est, 1);
    template_corr = zeros(n_est, 1);
    polarity = ones(n_est, 1);
    used_templates = false(n_template, 1);

    [~, order] = sort(max(corr_matrix, [], 2), 'descend');
    for oi = 1:numel(order)
        e = order(oi);
        available = corr_matrix(e, :);
        available(used_templates) = -Inf;
        [best_corr, t] = max(available);
        if isfinite(best_corr)
            template_idx(e) = t;
            template_corr(e) = best_corr;
            used_templates(t) = true;
            if signed_corr(e, t) < 0
                polarity(e) = -1;
            end
        end
    end

    labels = cell(n_est, 1);
    for e = 1:n_est
        if ~isnan(template_idx(e))
            labels{e} = template_labels{template_idx(e)};
        else
            labels{e} = sprintf('X%d', e);
        end
    end

    aligned_maps = estimated_maps;
    for e = 1:n_est
        aligned_maps(e, :) = polarity(e) * aligned_maps(e, :);
    end

    strong = template_corr >= p.Results.strong_threshold;
    alignment = struct( ...
        'labels', {labels}, ...
        'correlations', template_corr, ...
        'template_indices', template_idx, ...
        'polarity', polarity, ...
        'aligned_maps', aligned_maps, ...
        'corr_matrix', corr_matrix, ...
        'template_labels', {template_labels}, ...
        'channel_match_mode', channel_match_mode, ...
        'matched_channel_labels', {common_labels}, ...
        'estimated_channel_indices', idx_est, ...
        'template_channel_indices', idx_template, ...
        'n_common_channels', numel(idx_est), ...
        'mean_correlation', mean(template_corr(template_corr > 0)), ...
        'median_correlation', median(template_corr(template_corr > 0)), ...
        'min_correlation', min(template_corr(template_corr > 0)), ...
        'n_strong_matches', sum(strong), ...
        'strong_threshold', p.Results.strong_threshold);
end

function [estimated_common, template_common, mode, idx_est, idx_template] = use_first_common_channels(estimated_maps, template_maps)
    n_common = min(size(estimated_maps, 2), size(template_maps, 2));
    idx_est = 1:n_common;
    idx_template = 1:n_common;
    estimated_common = estimated_maps(:, idx_est);
    template_common = template_maps(:, idx_template);
    mode = 'first_common_channels';
end
