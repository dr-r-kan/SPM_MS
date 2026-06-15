function out = plot_simulated_backfit_confusion_summary(varargin)
%PLOT_SIMULATED_BACKFIT_CONFUSION_SUMMARY Summarise simulated backfit confusions.
%
% Produces a two-panel figure for the simulated microstate retrieval experiment:
%   Left: template-label confusion rates averaged across K-gap groups.
%   Right: mean backfit label accuracy for kmeans and SPM-VB at K_gap = 0:3.

    util = microstate_utilities();
    repo_cfg = util.load_config();

    p = inputParser;
    addParameter(p, 'output_dir', char(repo_cfg.simulation.out_dir), @(x) ischar(x) || isstring(x));
    addParameter(p, 'results_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', char(repo_cfg.paths.template_file), @(x) ischar(x) || isstring(x));
    addParameter(p, 'methods', {'kmeans_koenig', 'spm_vb'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'criteria', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'k_gaps', 0:3, @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'visible', false, @islogical);
    addParameter(p, 'resolution', double(util.get_field(repo_cfg.plotting, 'resolution', 300)), ...
        @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});

    cfg = p.Results;
    cfg.output_dir = util.resolve_path(char(cfg.output_dir), util.project_root());
    cfg.template_file = util.resolve_path(char(cfg.template_file), util.project_root());
    cfg.methods = cellstr(string(cfg.methods));
    cfg.criteria = cellstr(string(cfg.criteria));
    cfg.k_gaps = unique(double(cfg.k_gaps(:))', 'stable');

    if isempty(cfg.results_csv)
        cfg.results_csv = fullfile(cfg.output_dir, 'results', 'comparison_results.csv');
    else
        cfg.results_csv = util.resolve_path(char(cfg.results_csv), util.project_root());
    end
    if isempty(cfg.output_file)
        plot_dir = fullfile(cfg.output_dir, 'analysis_plots');
        util.ensure_dir(plot_dir);
        cfg.output_file = fullfile(plot_dir, 'backfit_confusion_summary.png');
    else
        cfg.output_file = util.resolve_path(char(cfg.output_file), util.project_root());
        out_dir = fileparts(cfg.output_file);
        if ~isempty(out_dir)
            util.ensure_dir(out_dir);
        end
    end

    if ~isfile(cfg.results_csv)
        error('Results manifest not found: %s', cfg.results_csv);
    end
    if ~isfile(cfg.template_file)
        error('Template file not found: %s', cfg.template_file);
    end

    T = readtable(cfg.results_csv, 'TextType', 'string');
    required_vars = {'method', 'criterion', 'K_true', 'K_estimated', 'backfit_diagnostic_file'};
    missing_vars = required_vars(~ismember(required_vars, T.Properties.VariableNames));
    if ~isempty(missing_vars)
        error('Results manifest is missing required columns: %s', strjoin(missing_vars, ', '));
    end

    method_mask = ismember(strtrim(lower(string(T.method))), strtrim(lower(string(cfg.methods))));
    if isempty(cfg.criteria)
        criterion_mask = true(height(T), 1);
    else
        criterion_mask = ismember(strtrim(lower(string(T.criterion))), strtrim(lower(string(cfg.criteria))));
    end
    file_vals = cellstr(string(T.backfit_diagnostic_file));
    valid_file = ~cellfun(@isempty, file_vals) & cellfun(@isfile, file_vals);
    if ismember('K_gap', T.Properties.VariableNames)
        gap_vals = double(T.K_gap);
    else
        gap_vals = double(T.K_true) - double(T.K_estimated);
    end
    gap_mask = ismember(gap_vals, cfg.k_gaps);
    keep_mask = method_mask & criterion_mask & valid_file & gap_mask;
    if ~any(keep_mask)
        error('No valid backfit diagnostic files matched the requested filters.');
    end

    T_plot = T(keep_mask, :);
    gap_vals = gap_vals(keep_mask);
    file_vals = file_vals(keep_mask);
    method_vals = cellstr(lower(strtrim(string(T_plot.method))));

    diagnostics = struct('file', {}, 'method', {}, 'k_gap', {}, 'labels', {}, 'counts', {}, ...
        'accuracy', {}, 'run_name', {});
    label_order = {};
    for i = 1:height(T_plot)
        file_i = char(file_vals{i});
        S = load(file_i, 'BackfitDiagnostics');
        if ~isfield(S, 'BackfitDiagnostics') || ~isfield(S.BackfitDiagnostics, 'ok') || ~S.BackfitDiagnostics.ok
            continue;
        end

        diag_i = S.BackfitDiagnostics;
        labels_i = cellstr(string(diag_i.template_labels(:)));
        counts_i = double(diag_i.confusion_counts);
        if isempty(labels_i) || isempty(counts_i)
            continue;
        end

        label_order = merge_label_order(label_order, labels_i);
        diagnostics(end + 1) = struct( ...
            'file', file_i, ...
            'method', method_vals{i}, ...
            'k_gap', double(gap_vals(i)), ...
            'labels', {labels_i}, ...
            'counts', counts_i, ...
            'accuracy', local_accuracy(diag_i), ...
            'run_name', char(string(T_plot.subject(i)))); %#ok<AGROW>
    end

    if isempty(diagnostics)
        error('No usable backfit diagnostics were found in the selected rows.');
    end

    if isempty(label_order)
        label_order = diagnostics(1).labels;
    end

    k_gap_values = cfg.k_gaps;
    gap_confusion_means = cell(numel(k_gap_values), 1);
    for g = 1:numel(k_gap_values)
        gap_mask_g = [diagnostics.k_gap] == k_gap_values(g);
        if ~any(gap_mask_g)
            gap_confusion_means{g} = nan(numel(label_order), numel(label_order));
            continue;
        end

        matrices = cell(1, sum(gap_mask_g));
        idx = 0;
        for d = find(gap_mask_g)
            idx = idx + 1;
            matrices{idx} = normalize_confusion_rows(reorder_confusion_matrix( ...
                diagnostics(d).counts, diagnostics(d).labels, label_order));
        end
        gap_confusion_means{g} = mean(cat(3, matrices{:}), 3, 'omitnan');
    end

    valid_gap_mats = gap_confusion_means(~cellfun(@(x) isempty(x) || all(isnan(x), 'all'), gap_confusion_means));
    if isempty(valid_gap_mats)
        overall_confusion = nan(numel(label_order), numel(label_order));
    else
        overall_confusion = mean(cat(3, valid_gap_mats{:}), 3, 'omitnan');
    end

    method_labels = {'kmeans_koenig', 'spm_vb'};
    method_titles = {'K-means', 'SPM-VB'};
    accuracy_summary = nan(numel(method_labels), numel(k_gap_values));
    accuracy_counts = zeros(numel(method_labels), numel(k_gap_values));
    for m = 1:numel(method_labels)
        for g = 1:numel(k_gap_values)
            mask = strcmp({diagnostics.method}, method_labels{m}) & [diagnostics.k_gap] == k_gap_values(g);
            if ~any(mask)
                continue;
            end
            accuracy_summary(m, g) = mean([diagnostics(mask).accuracy], 'omitnan');
            accuracy_counts(m, g) = sum(mask);
        end
    end

    fig = figure('Name', 'Simulated backfit confusion summary', ...
        'Color', 'white', 'Visible', char(util.on_off_string(cfg.visible)), 'NumberTitle', 'off', ...
        'Position', [70, 70, 1700, 820]);
    if ~cfg.visible
        cleaner = onCleanup(@() close_if_valid(fig)); %#ok<NASGU>
    end

    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(tl, 'Simulated backfit confusion summary', 'FontWeight', 'bold', 'FontSize', 15);

    ax_left = nexttile(tl, 1);
    render_confusion_panel(ax_left, overall_confusion, label_order, ...
        sprintf('Template-label confusion rate\n(mean across K-gap groups; n=%d runs)', numel(diagnostics)));

    ax_right = nexttile(tl, 2);
    render_accuracy_panel(ax_right, accuracy_summary, accuracy_counts, k_gap_values, method_titles);

    exportgraphics(fig, cfg.output_file, 'Resolution', cfg.resolution);

    out = struct();
    out.plot_file = cfg.output_file;
    out.results_csv = cfg.results_csv;
    out.template_file = cfg.template_file;
    out.label_order = label_order(:);
    out.k_gap_values = k_gap_values(:)';
    out.overall_confusion = overall_confusion;
    out.accuracy_summary = accuracy_summary;
    out.accuracy_counts = accuracy_counts;
    out.n_runs = numel(diagnostics);
end

function accuracy = local_accuracy(diag_i)
    accuracy = NaN;
    if isfield(diag_i, 'hard') && isstruct(diag_i.hard) && isfield(diag_i.hard, 'label_top1_accuracy')
        accuracy = double(diag_i.hard.label_top1_accuracy);
    end
end

function labels_out = merge_label_order(labels_a, labels_b)
    labels_out = labels_a(:);
    for i = 1:numel(labels_b)
        if ~any(strcmp(labels_out, labels_b{i}))
            labels_out{end + 1, 1} = labels_b{i}; %#ok<AGROW>
        end
    end
end

function matrix_out = reorder_confusion_matrix(matrix_in, labels_in, label_order)
    n = numel(label_order);
    matrix_out = nan(n, n);
    if isempty(matrix_in) || isempty(labels_in)
        return;
    end

    labels_in = cellstr(string(labels_in(:)));
    for r = 1:n
        src_r = find(strcmp(labels_in, label_order{r}), 1, 'first');
        if isempty(src_r)
            continue;
        end
        for c = 1:n
            src_c = find(strcmp(labels_in, label_order{c}), 1, 'first');
            if isempty(src_c)
                continue;
            end
            matrix_out(r, c) = matrix_in(src_r, src_c);
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
        else
            rownorm(r, finite_mask) = NaN;
        end
    end
end

function render_confusion_panel(ax, values, label_order, title_str)
    axes(ax);
    cla(ax);
    if isempty(values) || all(~isfinite(values), 'all')
        text(0.5, 0.5, 'No data', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 14);
        axis(ax, 'off');
        return;
    end

    imagesc(ax, values, [0 1]);
    axis(ax, 'square');
    colormap(ax, parula(256));
    cb = colorbar(ax);
    cb.Label.String = 'Confusion rate';
    cb.Label.FontWeight = 'bold';
    set(ax, 'XTick', 1:numel(label_order), 'XTickLabel', label_order, ...
        'YTick', 1:numel(label_order), 'YTickLabel', label_order, ...
        'XTickLabelRotation', 45, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
        'FontSize', 10);
    xlabel(ax, 'Estimated template label', 'FontWeight', 'bold');
    ylabel(ax, 'True template label', 'FontWeight', 'bold');
    title(ax, title_str, 'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none');

    for r = 1:size(values, 1)
        for c = 1:size(values, 2)
            v = values(r, c);
            if ~isfinite(v)
                continue;
            end
            text_color = [0 0 0];
            if v >= 0.55
                text_color = [1 1 1];
            end
            text(ax, c, r, sprintf('%.2f', v), 'HorizontalAlignment', 'center', ...
                'Color', text_color, 'FontWeight', 'bold', 'FontSize', 9);
        end
    end
end

function render_accuracy_panel(ax, accuracy_summary, accuracy_counts, k_gap_values, method_titles)
    axes(ax);
    cla(ax);
    hold(ax, 'on');

    x = categorical(compose('K_{true}-%d', k_gap_values));
    x = reordercats(x, cellstr(compose('K_{true}-%d', k_gap_values)));
    b = bar(ax, x, accuracy_summary', 'grouped', 'LineWidth', 1);
    b(1).FaceColor = [0.20 0.46 0.70];
    b(2).FaceColor = [0.87 0.48 0.16];

    ylim(ax, [0 1]);
    grid(ax, 'on');
    ax.GridAlpha = 0.25;
    ax.LineWidth = 1;
    ax.Color = 'white';
    ylabel(ax, 'Mean correct backfit proportion', 'FontWeight', 'bold');
    xlabel(ax, 'K estimated relative to K true', 'FontWeight', 'bold');
    title(ax, 'Backfit accuracy by K-gap and method', 'FontWeight', 'bold', 'FontSize', 12);
    legend(ax, b, method_titles, 'Location', 'southoutside', 'Orientation', 'horizontal');

    for m = 1:size(accuracy_summary, 1)
        for g = 1:size(accuracy_summary, 2)
            v = accuracy_summary(m, g);
            if ~isfinite(v)
                continue;
            end
            x_pos = b(m).XEndPoints(g);
            text(ax, x_pos, v + 0.025, sprintf('%.2f\n(n=%d)', v, accuracy_counts(m, g)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    hold(ax, 'off');
end

function close_if_valid(fig)
    if ~isempty(fig) && isgraphics(fig)
        close(fig);
    end
end