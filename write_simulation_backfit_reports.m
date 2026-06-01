function manifest = write_simulation_backfit_reports(T, output_dir)
% WRITE_SIMULATION_BACKFIT_REPORTS Aggregate coverage and confusion outputs.

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    manifest = table();
    if ~istable(T) || height(T) == 0 || ...
            ~ismember('backfit_diagnostic_file', T.Properties.VariableNames)
        return;
    end

    valid_file = ~cellfun(@isempty, cellstr(string(T.backfit_diagnostic_file))) & ...
        cellfun(@isfile, cellstr(string(T.backfit_diagnostic_file)));
    if ~any(valid_file)
        return;
    end

    T_valid = T(valid_file, :);
    coverage_vars = intersect({'backfit_coverage_corr', 'backfit_coverage_spearman', ...
        'backfit_coverage_mae', 'backfit_coverage_rmse', 'backfit_coverage_l1'}, ...
        T_valid.Properties.VariableNames, 'stable');
    if ~isempty(coverage_vars)
        coverage_summary = groupsummary(T_valid, {'method', 'criterion', 'K_true', 'K_estimated'}, 'mean', coverage_vars);
        writetable(coverage_summary, fullfile(output_dir, 'backfit_coverage_summary.csv'));
    end

    underfit_mask = double(T_valid.K_true) > double(T_valid.K_estimated);
    T_under = T_valid(underfit_mask, :);
    if height(T_under) == 0
        return;
    end

    group_keys = strcat( ...
        string(T_under.method), "|", ...
        string(T_under.criterion), "|", ...
        string(T_under.K_true), "|", ...
        string(T_under.K_estimated));
    [unique_keys, ~, key_idx] = unique(group_keys, 'stable');

    rows = cell(numel(unique_keys), 1);
    for g = 1:numel(unique_keys)
        group_rows = T_under(key_idx == g, :);
        [counts, labels] = aggregate_confusions(group_rows.backfit_diagnostic_file);
        if isempty(counts)
            continue;
        end
        row_norm = counts ./ max(sum(counts, 2), 1);

        method = char(string(group_rows.method(1)));
        criterion = char(string(group_rows.criterion(1)));
        K_true = double(group_rows.K_true(1));
        K_estimated = double(group_rows.K_estimated(1));
        file_stub = sprintf('%s__%s__Ktrue%d__Kest%d__delta%d', ...
            sanitize_stub(method), sanitize_stub(criterion), K_true, K_estimated, K_true - K_estimated);

        counts_csv = fullfile(output_dir, [file_stub '_counts.csv']);
        rownorm_csv = fullfile(output_dir, [file_stub '_row_normalized.csv']);
        counts_png = fullfile(output_dir, [file_stub '_counts.png']);
        rownorm_png = fullfile(output_dir, [file_stub '_row_normalized.png']);

        writetable(array2table(counts, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), counts_csv, 'WriteRowNames', true);
        writetable(array2table(row_norm, 'VariableNames', matlab.lang.makeValidName(labels), 'RowNames', labels), rownorm_csv, 'WriteRowNames', true);
        plot_confusion_heatmap(counts, labels, counts_png, sprintf('%s | %s | K_{true}=%d, K_{est}=%d', method, criterion, K_true, K_estimated), 'Counts');
        plot_confusion_heatmap(row_norm, labels, rownorm_png, sprintf('%s | %s | K_{true}=%d, K_{est}=%d', method, criterion, K_true, K_estimated), 'Row-normalized');

        rows{g} = table( ...
            string(method), string(criterion), K_true, K_estimated, K_true - K_estimated, ...
            height(group_rows), string(counts_csv), string(rownorm_csv), string(counts_png), string(rownorm_png), ...
            'VariableNames', {'method', 'criterion', 'K_true', 'K_estimated', 'K_gap', 'n_runs', ...
            'counts_csv', 'row_normalized_csv', 'counts_png', 'row_normalized_png'});
    end

    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        return;
    end
    manifest = vertcat(rows{:});
    writetable(manifest, fullfile(output_dir, 'backfit_confusion_manifest.csv'));
end

function [counts_total, labels] = aggregate_confusions(files)
    counts_total = [];
    labels = {};
    for i = 1:numel(files)
        file_i = char(string(files{i}));
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
            [labels, counts_total] = merge_confusion_counts(labels, counts_total, labels_i, counts_i);
        end
    end
end

function [labels_out, counts_out] = merge_confusion_counts(labels_a, counts_a, labels_b, counts_b)
    labels_out = unique([labels_a(:); labels_b(:)], 'stable');
    n = numel(labels_out);
    counts_out = zeros(n, n);
    idx_a = map_label_indices(labels_out, labels_a);
    idx_b = map_label_indices(labels_out, labels_b);
    counts_out(idx_a, idx_a) = counts_out(idx_a, idx_a) + counts_a;
    counts_out(idx_b, idx_b) = counts_out(idx_b, idx_b) + counts_b;
end

function idx = map_label_indices(master, subset)
    idx = zeros(numel(subset), 1);
    for i = 1:numel(subset)
        idx(i) = find(strcmp(master, subset{i}), 1, 'first');
    end
end

function plot_confusion_heatmap(values, labels, output_file, title_str, colorbar_label)
    fig = figure('Visible', 'off', 'Position', [100 100 900 760], 'Color', 'white');
    imagesc(values);
    axis square;
    colormap(parula(256));
    cb = colorbar;
    ylabel(cb, colorbar_label, 'FontSize', 10);
    set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels, ...
        'YTick', 1:numel(labels), 'YTickLabel', labels, ...
        'XTickLabelRotation', 45, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    xlabel('Estimated template label', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('True template label', 'FontSize', 11, 'FontWeight', 'bold');
    title(title_str, 'Interpreter', 'none', 'FontSize', 12, 'FontWeight', 'bold');
    for r = 1:size(values, 1)
        for c = 1:size(values, 2)
            text(c, r, format_heatmap_value(values(r, c)), ...
                'HorizontalAlignment', 'center', 'Color', 'black', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    saveas(fig, output_file);
    close(fig);
end

function s = format_heatmap_value(v)
    if abs(v - round(v)) < 1e-9
        s = sprintf('%d', round(v));
    else
        s = sprintf('%.2f', v);
    end
end

function s = sanitize_stub(s)
    s = lower(char(string(s)));
    s = regexprep(s, '[^a-z0-9]+', '_');
    s = regexprep(s, '^_+|_+$', '');
end
