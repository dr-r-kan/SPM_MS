function [Tall, Tsummary] = combine_slurm_array_results(out_root)
%COMBINE_SLURM_ARRAY_RESULTS Combine per-array-task simulation outputs.

    if nargin < 1 || isempty(out_root)
        out_root = getenv('OUT_ROOT');
        if isempty(out_root)
            out_root = fullfile('Output', 'slurm_sim_all_methods');
        end
    end

    files = dir(fullfile(out_root, 'task_*', 'results', 'comparison_results.csv'));
    if isempty(files)
        files = dir(fullfile(out_root, 'array_task_summaries', 'array_task_*_results.csv'));
    end
    if isempty(files)
        error('No comparison_results.csv files found below: %s', out_root);
    end

    tables = cell(numel(files), 1);
    for i = 1:numel(files)
        tables{i} = readtable(fullfile(files(i).folder, files(i).name));
    end
    Tall = vertcat(tables{:});

    combined_dir = fullfile(out_root, 'combined_results');
    if ~exist(combined_dir, 'dir')
        mkdir(combined_dir);
    end

    combined_csv = fullfile(combined_dir, 'comparison_results_all_tasks.csv');
    writetable(Tall, combined_csv);

    Tsummary = groupsummary(Tall, {'method', 'criterion'}, {'mean', 'std'}, {'K_correct', 'K_error'});
    Tsummary.accuracy_pct = 100 * Tsummary.mean_K_correct;
    summary_csv = fullfile(combined_dir, 'k_selection_summary_by_method_criterion.csv');
    writetable(Tsummary, summary_csv);

    fprintf('Combined %d files (%d rows).\n', numel(files), height(Tall));
    fprintf('Saved: %s\n', combined_csv);
    fprintf('Saved: %s\n', summary_csv);
end
