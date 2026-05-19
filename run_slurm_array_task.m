function T = run_slurm_array_task()
%RUN_SLURM_ARRAY_TASK Run one simulated-EEG condition from a Slurm job array.
%
% Environment variables understood by this wrapper:
%   OUT_ROOT, REPS, K_TRUE_VALS, SNR_DBS, OVERLAP_PROBS, K_CANDIDATES,
%   DURATION_S, SFREQ, MONTAGES, N_WORKERS, SPM_PATH, SET_FILE,
%   VALIDATE_SIMULATION, TASKS_PER_JOB.

    task_id_text = getenv_default('SLURM_ARRAY_TASK_ID', '');
    if isempty(task_id_text)
        task_id_text = getenv_default('ARRAY_TASK_ID', '1');
    end
    task_id = str2double(task_id_text);
    tasks_per_job = str2double(getenv_default('TASKS_PER_JOB', '1'));
    if ~isfinite(task_id) || task_id < 1
        error('Invalid array task id: %s', task_id_text);
    end
    if ~isfinite(tasks_per_job) || tasks_per_job < 1
        error('Invalid TASKS_PER_JOB: %s', getenv('TASKS_PER_JOB'));
    end
    tasks_per_job = round(tasks_per_job);

    reps = parse_num_vector(getenv_default('REPS', '1:8'));
    k_true_vals = parse_num_vector(getenv_default('K_TRUE_VALS', '4 5 6 7'));
    snr_dbs = parse_num_vector(getenv_default('SNR_DBS', '-9 -3 0 1 3'));
    overlap_probs = parse_num_vector(getenv_default('OVERLAP_PROBS', '0 0.5 1.0'));
    k_candidates = parse_num_vector(getenv_default('K_CANDIDATES', '2:10'));
    duration_s = str2double(getenv_default('DURATION_S', '300'));
    sfreq = str2double(getenv_default('SFREQ', '250'));
    n_workers = str2double(getenv_default('N_WORKERS', '1'));
    validate_simulation = parse_bool(getenv_default('VALIDATE_SIMULATION', 'true'));

    montages = parse_cellstr(getenv_default('MONTAGES', 'full,10-20-20,10-20-12'));
    spm_path = getenv_default('SPM_PATH', '');
    set_file = getenv_default('SET_FILE', 'MetaMaps_2023_06.set');
    out_root = getenv_default('OUT_ROOT', fullfile('Output', 'slurm_sim'));

    grid = build_condition_grid(reps, k_true_vals, snr_dbs, overlap_probs);
    n_tasks = size(grid, 1);
    first_idx = (task_id - 1) * tasks_per_job + 1;
    last_idx = min(task_id * tasks_per_job, n_tasks);
    if first_idx > n_tasks
        fprintf('Array task %d has no assigned conditions; grid has %d conditions.\n', task_id, n_tasks);
        T = table();
        return;
    end

    fprintf('Slurm array task %d running condition indices %d:%d of %d.\n', ...
        task_id, first_idx, last_idx, n_tasks);
    fprintf('Total array jobs needed for this grid: %d\n', ceil(n_tasks / tasks_per_job));

    outputs = cell(last_idx - first_idx + 1, 1);
    for ii = first_idx:last_idx
        rep = grid(ii, 1);
        k_true = grid(ii, 2);
        snr_db = grid(ii, 3);
        overlap_prob = grid(ii, 4);
        out_dir = fullfile(out_root, sprintf('task_%04d_rep%03d_K%02d_SNR%+03d_ovl%03d', ...
            ii, rep, k_true, round(snr_db), round(100 * overlap_prob)));

        fprintf('\n=== Condition %d/%d: rep=%d K=%d SNR=%+.1f overlap=%.2f ===\n', ...
            ii, n_tasks, rep, k_true, snr_db, overlap_prob);

        T_one = Bayesian_MS_Comparison_Pipeline( ...
            'out_dir', out_dir, ...
            'rep_vals', rep, ...
            'K_true_vals', k_true, ...
            'SNR_dbs', snr_db, ...
            'K_candidates', k_candidates, ...
            'duration_s', duration_s, ...
            'sfreq', sfreq, ...
            'set_file', set_file, ...
            'n_workers', n_workers, ...
            'montages', montages, ...
            'overlap_probs', overlap_prob, ...
            'spm_path', spm_path, ...
            'validate_simulation', validate_simulation, ...
            'verbose', true);
        outputs{ii - first_idx + 1} = T_one;
    end

    T = vertcat(outputs{:});
    combined_dir = fullfile(out_root, 'array_task_summaries');
    if ~exist(combined_dir, 'dir')
        mkdir(combined_dir);
    end
    writetable(T, fullfile(combined_dir, sprintf('array_task_%04d_results.csv', task_id)));
end

function grid = build_condition_grid(reps, k_true_vals, snr_dbs, overlap_probs)
    grid = [];
    for rep = reps
        for k_true = k_true_vals
            for snr_db = snr_dbs
                for overlap_prob = overlap_probs
                    grid(end+1, :) = [rep, k_true, snr_db, overlap_prob]; %#ok<AGROW>
                end
            end
        end
    end
end

function value = getenv_default(name, default_value)
    value = getenv(name);
    if isempty(value)
        value = default_value;
    end
end

function vals = parse_num_vector(text)
    text = strrep(strtrim(text), ',', ' ');
    if contains(text, ':')
        vals = str2num(text); %#ok<ST2NM>
    else
        vals = sscanf(text, '%f')';
    end
    if isempty(vals)
        error('Could not parse numeric vector: %s', text);
    end
end

function vals = parse_cellstr(text)
    parts = regexp(text, '[,;]', 'split');
    vals = strtrim(parts);
    vals = vals(~cellfun('isempty', vals));
end

function tf = parse_bool(text)
    text = lower(strtrim(text));
    tf = any(strcmp(text, {'1', 'true', 'yes', 'y', 'on'}));
end
