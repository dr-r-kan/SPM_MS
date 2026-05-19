function save_microstate_json(Results, Sim, output_file, META)
% SAVE_MICROSTATE_JSON Save microstate analysis to a plotting-friendly JSON file.

    if nargin < 4
        META = struct();
    end

    C = infer_channel_count(Results, Sim);
    ch_labels_raw = infer_channel_labels(Sim, META, C);
    ch_labels = sanitize_channel_labels_json(ch_labels_raw(1:C));

    json_data = struct();

    % ===== TRUE MICROSTATES =====
    json_data.true_microstates = struct();
    if isfield(Results, 'maps_true') && ~isempty(Results.maps_true) && ...
            isfield(Results, 'K_true') && isfinite(Results.K_true) && Results.K_true > 0
        for k = 1:Results.K_true
            state_key = sprintf('state_%d', k);
            state_data = struct();
            for c = 1:C
                state_data.(ch_labels{c}) = double(real(Results.maps_true(k, c)));
            end
            json_data.true_microstates.(state_key) = state_data;
        end
    end

    % ===== ESTIMATED MICROSTATES =====
    json_data.estimated_microstates = struct();
    for k = 1:Results.K_estimated
        state_key = sprintf('state_%d', k);
        state_data = struct();
        for c = 1:C
            state_data.(ch_labels{c}) = double(real(Results.centers(k, c)));
        end

        if isfield(Results, 'cluster_weights') && numel(Results.cluster_weights) >= k
            state_data.confidence = double(real(Results.cluster_weights(k)));
        end
        if isfield(Results, 'template_alignment') && ~isempty(Results.template_alignment)
            state_data = add_template_state_fields(state_data, Results.template_alignment, k);
        end

        json_data.estimated_microstates.(state_key) = state_data;
    end

    % ===== METADATA =====
    json_data.metadata = struct();
    if isfield(Results, 'method'), json_data.metadata.method = Results.method; end
    if isfield(Results, 'criterion'), json_data.metadata.criterion = Results.criterion; end
    if isfield(Results, 'K_true'), json_data.metadata.K_true = Results.K_true; end
    if isfield(Results, 'K_estimated'), json_data.metadata.K_estimated = Results.K_estimated; end
    if isfield(Results, 'K_model_selected'), json_data.metadata.K_model_selected = Results.K_model_selected; end
    if isfield(Results, 'K_effective_vals'), json_data.metadata.K_effective_vals = Results.K_effective_vals; end
    if isfield(Results, 'polarity_feature_info')
        json_data.metadata.polarity_feature_info = Results.polarity_feature_info;
    end
    if isfield(Results, 'polarity_duplicate_info') && isfield(Results, 'K_candidates')
        dup_counts = zeros(size(Results.K_candidates));
        for i = 1:numel(Results.K_candidates)
            if numel(Results.polarity_duplicate_info) >= i && ...
                    ~isempty(Results.polarity_duplicate_info{i}) && ...
                    isfield(Results.polarity_duplicate_info{i}, 'pairs')
                dup_counts(i) = size(Results.polarity_duplicate_info{i}.pairs, 1);
            end
        end
        json_data.metadata.polarity_duplicate_pair_counts = dup_counts;
    end
    if isfield(Results, 'SNR_dB'), json_data.metadata.SNR_dB = Results.SNR_dB; end
    if isfield(Results, 'duration_s'), json_data.metadata.duration_s = Results.duration_s; end
    if isfield(Sim, 'sfreq'), json_data.metadata.sfreq = Sim.sfreq; end
    json_data.metadata.n_channels = C;
    if isfield(Results, 'n_maps'), json_data.metadata.n_samples_analyzed = Results.n_maps; end
    if isfield(Results, 'runtime'), json_data.metadata.runtime_s = Results.runtime; end

    if isfield(Results, 'best_criterion_value') && isfield(Results, 'criterion') && ...
            (strcmp(Results.criterion, 'free_energy') || strcmp(Results.method, 'spm_vb'))
        json_data.metadata.model_evidence_free_energy = Results.best_criterion_value;
    end
    if isfield(Results, 'template_alignment') && ~isempty(Results.template_alignment)
        json_data.metadata.template_alignment = template_alignment_summary(Results.template_alignment);
    end

    if ~isempty(META) && isstruct(META)
        mfields = fieldnames(META);
        for mf = 1:length(mfields)
            try
                json_data.metadata.(mfields{mf}) = META.(mfields{mf});
            catch
            end
        end
    end

    % ===== RECOVERY METRICS =====
    if isfield(Results, 'recovery_metrics') && ~isempty(Results.recovery_metrics)
        metrics = Results.recovery_metrics;
    else
        metrics = empty_recovery_metrics();
    end
    json_data.recovery = struct();
    json_data.recovery.n_matched = double(metrics.n_matched);
    json_data.recovery.sensitivity = double(metrics.sensitivity);
    json_data.recovery.precision = double(metrics.precision);
    json_data.recovery.f1_score = double(metrics.f1_score);
    json_data.recovery.mean_recovery_matched = double(metrics.mean_recovery_matched);
    json_data.recovery.mean_recovery_padded = double(metrics.mean_recovery_padded);

    % ===== CHANNEL INFORMATION =====
    json_data.channel_info = struct();
    json_data.channel_info.labels = ch_labels_raw(1:C);
    json_data.channel_info.labels_sanitized = ch_labels;
    json_data.channel_info.n_channels = C;

    % ===== CHANNEL POSITIONS =====
    json_data.electrode_positions = struct();
    if isfield(Sim, 'pos') && ~isempty(Sim.pos) && size(Sim.pos, 1) >= C
        for c = 1:C
            json_data.electrode_positions.(ch_labels{c}) = struct( ...
                'x', double(real(Sim.pos(c, 1))), ...
                'y', double(real(Sim.pos(c, 2))), ...
                'z', double(real(Sim.pos(c, 3))));
        end
    end

    % ===== MATCHING INFORMATION =====
    if isfield(metrics, 'match_assignment') && ~isempty(metrics.match_assignment)
        json_data.matches = struct();
        for i = 1:size(metrics.match_assignment, 1)
            est_idx = metrics.match_assignment(i, 1);
            true_idx = metrics.match_assignment(i, 2);
            similarity = metrics.match_similarities(i);
            match_key = sprintf('match_%d', i);
            json_data.matches.(match_key) = struct( ...
                'estimated_state', double(est_idx), ...
                'true_state', double(true_idx), ...
                'similarity', double(real(similarity)));
        end
    else
        json_data.matches = struct();
    end

    json_data = clean_complex_values(json_data);

    try
        json_str = jsonencode(json_data);
    catch ME
        fprintf('ERROR: JSON encoding failed: %s\n', ME.message);
        fid = fopen(output_file, 'w');
        fprintf(fid, '{\n');
        fprintf(fid, '  "error": "JSON encoding failed",\n');
        fprintf(fid, '  "reason": "%s"\n', ME.message);
        fprintf(fid, '}\n');
        fclose(fid);
        return;
    end

    fid = fopen(output_file, 'w');
    if fid < 0
        error('Could not open output JSON file: %s', output_file);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', json_str);
end

function C = infer_channel_count(Results, Sim)
    if isfield(Results, 'centers') && ~isempty(Results.centers)
        C = size(Results.centers, 2);
    elseif isfield(Results, 'maps_true') && ~isempty(Results.maps_true)
        C = size(Results.maps_true, 2);
    elseif isfield(Sim, 'X_noisy') && ~isempty(Sim.X_noisy)
        C = size(Sim.X_noisy, 1);
    else
        C = 0;
    end
end

function ch_labels_raw = infer_channel_labels(Sim, META, C)
    ch_labels_raw = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        ch_labels_raw = cellstr(Sim.channel_labels);
    elseif isfield(Sim, 'chanlocs') && ~isempty(Sim.chanlocs)
        ncl = length(Sim.chanlocs);
        ch_labels_raw = cell(ncl, 1);
        for i = 1:ncl
            if isfield(Sim.chanlocs(i), 'labels') && ~isempty(Sim.chanlocs(i).labels)
                ch_labels_raw{i} = char(Sim.chanlocs(i).labels);
            else
                ch_labels_raw{i} = sprintf('Ch%03d', i);
            end
        end
    elseif isfield(META, 'channel_labels') && ~isempty(META.channel_labels)
        ch_labels_raw = cellstr(META.channel_labels);
    end
    if isempty(ch_labels_raw) || length(ch_labels_raw) < C
        ch_labels_raw = arrayfun(@(i) sprintf('Ch%03d', i), 1:C, 'UniformOutput', false);
    end
end

function state_data = add_template_state_fields(state_data, ta, k)
    if isfield(ta, 'labels') && numel(ta.labels) >= k
        state_data.template_label = ta.labels{k};
    end
    if isfield(ta, 'correlations') && numel(ta.correlations) >= k
        state_data.template_correlation = double(real(ta.correlations(k)));
    end
    if isfield(ta, 'polarity') && numel(ta.polarity) >= k
        state_data.template_polarity = double(real(ta.polarity(k)));
    end
end

function s = template_alignment_summary(ta)
    s = struct();
    fields = {'mean_correlation', 'median_correlation', 'min_correlation', ...
        'n_strong_matches', 'strong_threshold', 'n_common_channels'};
    for i = 1:numel(fields)
        f = fields{i};
        if isfield(ta, f)
            s.(f) = double(real(ta.(f)));
        end
    end
    if isfield(ta, 'channel_match_mode')
        s.channel_match_mode = ta.channel_match_mode;
    end
    if isfield(ta, 'labels')
        s.labels = ta.labels;
    end
    if isfield(ta, 'correlations')
        s.correlations = double(real(ta.correlations(:)'));
    end
    if isfield(ta, 'template_labels')
        s.template_labels = ta.template_labels;
    end
end

function metrics = empty_recovery_metrics()
    metrics = struct( ...
        'n_matched', 0, ...
        'sensitivity', NaN, ...
        'precision', NaN, ...
        'f1_score', NaN, ...
        'mean_recovery_matched', NaN, ...
        'mean_recovery_padded', NaN, ...
        'match_similarities', []);
end

function s = clean_complex_values(s)
    if isstruct(s)
        fields = fieldnames(s);
        for f = 1:length(fields)
            field_name = fields{f};
            field_val = s.(field_name);
            if isstruct(field_val)
                s.(field_name) = clean_complex_values(field_val);
            elseif iscell(field_val)
                for c = 1:numel(field_val)
                    if isstruct(field_val{c})
                        field_val{c} = clean_complex_values(field_val{c});
                    elseif isnumeric(field_val{c})
                        field_val{c} = double(real(field_val{c}));
                    end
                end
                s.(field_name) = field_val;
            elseif isnumeric(field_val)
                s.(field_name) = double(real(field_val));
            end
        end
    elseif iscell(s)
        for c = 1:numel(s)
            if isstruct(s{c})
                s{c} = clean_complex_values(s{c});
            elseif isnumeric(s{c})
                s{c} = double(real(s{c}));
            end
        end
    end
end

function sanitized = sanitize_channel_labels_json(ch_labels)
% SANITIZE_CHANNEL_LABELS_JSON Convert channel labels to valid struct fields.

    sanitized = cell(size(ch_labels));
    for i = 1:length(ch_labels)
        label = ch_labels{i};
        if ~ischar(label)
            label = char(label);
        end
        label = regexprep(label, '[-/\\\s\.\,\(\)\[\]\{\}]', '_');
        label = regexprep(label, '^_+|_+$', '');
        if isempty(label) || isempty(regexp(label(1), '[A-Za-z]', 'once'))
            label = ['Ch' label];
        end
        sanitized{i} = matlab.lang.makeValidName(label);
    end
end
