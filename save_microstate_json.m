function save_microstate_json(Results, Sim, output_file, META)
% SAVE_MICROSTATE_JSON: Save microstate analysis to JSON format
%
% Saves all microstate data in a clean JSON structure for Python processing
%
% INPUTS:
%   Results     - Results structure from fitting function
%   Sim         - Simulation structure
%   output_file - Output JSON file path

    % Determine channel count and labels from Sim when available
    C = size(Results.maps_true, 2);

    % Try to get human-readable labels from Sim (preferred)
    ch_labels_raw = {};
    if isfield(Sim, 'channel_labels') && ~isempty(Sim.channel_labels)
        ch_labels_raw = Sim.channel_labels;
    elseif isfield(Sim, 'chanlocs') && ~isempty(Sim.chanlocs)
        try
            ncl = length(Sim.chanlocs);
            ch_labels_raw = cell(ncl, 1);
            for i = 1:ncl
                if isfield(Sim.chanlocs(i), 'labels') && ~isempty(Sim.chanlocs(i).labels)
                    ch_labels_raw{i} = Sim.chanlocs(i).labels;
                else
                    ch_labels_raw{i} = sprintf('Ch%03d', i);
                end
            end
        catch
            ch_labels_raw = {};
        end
    elseif nargin > 3 && isfield(META, 'channel_labels') && ~isempty(META.channel_labels)
        % Fallback: try META structure if provided
        ch_labels_raw = META.channel_labels;
    end

    % If still empty or length mismatch, fall back to generic labels
    if isempty(ch_labels_raw) || length(ch_labels_raw) < C
        ch_labels_raw = arrayfun(@(i) sprintf('Ch%03d', i), 1:C, 'UniformOutput', false);
    end

    % Create sanitized labels for use as struct field names (matches plotting helper)
    ch_labels = sanitize_channel_labels_json(ch_labels_raw(1:C));

    
    % Create output structure
    json_data = struct();
    
    % ===== TRUE MICROSTATES =====
    json_data.true_microstates = struct();
    for k = 1:Results.K_true
        state_key = sprintf('state_%d', k);
        state_data = struct();
        for c = 1:C
            % ✅ FIX: Ensure real values
            val = Results.maps_true(k, c);
            if ~isreal(val)
                val = real(val);
            end
            % Use sanitized label as JSON key
            state_data.(ch_labels{c}) = double(val);
        end
        json_data.true_microstates.(state_key) = state_data;
    end
    
    % ===== ESTIMATED MICROSTATES =====
    json_data.estimated_microstates = struct();
    for k = 1:Results.K_estimated
        state_key = sprintf('state_%d', k);
        state_data = struct();
        for c = 1:C
            % ✅ FIX: Ensure real values
            val = Results.centers(k, c);
            if ~isreal(val)
                val = real(val);
            end
            % Use sanitized label as JSON key
            state_data.(ch_labels{c}) = double(val);
        end
        
        % Add confidence measure if available
        if isfield(Results, 'cluster_weights') && length(Results.cluster_weights) >= k
            state_data.confidence = Results.cluster_weights(k);
        end
        
        json_data.estimated_microstates.(state_key) = state_data;
    end
    
    % ===== METADATA =====
    json_data.metadata = struct();

    % Populate from Results where available
    if isfield(Results, 'method'), json_data.metadata.method = Results.method; end
    if isfield(Results, 'criterion'), json_data.metadata.criterion = Results.criterion; end
    if isfield(Results, 'K_true'), json_data.metadata.K_true = Results.K_true; end
    if isfield(Results, 'K_estimated'), json_data.metadata.K_estimated = Results.K_estimated; end
    if isfield(Results, 'SNR_dB'), json_data.metadata.SNR_dB = Results.SNR_dB; end
    if isfield(Results, 'duration_s'), json_data.metadata.duration_s = Results.duration_s; end
    if isfield(Sim, 'sfreq'), json_data.metadata.sfreq = Sim.sfreq; end
    json_data.metadata.n_channels = C;
    if isfield(Results, 'n_maps'), json_data.metadata.n_samples_analyzed = Results.n_maps; end
    if isfield(Results, 'runtime'), json_data.metadata.runtime_s = Results.runtime; end
    
    % Add Free Energy if available (Global Model Confidence)
    if isfield(Results, 'best_criterion_value') && ...
            (strcmp(Results.criterion, 'free_energy') || strcmp(Results.method, 'spm_vb'))
        json_data.metadata.model_evidence_free_energy = Results.best_criterion_value;
    end
    if isfield(Results, 'free_energy')
        % If full vector is available, save the one corresponding to K_estimated
        % (This might be redundant with best_criterion_value but explicit)
    end

    % Merge optional metadata (pipeline-level) if provided - values in META override previous
    if nargin >= 4 && ~isempty(META) && isstruct(META)
        mfields = fieldnames(META);
        for mf = 1:length(mfields)
            try
                json_data.metadata.(mfields{mf}) = META.(mfields{mf});
            catch
            end
        end
    end
    
    % ===== RECOVERY METRICS =====
    metrics = Results.recovery_metrics;
    json_data.recovery = struct();
    json_data.recovery.n_matched = double(metrics.n_matched);
    json_data.recovery.sensitivity = double(metrics.sensitivity);
    json_data.recovery.precision = double(metrics.precision);
    json_data.recovery.f1_score = double(metrics.f1_score);
    json_data.recovery.mean_recovery_matched = double(metrics.mean_recovery_matched);
    json_data.recovery.mean_recovery_padded = double(metrics.mean_recovery_padded);
    
    % ===== CHANNEL INFORMATION =====
    json_data.channel_info = struct();
    % Preserve original (raw) labels and provide sanitized labels for consumers
    json_data.channel_info.labels = ch_labels_raw(1:C);
    json_data.channel_info.labels_sanitized = ch_labels;
    json_data.channel_info.n_channels = C;
    
    % ===== CHANNEL POSITIONS =====
    json_data.electrode_positions = struct();
    for c = 1:C
        pos_x = Sim.pos(c, 1);
        pos_y = Sim.pos(c, 2);
        pos_z = Sim.pos(c, 3);
        
        % ✅ FIX: Ensure real values
        if ~isreal(pos_x), pos_x = real(pos_x); end
        if ~isreal(pos_y), pos_y = real(pos_y); end
        if ~isreal(pos_z), pos_z = real(pos_z); end
        
        % Use sanitized label as key but preserve raw labels above
        json_data.electrode_positions.(ch_labels{c}) = struct(...
            'x', double(pos_x), ...
            'y', double(pos_y), ...
            'z', double(pos_z));
    end
    
    % ===== MATCHING INFORMATION =====
    if isfield(metrics, 'match_assignment') && ~isempty(metrics.match_assignment)
        json_data.matches = struct();
        for i = 1:size(metrics.match_assignment, 1)
            est_idx = metrics.match_assignment(i, 1);
            true_idx = metrics.match_assignment(i, 2);
            similarity = metrics.match_similarities(i);
            
            % ✅ FIX: Ensure real values
            if ~isreal(similarity)
                similarity = real(similarity);
            end
            
            match_key = sprintf('match_%d', i);
            json_data.matches.(match_key) = struct(...
                'estimated_state', double(est_idx), ...
                'true_state', double(true_idx), ...
                'similarity', double(similarity));
        end
    else
        json_data.matches = struct();
    end
    
    % ✅ FIX: Recursively clean any complex values before encoding
    json_data = clean_complex_values(json_data);
    
    % Convert to JSON and save
    try
        json_str = jsonencode(json_data);
    catch ME
        fprintf('ERROR: JSON encoding failed: %s\n', ME.message);
        fprintf('File: %s\n', output_file);
        fprintf('Attempting alternative serialization...\n');
        
        % Last resort: save as text
        fid = fopen(output_file, 'w');
        fprintf(fid, '{\n');
        fprintf(fid, '  "error": "JSON encoding failed",\n');
        fprintf(fid, '  "reason": "%s",\n', ME.message);
        fprintf(fid, '  "K_true": %d,\n', Results.K_true);
        fprintf(fid, '  "K_estimated": %d\n', Results.K_estimated);
        fprintf(fid, '}\n');
        fclose(fid);
        return;
    end
    
    fid = fopen(output_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

function s = clean_complex_values(s)
% ✅ Recursively clean complex values from struct
    
    if isstruct(s)
        fields = fieldnames(s);
        for f = 1:length(fields)
            field_name = fields{f};
            field_val = s.(field_name);
            
            if isstruct(field_val)
                % Recurse into nested struct
                s.(field_name) = clean_complex_values(field_val);
            elseif iscell(field_val)
                % Process cell array
                for c = 1:numel(field_val)
                    if isstruct(field_val{c})
                        field_val{c} = clean_complex_values(field_val{c});
                    elseif isnumeric(field_val{c})
                        if ~isreal(field_val{c})
                            field_val{c} = real(field_val{c});
                        end
                        field_val{c} = double(field_val{c});
                    end
                end
                s.(field_name) = field_val;
            elseif isnumeric(field_val)
                % Clean numeric values
                if ~isreal(field_val)
                    field_val = real(field_val);
                end
                s.(field_name) = double(field_val);
            end
        end
    elseif iscell(s)
        for c = 1:numel(s)
            if isstruct(s{c})
                s{c} = clean_complex_values(s{c});
            elseif isnumeric(s{c}) && ~isreal(s{c})
                s{c} = real(s{c});
            end
        end
    end
end

function sanitized = sanitize_channel_labels_json(ch_labels)
% SANITIZE_CHANNEL_LABELS_JSON: Convert channel labels to valid struct field names
% Ensures labels are valid JSON object keys when used as MATLAB struct fields.

    sanitized = cell(size(ch_labels));
    for i = 1:length(ch_labels)
        label = ch_labels{i};
        if ~ischar(label)
            label = char(label);
        end

        % Replace problematic characters with underscore
        label = regexprep(label, '[-/\\\s\.\,\(\)\[\]\{\}]', '_');

        % Remove leading/trailing underscores
        label = regexprep(label, '^_+|_+$', '');

        % If empty or starts with digit, prefix with 'Ch'
        if isempty(label) || (~isempty(label) && ~isempty(regexp(label(1), '[A-Za-z]', 'once'))==0)
            label = ['Ch' label];
        end

        % Ensure it is a valid MATLAB field name (no spaces, starts with letter)
        label = matlab.lang.makeValidName(label);

        sanitized{i} = label;
    end
end