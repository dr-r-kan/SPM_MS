function save_microstate_json(Results, Sim, output_file)
% SAVE_MICROSTATE_JSON: Save microstate analysis to JSON format
%
% Saves all microstate data in a clean JSON structure for Python processing
%
% INPUTS:
%   Results     - Results structure from fitting function
%   Sim         - Simulation structure
%   output_file - Output JSON file path

    % Create channel labels
    C = size(Results.maps_true, 2);
    ch_labels = arrayfun(@(i) sprintf('Ch%03d', i), 1:C, 'UniformOutput', false);
    
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
            state_data.(ch_labels{c}) = double(val);
        end
        json_data.estimated_microstates.(state_key) = state_data;
    end
    
    % ===== METADATA =====
    json_data.metadata = struct();
    json_data.metadata.method = Results.method;
    json_data.metadata.criterion = Results.criterion;
    json_data.metadata.K_true = Results.K_true;
    json_data.metadata.K_estimated = Results.K_estimated;
    json_data.metadata.SNR_dB = Results.SNR_dB;
    json_data.metadata.duration_s = Results.duration_s;
    json_data.metadata.sfreq = Sim.sfreq;
    json_data.metadata.n_channels = C;
    json_data.metadata.n_samples_analyzed = Results.n_maps;
    json_data.metadata.runtime_s = Results.runtime;
    
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
    json_data.channel_info.labels = ch_labels;
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
        
        json_data.electrode_positions.(ch_labels{c}) = struct(...
            'x', double(pos_x), ...
            'y', double(pos_y), ...
            'z', double(pos_z));
    end
    
    % ===== MATCHING INFORMATION =====
    if ~isempty(metrics.match_assignment)
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