function create_artifact_template()
% CREATE_ARTIFACT_TEMPLATE: Generate artefact_template.set from raw EEG file
%
% This function loads a real EEG recording, matches it to a template montage, 
% interpolates missing channels, and saves it as EEGLAB format for use as 
% artifact injection in microstate simulations.
%
% The output file will have:
% - Same number of channels as template_set
% - Same channel locations as template_set
% - Data interpolated from raw file using spherical interpolation
%
% Requirements:
% - EEGLAB with interpolation support (eeg_interp)
% - No Python required

    fprintf('\n========================================\n');
    fprintf('Creating Artifact Template\n');
    fprintf('========================================\n\n');
    
    % Define paths
    raw_file = 'E:\EEGs\SPM_MS\isoelectric_eeg.set';
    template_set = 'E:\EEGs\SPM_MS\MetaMaps_2023_06.set';
    output_dir = 'E:\EEGs\SPM_MS';
    output_file = fullfile(output_dir, 'artefact_template.set');
    
    % Check files exist
    fprintf('1. Checking input files...\n');
    if ~isfile(raw_file)
        error('Raw EEG file not found: %s', raw_file);
    end
    fprintf('   ✓ Raw EEG: %s\n', raw_file);
    
    if ~isfile(template_set)
        error('Template SET file not found: %s', template_set);
    end
    fprintf('   ✓ Template SET: %s\n', template_set);
    
    % Create output directory if needed
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    
    % Load raw EEG file using EEGLAB
    fprintf('\n2. Loading raw EEG file with EEGLAB...\n');
    try
        EEG_raw = pop_loadset('filename', raw_file);
    catch ME
        error('Failed to load EEG file: %s\n%s', raw_file, ME.message);
    end
    
    if isempty(EEG_raw) || isempty(EEG_raw.data)
        error('Failed to load data from EEG file');
    end
    
    fprintf('   ✓ Loaded %d channels\n', EEG_raw.nbchan);
    fprintf('   ✓ Duration: %.1f seconds\n', EEG_raw.times(end));
    fprintf('   ✓ Sampling rate: %.0f Hz\n', EEG_raw.srate);
    
    % Load template SET to get channel locations
    fprintf('\n3. Loading template SET for montage...\n');
    EEG_template = pop_loadset('filename', template_set);
    fprintf('   ✓ Loaded template with %d channels\n', EEG_template.nbchan);
    
    if isempty(EEG_template.chanlocs)
        error('Template has no channel location information');
    end
    fprintf('   ✓ Template has %d channel locations\n', length(EEG_template.chanlocs));
    
    % Verify template has valid coordinates
    valid_locs = ~cellfun('isempty', {EEG_template.chanlocs.X});
    fprintf('   ✓ %d channels with valid 3D coordinates\n', sum(valid_locs));
    
    % Prepare raw EEG for interpolation
    fprintf('\n4. Preparing channels for interpolation...\n');
    
    % Get channel names
    template_names = {EEG_template.chanlocs.labels};
    raw_names = {EEG_raw.chanlocs.labels};
    
    fprintf('   Raw file channels: %d\n', length(raw_names));
    fprintf('   Template channels: %d\n', length(template_names));
    
    % Find common channels between raw and template
    [common_names, raw_idx, template_idx] = intersect(raw_names, template_names, 'stable');
    
    if isempty(common_names)
        % If no exact name matches, try case-insensitive matching
        fprintf('   ⚠ No exact name matches, trying case-insensitive...\n');
        [common_names, raw_idx, template_idx] = intersect_case_insensitive(raw_names, template_names);
        
        if isempty(common_names)
            error('No matching channel names between raw and template.\nRaw: %s\nTemplate: %s', ...
                sprintf('%s ', raw_names{:}), sprintf('%s ', template_names{:}));
        end
    end
    
    fprintf('   ✓ Found %d common channels\n', length(common_names));
    fprintf('     %s\n', sprintf('%s ', common_names{:}));
    
    % Now build the full structure with all template channels
    % but keeping data only for common channels
    fprintf('\n5. Building full montage structure...\n');
    
    % Start with the raw data
    EEG = EEG_raw;
    
    % Get the data for common channels from raw file
    data_common = EEG.data(raw_idx, :, :);
    
    % Create new data matrix with all template channels (zeros for missing)
    n_samples = EEG.pnts;
    n_trials = EEG.trials;
    data_full = zeros(EEG_template.nbchan, n_samples, n_trials);
    
    % Fill in the common channel data at the correct template positions
    data_full(template_idx, :, :) = data_common;
    
    % Update EEG structure with full data and template locations
    EEG.data = data_full;
    EEG.nbchan = EEG_template.nbchan;
    EEG.chanlocs = EEG_template.chanlocs;
    
    fprintf('   ✓ Created full data matrix: %d channels x %d samples x %d trials\n', ...
        EEG.nbchan, EEG.pnts, EEG.trials);
    fprintf('   ✓ Assigned %d channels with data\n', length(common_names));
    fprintf('   ✓ Ready to interpolate %d missing channels\n', EEG_template.nbchan - length(common_names));
    
    % Identify which channels are MISSING (zeros in data)
    missing_idx = setdiff(1:EEG_template.nbchan, template_idx);
    missing_names = template_names(missing_idx);
    
    fprintf('\n6. Performing spherical interpolation...\n');
    fprintf('   Missing channels: %s\n', sprintf('%s ', missing_names{:}));
    
    % Use EEGLAB's eeg_interp to interpolate missing channels
    % eeg_interp will use spherical interpolation to fill in the missing data
    try
        EEG = eeg_interp(EEG, missing_idx, 'spherical');
        fprintf('   ✓ Interpolation successful\n');
    catch ME
        error('EEGLAB interpolation failed: %s\n\nMake sure EEGLAB is on your MATLAB path.', ME.message);
    end
    
    % Verify final structure
    fprintf('\n7. Verifying final structure...\n');
    fprintf('   ✓ Final channels: %d\n', EEG.nbchan);
    
    % Check all channels have locations
    has_X = ~cellfun('isempty', {EEG.chanlocs.X});
    has_Y = ~cellfun('isempty', {EEG.chanlocs.Y});
    has_Z = ~cellfun('isempty', {EEG.chanlocs.Z});
    has_all_coords = has_X & has_Y & has_Z;
    
    fprintf('   ✓ Channels with X coordinate: %d\n', sum(has_X));
    fprintf('   ✓ Channels with Y coordinate: %d\n', sum(has_Y));
    fprintf('   ✓ Channels with Z coordinate: %d\n', sum(has_Z));
    fprintf('   ✓ Channels with all 3D coordinates: %d\n', sum(has_all_coords));
    
    % Check for missing data or bad coordinates
    missing_coords = sum(~has_all_coords);
    if missing_coords > 0
        fprintf('   ⚠ WARNING: %d channels lack complete 3D coordinate data\n', missing_coords);
        fprintf('   Channels without complete coords: %s\n', sprintf('%s ', EEG.chanlocs(~has_all_coords).labels));
    end
    
    % Check data statistics
    data_min = min(EEG.data(:));
    data_max = max(EEG.data(:));
    data_std = std(EEG.data(:));
    
    fprintf('   ✓ Data range: [%.2f, %.2f] µV\n', data_min, data_max);
    fprintf('   ✓ Data std dev: %.2f µV\n', data_std);
    
    % Prepare for export
    fprintf('\n8. Preparing for export...\n');
    EEG.setname = 'Artifact Template';
    fprintf('   ✓ Set name: %s\n', EEG.setname);
    
    % Save as SET file
    fprintf('\n9. Saving as EEGLAB SET...\n');
    try
        EEG = pop_saveset(EEG, 'filename', output_file);
        
        if isfile(output_file)
            file_size_mb = dir(output_file).bytes / 1024 / 1024;
            fprintf('   ✓ Successfully saved!\n');
            fprintf('   ✓ File: %s\n', output_file);
            fprintf('   ✓ Size: %.1f MB\n', file_size_mb);
            fprintf('   ✓ Channels: %d (with interpolated data)\n', EEG.nbchan);
            fprintf('   ✓ Duration: %.1f seconds\n', EEG.times(end));
        else
            error('File not created');
        end
        
    catch ME
        error('Failed to save SET file: %s', ME.message);
    end
    
    fprintf('\n========================================\n');
    fprintf('✓ Artifact Template Created Successfully!\n');
    fprintf('========================================\n');
    fprintf('\nTemplate ready for microstate EEG generation.\n');
    fprintf('All %d channels have locations and interpolated data.\n', EEG.nbchan);
    fprintf('File: %s\n\n', output_file);
end

function [common_names, raw_idx, template_idx] = intersect_case_insensitive(raw_names, template_names)
% INTERSECT_CASE_INSENSITIVE: Find matching channel names regardless of case
    
    common_names = {};
    raw_idx = [];
    template_idx = [];
    
    % Convert to lowercase for comparison
    raw_names_lower = cellfun(@lower, raw_names, 'UniformOutput', false);
    template_names_lower = cellfun(@lower, template_names, 'UniformOutput', false);
    
    % Find matches
    for i = 1:length(raw_names)
        idx = find(strcmp(raw_names_lower{i}, template_names_lower));
        if ~isempty(idx)
            common_names = [common_names, raw_names(i)];
            raw_idx = [raw_idx, i];
            template_idx = [template_idx, idx(1)];
        end
    end
end