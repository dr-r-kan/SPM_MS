function validate_fixes()
% VALIDATE_FIXES: Check that all critical bugs are fixed
%
% Run this BEFORE running the pipeline to verify fixes

    fprintf('\n========================================\n');
    fprintf('VALIDATING FIXES\n');
    fprintf('========================================\n\n');
    
    all_ok = true;
    
    % ===== CHECK 1: fit_microstate_kmeans_standard.m =====
    fprintf('1. fit_microstate_kmeans_standard.m:\n');
    try
        % Test that it returns all required fields
        Sim = generate_test_sim(5, 2.5, 60);
        Results = fit_microstate_kmeans_standard(Sim, 2:6, 'silhouette');
        
        required_fields = {'gev_vals', 'silhouette_vals', 'within_ss', 'free_energy_vals'};
        missing = {};
        for i = 1:length(required_fields)
            if ~isfield(Results, required_fields{i})
                missing{end+1} = required_fields{i};
            end
        end
        
        if isempty(missing)
            fprintf('   ✓ All required fields present\n');
        else
            fprintf('   ✗ MISSING FIELDS: %s\n', strjoin(missing, ', '));
            all_ok = false;
        end
    catch ME
        fprintf('   ✗ ERROR: %s\n', ME.message);
        all_ok = false;
    end
    
    % ===== CHECK 2: fit_microstate_vb_kmeans.m =====
    fprintf('\n2. fit_microstate_vb_kmeans.m:\n');
    try
        Sim = generate_test_sim(5, 2.5, 60);
        Results = fit_microstate_vb_kmeans(Sim, 2:6, 'free_energy');
        
        required_fields = {'free_energy_vals', 'silhouette_vals', 'within_ss', 'gev_vals'};
        missing = {};
        for i = 1:length(required_fields)
            if ~isfield(Results, required_fields{i})
                missing{end+1} = required_fields{i};
            end
        end
        
        if isempty(missing)
            fprintf('   ✓ All required fields present\n');
        else
            fprintf('   ✗ MISSING FIELDS: %s\n', strjoin(missing, ', '));
            all_ok = false;
        end
    catch ME
        fprintf('   ✗ ERROR: %s\n', ME.message);
        all_ok = false;
    end
    
    % ===== CHECK 3: fit_microstate_spm_vb.m =====
    fprintf('\n3. fit_microstate_spm_vb.m:\n');
    try
        if ~exist('spm_mix', 'file')
            fprintf('   ⊘ SPM not available (skipping)\n');
        else
            Sim = generate_test_sim(5, 2.5, 60);
            Results = fit_microstate_spm_vb(Sim, 2:6, 'elbow_sil_combined');
            
            required_fields = {'free_energy', 'silhouette_vals', 'free_energy_vals', 'gev_vals', 'within_ss'};
            missing = {};
            for i = 1:length(required_fields)
                if ~isfield(Results, required_fields{i})
                    missing{end+1} = required_fields{i};
                end
            end
            
            if isempty(missing)
                fprintf('   ✓ All required fields present\n');
            else
                fprintf('   ✗ MISSING FIELDS: %s\n', strjoin(missing, ', '));
                all_ok = false;
            end
        end
    catch ME
        fprintf('   ✗ ERROR: %s\n', ME.message);
        all_ok = false;
    end
    
    % ===== CHECK 4: fit_microstate_dp_mixture.m =====
    fprintf('\n4. fit_microstate_dp_mixture.m:\n');
    try
        Sim = generate_test_sim(5, 2.5, 60);
        Results = fit_microstate_dp_mixture(Sim, 2:6, 'free_energy');
        
        required_fields = {'free_energy_vals', 'silhouette_vals', 'within_ss', 'gev_vals'};
        missing = {};
        for i = 1:length(required_fields)
            if ~isfield(Results, required_fields{i})
                missing{end+1} = required_fields{i};
            end
        end
        
        if isempty(missing)
            fprintf('   ✓ All required fields present\n');
        else
            fprintf('   ✗ MISSING FIELDS: %s\n', strjoin(missing, ', '));
            all_ok = false;
        end
    catch ME
        fprintf('   ✗ ERROR: %s\n', ME.message);
        all_ok = false;
    end
    
    % ===== CHECK 5: select_K_by_criterion =====
    fprintf('\n5. select_K_by_criterion function:\n');
    try
        Sim = generate_test_sim(5, 2.5, 60);
        Results = fit_microstate_kmeans_standard(Sim, 2:6, 'silhouette');
        
        criteria = {'silhouette', 'gev', 'elbow', 'free_energy'};
        for i = 1:length(criteria)
            K_sel = select_K_by_criterion(Results, criteria{i});
            if isnan(K_sel)
                fprintf('   ⊘ Criterion "%s": Not available (expected)\n', criteria{i});
            else
                fprintf('   ✓ Criterion "%s": K=%d\n', criteria{i}, K_sel);
            end
        end
    catch ME
        fprintf('   ✗ ERROR: %s\n', ME.message);
        all_ok = false;
    end
    
    % ===== SUMMARY =====
    fprintf('\n========================================\n');
    if all_ok
        fprintf('✓ ALL CHECKS PASSED - Ready to run pipeline!\n');
    else
        fprintf('✗ SOME CHECKS FAILED - Fix issues before running\n');
    end
    fprintf('========================================\n\n');
end

% ===== HELPER =====

function Sim = generate_test_sim(K, SNR, dur)
    % Generate minimal test simulation
    Sim = generate_microstate_eeg(K, SNR, dur, 250, 12345);
end

function K_selected = select_K_by_criterion(Results, criterion)
    % Local copy of selection function
    switch criterion
        case 'silhouette'
            if isfield(Results, 'silhouette_vals') && ~isempty(Results.silhouette_vals)
                sil = Results.silhouette_vals;
                if length(sil) > 4
                    [~, idx] = max(sil(2:(end-1)));
                    idx = idx + 1;
                else
                    [~, idx] = max(sil);
                end
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
        case 'gev'
            if isfield(Results, 'gev_vals') && ~isempty(Results.gev_vals)
                [~, idx] = max(Results.gev_vals);
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
        case 'elbow'
            if isfield(Results, 'within_ss') && ~isempty(Results.within_ss)
                [~, idx] = min(Results.within_ss);
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
        case 'free_energy'
            if isfield(Results, 'free_energy_vals') && ~isempty(Results.free_energy_vals)
                [~, idx] = max(Results.free_energy_vals);
                K_selected = Results.K_candidates(idx);
            else
                K_selected = NaN;
            end
        otherwise
            K_selected = NaN;
    end
end