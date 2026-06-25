% Plot canonical template microstates A-G from the MetaMaps template file.
%
% Diagnostic version: compares the current shared loader output against the
% labelled MetaMaps reference (raw geometry + 90 deg anticlockwise).
%
% Run this script from the repository root after EEGLAB is on the MATLAB path.

util = microstate_utilities();
cfg = util.load_config();

template_file = char(cfg.paths.template_file);
output_dir = fullfile(char(cfg.paths.single_plot_dir), 'template_microstates');
output_file = fullfile(output_dir, 'canonical_template_microstates_A_to_G_loader_check.png');
figure_visible = true;

if exist('pop_loadset', 'file') ~= 2
    error('EEGLAB pop_loadset is not on the MATLAB path.');
end
if exist('topoplot', 'file') ~= 2
    error('EEGLAB topoplot is not on the MATLAB path.');
end
if ~isfile(template_file)
    error('Template file not found: %s', template_file);
end

EEG = pop_loadset('filename', template_file);
[template_maps, template_labels] = load_template_maps_raw_local(EEG, 7);
[shared_maps, shared_labels, ~, shared_chanlocs] = load_metamaps_templates(template_file, 'K', 7);
expected_labels = arrayfun(@(k) char('A' + k - 1), 1:7, 'UniformOutput', false);
if ~isequal(cellstr(string(template_labels(:)))', expected_labels)
    error('Expected template labels A-G, got: %s', strjoin(cellstr(string(template_labels(:)))', ', '));
end
if ~isequal(cellstr(string(shared_labels(:)))', expected_labels)
    error('Shared loader did not return canonical labels A-G.');
end

if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
    error('Template file does not contain channel locations.');
end
if numel(EEG.chanlocs) < size(template_maps, 2)
    error('Template file has %d chanlocs but template maps have %d channels.', ...
        numel(EEG.chanlocs), size(template_maps, 2));
end

chanlocs_raw = EEG.chanlocs(1:size(template_maps, 2));
keep = has_usable_topoplot_location_local(chanlocs_raw);
if nnz(keep) < 4
    error('Fewer than four template channels have usable topoplot geometry.');
end
chanlocs_raw = chanlocs_raw(keep);
template_maps = template_maps(:, keep);
template_maps = util.normalize_maps(template_maps);
reference_chanlocs = rotate_chanlocs_for_display_local(chanlocs_raw, 90);

if size(shared_maps, 1) ~= size(template_maps, 1)
    error('Shared loader map count does not match raw reference.');
end
if numel(shared_chanlocs) < 4
    error('Shared loader returned fewer than four usable chanlocs.');
end

row_specs = { ...
    struct('label', 'Shared loader current', 'maps', shared_maps, 'chanlocs', shared_chanlocs), ...
    struct('label', 'Reference: labelled +90 anticlockwise', 'maps', template_maps, 'chanlocs', reference_chanlocs)};

clim = max(abs([shared_maps(:); template_maps(:)]));
if ~isfinite(clim) || clim <= eps
    clim = 1;
end

util.ensure_dir(output_dir);
fig = figure( ...
    'Name', 'Canonical template microstates A-G orientation check', ...
    'Color', 'white', ...
    'NumberTitle', 'off', ...
    'Visible', local_onoff(figure_visible), ...
    'Position', [100, 100, 1800, 860]);
colormap(fig, 'jet');

for r = 1:numel(row_specs)
    chanlocs_this = row_specs{r}.chanlocs;
    maps_this = row_specs{r}.maps;
    for k = 1:7
        ax = subplot(numel(row_specs), 7, (r - 1) * 7 + k, 'Parent', fig);
        topoplot(maps_this(k, :), chanlocs_this, ...
            'electrodes', 'off', ...
            'numcontour', 6, ...
            'maplimits', [-clim clim]);
        axis(ax, 'off');
        if r == 1
            title(ax, template_labels{k}, 'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');
        end
        if k == 1
            text(ax, -0.24, 0.5, row_specs{r}.label, ...
                'Units', 'normalized', ...
                'Rotation', 90, ...
                'HorizontalAlignment', 'center', ...
                'FontWeight', 'bold', ...
                'FontSize', 12, ...
                'Interpreter', 'none');
        end
    end
end

sgtitle(fig, sprintf('Canonical template microstates A-G | shared loader vs confirmed reference | %s', template_file), ...
    'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');
exportgraphics(fig, output_file, 'Resolution', double(cfg.plotting.resolution));

fprintf('Saved canonical template loader check: %s\n', output_file);

function [template_maps, template_labels] = load_template_maps_raw_local(EEG, K)
    if isfield(EEG, 'msinfo') && isfield(EEG.msinfo, 'MSMaps') && numel(EEG.msinfo.MSMaps) >= K && ...
            isfield(EEG.msinfo.MSMaps(K), 'Maps') && ~isempty(EEG.msinfo.MSMaps(K).Maps)
        rec = EEG.msinfo.MSMaps(K);
        template_maps = double(rec.Maps);
        if size(template_maps, 1) ~= K && size(template_maps, 2) == K
            template_maps = template_maps';
        end
        if isfield(rec, 'Labels') && numel(rec.Labels) >= K
            template_labels = cellstr(string(rec.Labels));
            template_labels = template_labels(1:K);
        else
            template_labels = arrayfun(@(i) char('A' + i - 1), 1:K, 'UniformOutput', false);
        end
        [template_labels, sort_idx] = sort_microstate_labels_local(template_labels);
        template_maps = template_maps(sort_idx, :);
        return;
    end

    data = double(squeeze(EEG.data));
    if ~ismatrix(data)
        error('Template data must be a 2-D channels x maps or maps x channels matrix.');
    end

    if isfield(EEG, 'nbchan') && size(data, 1) == EEG.nbchan
        all_maps = data';
    elseif isfield(EEG, 'nbchan') && size(data, 2) == EEG.nbchan
        all_maps = data;
    elseif size(data, 1) > size(data, 2)
        all_maps = data';
    else
        all_maps = data;
    end

    n_maps_total = size(all_maps, 1);
    if n_maps_total >= 22 && K == 7
        idx = 16:22;
        template_labels = {'D', 'A', 'C', 'F', 'B', 'G', 'E'};
    elseif K <= n_maps_total
        idx = (n_maps_total - K + 1):n_maps_total;
        template_labels = arrayfun(@(i) char('A' + i - 1), 1:K, 'UniformOutput', false);
    else
        error('Requested K=%d templates, but only %d maps were found.', K, n_maps_total);
    end

    template_maps = all_maps(idx, :);
    [template_labels, sort_idx] = sort_microstate_labels_local(template_labels);
    template_maps = template_maps(sort_idx, :);
end

function [labels_out, sort_idx] = sort_microstate_labels_local(labels_in)
    labels = cellstr(string(labels_in(:)));
    keys = lower(strtrim(labels));
    [~, sort_idx] = sort(keys);
    labels_out = labels(sort_idx);
end

function visible_str = local_onoff(flag)
    if flag
        visible_str = 'on';
    else
        visible_str = 'off';
    end
end

function valid = has_usable_topoplot_location_local(chanlocs)
    valid = false(1, numel(chanlocs));
    for i = 1:numel(chanlocs)
        has_polar = isfield(chanlocs(i), 'theta') && ~isempty(chanlocs(i).theta) && ...
            isfield(chanlocs(i), 'radius') && ~isempty(chanlocs(i).radius) && ...
            isfinite(double(chanlocs(i).theta)) && isfinite(double(chanlocs(i).radius)) && ...
            double(chanlocs(i).radius) > 0 && double(chanlocs(i).radius) <= 0.5;
        has_xyz = isfield(chanlocs(i), 'X') && ~isempty(chanlocs(i).X) && ...
            isfield(chanlocs(i), 'Y') && ~isempty(chanlocs(i).Y) && ...
            isfield(chanlocs(i), 'Z') && ~isempty(chanlocs(i).Z) && ...
            all(isfinite(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z]))) && ...
            norm(double([chanlocs(i).X chanlocs(i).Y chanlocs(i).Z])) > eps;
        valid(i) = has_polar || has_xyz;
    end
end

function chanlocs_out = rotate_chanlocs_for_display_local(chanlocs_in, angle_deg)
    chanlocs_out = chanlocs_in;
    if isempty(chanlocs_in) || ~isfinite(angle_deg) || abs(angle_deg) <= eps
        return;
    end

    rot = [cosd(angle_deg) -sind(angle_deg); sind(angle_deg) cosd(angle_deg)];
    for i = 1:numel(chanlocs_out)
        has_xy = isfield(chanlocs_out(i), 'X') && ~isempty(chanlocs_out(i).X) && ...
            isfield(chanlocs_out(i), 'Y') && ~isempty(chanlocs_out(i).Y) && ...
            all(isfinite(double([chanlocs_out(i).X chanlocs_out(i).Y])));
        if has_xy
            xy = rot * double([chanlocs_out(i).X; chanlocs_out(i).Y]);
            chanlocs_out(i).X = xy(1);
            chanlocs_out(i).Y = xy(2);
        end

        has_theta_radius = isfield(chanlocs_out(i), 'theta') && ~isempty(chanlocs_out(i).theta) && ...
            isfield(chanlocs_out(i), 'radius') && ~isempty(chanlocs_out(i).radius) && ...
            isfinite(double(chanlocs_out(i).theta)) && isfinite(double(chanlocs_out(i).radius));
        if has_theta_radius
            chanlocs_out(i).theta = wrap_display_angle_local(double(chanlocs_out(i).theta) + angle_deg);
        elseif has_xy
            chanlocs_out(i).theta = wrap_display_angle_local(atan2d(chanlocs_out(i).Y, chanlocs_out(i).X));
        end
    end
end

function angle_out = wrap_display_angle_local(angle_in)
    angle_out = mod(angle_in + 180, 360) - 180;
end
