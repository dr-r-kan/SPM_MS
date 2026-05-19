function [template_maps, template_labels, channel_labels, chanlocs] = load_metamaps_templates(template_file, varargin)
% LOAD_METAMAPS_TEMPLATES Load MetaMaps microstate templates from an EEGLAB .set file.
%
% Returns maps as K x channels, zero-mean/unit-norm per map.  The
% MetaMaps_2023_06.set file stores K=4,5,6,7 solutions concatenated as
% 4+5+6+7=22 maps; for K=7 we use the final seven maps and their documented
% order D,A,C,F,B,G,E.

    p = inputParser;
    addRequired(p, 'template_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K', 7, @isnumeric);
    parse(p, template_file, varargin{:});
    K = p.Results.K;

    if ~exist(char(template_file), 'file')
        error('Template file not found: %s', char(template_file));
    end
    if ~exist('pop_loadset', 'file')
        error('EEGLAB pop_loadset not found on the MATLAB path.');
    end

    EEG = pop_loadset('filename', char(template_file));
    data = double(squeeze(EEG.data));
    if ndims(data) ~= 2
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
    if K == 7 && n_maps_total >= 22
        idx = (n_maps_total - 6):n_maps_total;
        template_labels = {'D', 'A', 'C', 'F', 'B', 'G', 'E'};
    elseif K <= n_maps_total
        idx = (n_maps_total - K + 1):n_maps_total;
        template_labels = arrayfun(@(i) char('A' + i - 1), 1:K, 'UniformOutput', false);
    else
        error('Requested K=%d templates, but only %d maps were found.', K, n_maps_total);
    end

    template_maps = all_maps(idx, :);
    util = microstate_utilities_SHARED();
    template_maps = util.normalize_maps(template_maps);

    chanlocs = [];
    channel_labels = {};
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        chanlocs = EEG.chanlocs;
        channel_labels = cell(1, numel(chanlocs));
        for c = 1:numel(chanlocs)
            if isfield(chanlocs(c), 'labels') && ~isempty(chanlocs(c).labels)
                channel_labels{c} = char(chanlocs(c).labels);
            else
                channel_labels{c} = sprintf('Ch%03d', c);
            end
        end
    else
        channel_labels = arrayfun(@(c) sprintf('Ch%03d', c), 1:size(template_maps, 2), 'UniformOutput', false);
    end
end
