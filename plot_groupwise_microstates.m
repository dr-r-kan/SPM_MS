function [plot_file, PlotInfo] = plot_groupwise_microstates(hierarchical_mat, varargin)
% PLOT_GROUPWISE_MICROSTATES
%
% Write only the groupwise microstate figure.

    [files, PlotInfo] = plot_hier_ms( ...
        hierarchical_mat, ...
        varargin{:}, ...
        'include_global', false, ...
        'include_groups', true, ...
        'include_conditions', false);

    plot_file = '';
    if isstruct(files) && isfield(files, 'groupwise')
        plot_file = files.groupwise;
    end
end
