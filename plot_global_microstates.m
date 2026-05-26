function [plot_file, PlotInfo] = plot_global_microstates(hierarchical_mat, varargin)
% PLOT_GLOBAL_MICROSTATES
%
% Write only the single-row global microstate figure.

    [files, PlotInfo] = plot_hier_ms( ...
        hierarchical_mat, ...
        varargin{:}, ...
        'include_global', true, ...
        'include_groups', false, ...
        'include_conditions', false);

    plot_file = '';
    if isstruct(files) && isfield(files, 'global')
        plot_file = files.global;
    end
end
