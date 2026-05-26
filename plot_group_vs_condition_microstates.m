function [plot_file, PlotInfo] = plot_group_vs_condition_microstates(hierarchical_mat, varargin)
% PLOT_GROUP_VS_CONDITION_MICROSTATES
%
% Write only the group-vs-condition microstate figure.

    [files, PlotInfo] = plot_hier_ms( ...
        hierarchical_mat, ...
        varargin{:}, ...
        'include_global', false, ...
        'include_groups', false, ...
        'include_conditions', true);

    plot_file = '';
    if isstruct(files) && isfield(files, 'group_vs_condition')
        plot_file = files.group_vs_condition;
    end
end
