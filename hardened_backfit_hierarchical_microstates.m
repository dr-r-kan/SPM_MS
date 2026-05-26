function Results = hardened_backfit_hierarchical_microstates(hierarchical_results_mat, varargin)
% HARDENED_BACKFIT_HIERARCHICAL_MICROSTATES
%
% Compatibility wrapper that forwards to ms_backfit.

    Results = ms_backfit(hierarchical_results_mat, varargin{:});
end