function T = diagnose_chanlocs_for_topoplot(eeg_file)
% DIAGNOSE_CHANLOCS_FOR_TOPOPLOT Summarize channel locations used for scalp plots.

    if nargin < 1
        eeg_file = 'E:\EEGs\TEST_EEG\sub_001_j_raw.set';
    end
    EEG = pop_loadset('filename', eeg_file);
    labels = cell(EEG.nbchan, 1);
    theta = nan(EEG.nbchan, 1);
    radius = nan(EEG.nbchan, 1);
    X = nan(EEG.nbchan, 1);
    Y = nan(EEG.nbchan, 1);
    Z = nan(EEG.nbchan, 1);
    for i = 1:EEG.nbchan
        labels{i} = char(EEG.chanlocs(i).labels);
        if isfield(EEG.chanlocs, 'theta') && ~isempty(EEG.chanlocs(i).theta), theta(i) = EEG.chanlocs(i).theta; end
        if isfield(EEG.chanlocs, 'radius') && ~isempty(EEG.chanlocs(i).radius), radius(i) = EEG.chanlocs(i).radius; end
        if isfield(EEG.chanlocs, 'X') && ~isempty(EEG.chanlocs(i).X), X(i) = EEG.chanlocs(i).X; end
        if isfield(EEG.chanlocs, 'Y') && ~isempty(EEG.chanlocs(i).Y), Y(i) = EEG.chanlocs(i).Y; end
        if isfield(EEG.chanlocs, 'Z') && ~isempty(EEG.chanlocs(i).Z), Z(i) = EEG.chanlocs(i).Z; end
    end
    T = table((1:EEG.nbchan)', labels, theta, radius, X, Y, Z, ...
        'VariableNames', {'idx', 'label', 'theta', 'radius', 'X', 'Y', 'Z'});
    fprintf('Channels: %d\n', EEG.nbchan);
    fprintf('Finite radius: %d, radius==0: %d, radius<=0.5: %d\n', ...
        sum(isfinite(radius)), sum(radius == 0), sum(radius <= 0.5 & isfinite(radius)));
    disp(T(1:min(40, height(T)), :));
    zero_idx = find(isfinite(radius) & radius == 0);
    if ~isempty(zero_idx)
        fprintf('Radius zero labels: %s\n', strjoin(labels(zero_idx), ', '));
    end
end
