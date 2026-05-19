function [features, whitening_params] = preprocess_microstate_maps(maps)
% Unified preprocessing: zero mean, whitening (unit covariance)
% Input: maps (N_samples x N_channels)
% Output: features (N_samples x N_channels), whitening_params (struct)

if isempty(maps)
    features = maps;
    whitening_params = struct();
    return;
end

% Zero mean (across samples)
mu = mean(maps, 1);
X_centered = bsxfun(@minus, maps, mu);

% Whitening (unit covariance)
C = cov(X_centered);
[V, D] = eig(C);
D = diag(D);
tol = 1e-10;
D(D < tol) = tol; % Avoid division by zero
W = V * diag(1 ./ sqrt(D)) * V';
features = X_centered * W;

whitening_params.mu = mu;
whitening_params.W = W;
end
