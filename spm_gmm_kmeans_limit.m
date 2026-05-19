function result = spm_gmm_kmeans_limit(X, K, varargin)
% SPM_GMM_KMEANS_LIMIT: GMM in the K-means limit (isotropic, hard E-step, free energy)
%   X: N x D data matrix (whitened)
%   K: number of clusters
%   Optional: 'max_iter', 'tol', 'variance_floor', 'restarts'
% Returns: result struct with fields: means, labels, sigma2, free_energy, converged

p = inputParser;
p.addParameter('max_iter', 200, @isnumeric);
p.addParameter('tol', 1e-6, @isnumeric);
p.addParameter('variance_floor', 1e-3, @isnumeric);
p.addParameter('restarts', 20, @isnumeric);
p.parse(varargin{:});
max_iter = p.Results.max_iter;
tol = p.Results.tol;
variance_floor = p.Results.variance_floor;
R = p.Results.restarts;

[N, D] = size(X);
best_fe = -Inf;
best_result = struct();

for r = 1:R
    % K-means++ init
    means = X(randperm(N,1),:);
    for k = 2:K
        dists = min(pdist2(X,means),[],2);
        probs = dists.^2 / sum(dists.^2);
        cumprobs = cumsum(probs);
        idx = find(rand < cumprobs,1);
        means = [means; X(idx,:)];
    end
    sigma2 = var(X(:));
    sigma2 = max(sigma2, variance_floor);
    labels = zeros(N,1);
    converged = false;
    fe_hist = [];
    for iter = 1:max_iter
        % E-step: assign to closest mean (hard, argmax posterior)
        dists = pdist2(X, means, 'euclidean').^2;
        [~, labels] = min(dists, [], 2);
        % M-step: update means
        for k = 1:K
            if any(labels==k)
                means(k,:) = mean(X(labels==k,:),1);
            else
                means(k,:) = X(randi(N),:); % reinit empty cluster
            end
        end
        % Update variance (isotropic, with floor)
        sigma2 = mean(arrayfun(@(k) mean(sum((X(labels==k,:)-means(k,:)).^2,2)), 1:K));
        sigma2 = max(sigma2, variance_floor);
        % Free energy (SPM style, up to const)
        loglik = -0.5*D*log(2*pi*sigma2) - 0.5/sigma2 * sum(sum((X - means(labels,:)).^2));
        fe = loglik;
        fe_hist = [fe_hist; fe];
        % Check convergence
        if iter>1 && abs(fe_hist(end)-fe_hist(end-1))<tol
            converged = true;
            break;
        end
    end
    if fe > best_fe
        best_fe = fe;
        best_result = struct('means',means,'labels',labels,'sigma2',sigma2,'free_energy',fe,'converged',converged,'fe_hist',fe_hist);
    end
end
result = best_result;
end
