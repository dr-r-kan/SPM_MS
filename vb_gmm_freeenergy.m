function out = vb_gmm_freeenergy(X, Kmax, opts)
% vb_gmm_freeenergy  Robust Variational Bayes GMM with annealing, ARD(α), and background.
% Columns are samples.
% out = vb_gmm_freeenergy(X, Kmax, opts)
%
% Options (superset of original):
%   .Kgrid, .max_iter, .tol, .restarts, .diag_cov, .verbose
%   .whiten           : true/false (default false); PCA-whiten X internally
%   .anneal_beta      : vector of β in (0,1], e.g. [0.3 0.6 0.85 1] (default [1])
%   .learn_alpha_vec  : learn a vector α0(k) (ARD on comps). Default true.
%   .alpha0_min       : lower bound for α0(k) (default 1e-4)
%   .alpha0_maxit     : Newton steps for α0 updates (default 30)
%   .tau_prune        : prune if Nk/T < tau_prune (default 0.005)
%   .split_once       : one targeted split pass (default true)
%   .merge_once       : one greedy merge pass (default true)
%   .merge_thresh     : Bhattacharyya threshold for candidate merges (default 0.35)
%   .use_background   : 't' | 'uniform' | 'none' (default 'none')
%   .bg_df            : df for t background (default 4)
%   .bg_weight        : prior effective weight of background (default 0.02)
%
% Outputs: same fields as before + richer info_per_K.
% Licence: MIT.

% ---------------------- housekeeping ----------------------
if nargin < 2 || isempty(Kmax), Kmax = 10; end
if nargin < 3, opts = struct; end


[Xorig, ~, ~] = normalise_shape(X);
X = Xorig; N = size(X,1); T = size(X,2);

Kgrid       = getf(opts,'Kgrid',1:Kmax);
max_iter    = getf(opts,'max_iter',500);
tol         = getf(opts,'tol',1e-6);
restarts    = getf(opts,'restarts',3);
diag_cov    = getf(opts,'diag_cov',false);
verbose     = getf(opts,'verbose',0);
whiten      = getf(opts,'whiten',false);
beta_sched  = getf(opts,'anneal_beta',1);
beta_sched  = beta_sched(:)'; beta_sched(beta_sched<=0)=[]; beta_sched = unique(min(beta_sched,1),'stable');

learn_alpha_vec = getf(opts,'learn_alpha_vec',true);
alpha0_min      = getf(opts,'alpha0_min',1e-4);
alpha0_maxit    = getf(opts,'alpha0_maxit',30);

tau_prune   = getf(opts,'tau_prune',0.005);
split_once  = getf(opts,'split_once',true);
merge_once  = getf(opts,'merge_once',true);
merge_thresh= getf(opts,'merge_thresh',0.35);

bg_mode     = getf(opts,'use_background','none');   % 't'|'uniform'|'none'
bg_df       = getf(opts,'bg_df',4);
bg_weight   = getf(opts,'bg_weight',0.02);
use_bg      = any(strcmpi(bg_mode,{'t','uniform'}));

% ---------------- empirical moments & whitening ----------------
xbar = mean(X,2);
Semp = cov_fallback(X);
Semp = regularise_cov(Semp, 1e-8);

if whiten
    [Uw,Sw,~] = svd(Semp,'econ');
    wD = diag(Sw); wD = max(wD, 1e-12);
    Wmat = diag(1./sqrt(wD)) * Uw';   % Xw = Wmat*(X - xbar)
    X = Wmat * (X - xbar);
    xbar_w = zeros(size(X,1),1);
    Semp_w = eye(size(X,1));
else
    Wmat = []; xbar_w = xbar; Semp_w = Semp;
end

N = size(X,1); T = size(X,2);

% ---------------- default NIW/Dirichlet priors ----------------
nu0   = N + 2;
beta0 = 1.0;
W0    = inv(Semp_w) * (nu0 - N - 1);      % => E[Σ]=Semp_w
W0    = symm(W0);

% ---------------- containers ----------------
Kmax_eff   = max(Kgrid);
F_best     = -inf*ones(1,Kmax_eff);
model_best = cell(1,Kmax_eff);
info_best  = cell(1,Kmax_eff);

% ---------------- K loop ----------------
for K = Kgrid
    Fk   = -inf; best = []; bestinfo = [];
    for r = 1:restarts
        % Seed by kmeans++ on whitened/centred X
        mu0 = kmeanspp_seed(X, K);
        [M,Fhist,infoK] = vb_em_core(X, mu0, K, beta_sched, ...
                                     beta0, W0, nu0, xbar_w, ...
                                     max_iter, tol, diag_cov, verbose>1, ...
                                     learn_alpha_vec, alpha0_min, alpha0_maxit, ...
                                     tau_prune, split_once, merge_once, merge_thresh, ...
                                     use_bg, bg_mode, bg_df, bg_weight);
        Fend = Fhist(end);
        if verbose
            fprintf('K=%d restart %d: F=%.6f (iters=%d, K_eff=%d)\n', K, r, Fend, numel(Fhist), infoK.K_eff);
        end
        if Fend > Fk
            Fk = Fend; best = M; bestinfo = infoK;
        end
    end

    % Unwhiten parameters if needed
    if whiten
        best = unwhiten_model(best, Wmat, xbar);
        bestinfo.whitened = true;
    else
        bestinfo.whitened = false;
    end

    F_best(K)     = Fk;
    model_best{K} = best;
    info_best{K}  = bestinfo;
end

% ---------- select K* by ELBO ----------
[~,ix] = max(F_best(Kgrid));
K_star = Kgrid(ix);
Mstar  = model_best{K_star};
R      = Mstar.R;
[~, labels] = max(R, [], 1);

% Pack outputs
out = struct();
out.K_star       = K_star;
out.F_per_K      = F_best(Kgrid);
out.labels       = labels;
out.R            = R;
out.model        = Mstar;
out.models_per_K = model_best(Kgrid);
out.info_per_K   = info_best(Kgrid);
out.opts_used    = struct('learn_alpha_vec',learn_alpha_vec,'tau_prune',tau_prune, ...
                          'split_once',split_once,'merge_once',merge_once, ...
                          'diag_cov',diag_cov,'whiten',whiten, ...
                          'anneal_beta',beta_sched,'background',bg_mode);
end

% =======================================================================
%                                CORE VB
% =======================================================================
function [M,Fhist,infoK] = vb_em_core(X, mu0, K, beta_sched, ...
                                      beta0, W0, nu0, m0, ...
                                      max_iter, tol, diag_cov, very_verbose, ...
                                      learn_alpha_vec, alpha0_min, alpha0_maxit, ...
                                      tau_prune, split_once, merge_once, merge_thresh, ...
                                      use_bg, bg_mode, bg_df, bg_weight)
[N,T] = size(X);

% ----- initialise responsibilities from means -----
R = seed_soft_resp(X, mu0);

% ----- priors -----
alpha0 = (1/max(K,1)) * ones(K,1);   % initial symmetric; will become vector via learning

% ----- initial posteriors -----
[alpha, m, beta, W, nu] = posterior_from_R(X,R,alpha0,m0,beta0,W0,nu0,diag_cov);

Fhist = -inf;
for b = 1:numel(beta_sched)
    beta_like = beta_sched(b);  % deterministic annealing β
    converged = false;

    for it = 1:max_iter
        % ---- E step (tempered) ----
        log_rho = expected_log_gauss_fast(X, m, beta, W, nu, diag_cov);   % K x T
        Elogpi  = psi(alpha) - psi(sum(alpha));
        log_rho = bsxfun(@plus, beta_like*log_rho, Elogpi);

        % ----- add background if requested -----
        if use_bg
            switch lower(bg_mode)
                case 't'
                    log_bg = log_student_t(X, m0, inv(W0)/(nu0 - N - 1), bg_df);
                case 'uniform'
                    log_bg = log_uniform_box(X);
                otherwise
                    log_bg = -inf(1,T);
            end
            log_rho = [log_rho; log_bg + log(max(bg_weight,1e-6))];
        end

        % normalise responsibilities
        log_rho = bsxfun(@minus, log_rho, max(log_rho,[],1));
        rho = exp(log_rho);
        rho = bsxfun(@rdivide, rho, sum(rho,1));

        % if background used, split back into R (KxT) and Rbg (1xT)
        if use_bg
            Rbg = rho(end,:); rho = rho(1:end-1,:);
        else
            Rbg = zeros(1,T);
        end
        R = rho;

        % ---- prune tiny components (by effective mass) ----
        Nk = sum(R,2);
        keep = Nk >= tau_prune*T;
        if any(~keep) && sum(keep)>=1
            R = R(keep,:);
            alpha = alpha(keep); m = m(keep); beta = beta(keep); W = W(keep); nu = nu(keep);
            % Normalise alpha0 to current K
            a0 = alpha0(:);
            if numel(a0) ~= size(R,1)
                a0 = repmat(mean(double(a0)), size(R,1), 1);
            else
                a0 = a0(keep);
            end
            alpha0 = a0;
            K = size(R,1);
        end

        % ---- M step ----
        [alpha, m, beta, W, nu] = posterior_from_R(X,R,alpha0,m0,beta0,W0,nu0,diag_cov);

        % ---- learn vector α0 (ARD) ----
        if learn_alpha_vec
            alpha0 = optimise_dirichlet_alpha_vec(alpha, alpha0, alpha0_min, alpha0_maxit);
            Nk = sum(R,2);
            alpha = alpha0 + Nk;  % refresh
        end

        % ---- ELBO (tempered) ----
        F = elbo_tempered(X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,beta_like,Rbg,use_bg);
        Fhist(end+1) = F; %#ok<AGROW>

        if very_verbose && mod(it,10)==0
            fprintf('  β=%.2f it %d: F=%.6f (K=%d)\n', beta_like, it, F, numel(alpha));
        end
        if it>1 && abs(Fhist(end)-Fhist(end-1)) < tol*max(1,abs(Fhist(end-1)))
            converged = true; break
        end
    end
    if ~converged, it = max_iter; end
end

% ---- one split pass (targeted) ----
if split_once && size(R,1)>=1
    [R2,alpha2,m2,beta2,W2,nu2,F2,improved] = try_one_split(X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,@elbo_plain);
    if improved
        R=R2; alpha=alpha2; m=m2; beta=beta2; W=W2; nu=nu2; Fhist(end+1)=F2;
    end
end

% ---- one merge pass (greedy best) ----
if merge_once && size(R,1) > 1
    [R2,alpha2,m2,beta2,W2,nu2,F2,improved] = try_one_merge(X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,merge_thresh,@elbo_plain);
    if improved
        R=R2; alpha=alpha2; m=m2; beta=beta2; W=W2; nu=nu2; Fhist(end+1)=F2;
    end
end

% ---- final info, ICL, etc. ----
ll = expected_log_gauss_fast(X, m, beta, W, nu, diag_cov);
Eloglik = sum(sum(R .* ll));
K_eff = size(R,1);
pK = K_eff*(N + N*(N+1)/2) + (K_eff-1);
BIC = -2*Eloglik + pK * log(T);
ICL = BIC + 2*sum(sum(R .* log(max(R,eps))));

Epi = alpha./sum(alpha);

M = struct('alpha',alpha,'m',{m},'beta',beta,'W',{W},'nu',nu,'Epi',Epi,'R',R,'Fhist',Fhist);
infoK = struct('F_end',Fhist(end), 'Eloglik',Eloglik, 'ICL',ICL, 'K_eff',K_eff, ...
               'alpha0_final', alpha0(:)', 'used_background', use_bg, 'bg_mode', bg_mode);

end

% =======================================================================
%                              SUBROUTINES
% =======================================================================

function [alpha, m, beta, W, nu] = posterior_from_R(X,R,alpha0,m0,beta0,W0,nu0,diag_cov)
[N,~] = size(X); K = size(R,1);
Nk = sum(R,2);
xk = (X * R') ./ max(Nk',eps);

alpha = alpha0 + Nk;
beta  = beta0  + Nk;
nu    = nu0    + Nk;

m = cell(1,K); W = cell(1,K);
iW0 = inv(W0);
for k=1:K
    mk = (beta0 * m0 + Nk(k) * xk(:,k)) / beta(k);
    m{k} = mk;
    DX = X - mk;
    Sk = (DX .* R(k,:)) * DX';
    dx = xk(:,k) - m0;
    C  = beta0 * Nk(k) / beta(k) * (dx*dx');
    S_post = iW0 + Sk + C;
    if diag_cov, S_post = diag(max(diag(S_post), 1e-12)); end
    W{k} = symm(inv_chol(S_post));
end
nu = max(nu, N + 1 + 1e-6); % guard
end

function loglik = expected_log_gauss_fast(X, m, beta, W, nu, diag_cov)
[N,T] = size(X); K = numel(m);
ElogLambda = zeros(K,1);
loglik = zeros(K,T);
for k=1:K
    if diag_cov
        Wk = diag(max(diag(W{k}), 1e-12));
        ElogLambda(k) = sum(psi(0.5*(nu(k) - (0:(N-1))'))) + N*log(2) + logdet_diag(Wk);
        DX = X - m{k};
        quad = N/beta(k) + nu(k)*sum((DX.^2).*diag(Wk),1);
    else
        Wk = symm(W{k});
        ElogLambda(k) = sum(psi(0.5*(nu(k) - (0:(N-1))'))) + N*log(2) + logdet_spd(Wk);
        DX = X - m{k};
        WX = Wk * DX;                       % N x T
        quad = N/beta(k) + nu(k)*sum(DX.*WX, 1);
    end
    loglik(k,:) = 0.5*(ElogLambda(k) - N*log(2*pi)) - 0.5*quad;
end
end

function F = elbo_tempered(X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,beta_like,Rbg,use_bg)
% Tempered ELBO; reduces to standard when beta_like=1 and no background.

% --- ensure alpha0 matches alpha in size/orientation ---
a  = alpha(:);                  % Kx1
a0 = alpha0(:);                 % ?x1
if numel(a0) ~= numel(a)
    % If a0 is wrong length (can happen after prune/split/merge),
    % fall back to a scalar equal to its mean and broadcast.
    a0 = repmat(mean(double(alpha0(:))), size(a));
end

ll = expected_log_gauss_fast(X, m, beta, W, nu, diag_cov);   % K x T
E_log_pX = beta_like * sum(sum(R .* ll));

Elogpi   = psi(a) - psi(sum(a));
E_log_pZ = sum(sum(bsxfun(@times, R, Elogpi)));
E_log_qZ = sum(sum(R .* log(max(R,eps))));

E_log_ppi = dirichlet_log_norm(a0) + sum((a0 - 1) .* (psi(a) - psi(sum(a))));
E_log_qpi = dirichlet_log_norm(a)  - ((sum(a)-numel(a))*psi(sum(a)) - sum((a-1).*psi(a)));

E_log_pML = niw_E_log_p_sum(m,beta,W,nu,m0,beta0,W0,nu0,diag_cov);
E_log_qML = 0;
N = size(X,1);
for k=1:numel(m)
    E_log_qML = E_log_qML + 0.5*N*log(beta(k)/(2*pi)) ...
              + 0.5*ElogLambda_single(W{k},nu(k),N,diag_cov) ...
              - 0.5*N + wishart_entropy(W{k},nu(k),N);
end

F = E_log_pX + E_log_pZ + E_log_ppi + E_log_pML - (E_log_qZ + E_log_qpi + E_log_qML);

% Background contributes only to the assignment entropy if used; we omit a constant base-measure term.
if use_bg
    F = F - sum(Rbg .* log(max(Rbg,eps)));
end
end

function F = elbo_plain(X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov)
F = elbo_tempered(X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,1,zeros(1,size(X,2)),false);
end


function [R2,alpha2,m2,beta2,W2,nu2,F2,improved] = try_one_split( ...
    X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,elbo_fun)

% Safe, ELBO-guarded single split with alpha0 normalisation.
K = size(R,1);
improved = false;
R2=R; alpha2=alpha; m2=m; beta2=beta; W2=W; nu2=nu; F2=-inf;

if K < 1
    return;
end

% Choose the "widest" component by posterior mean covariance trace
N = size(X,1);
trcov = zeros(1,K);
Nk = sum(R,2);
for k=1:K
    Sigk = inv(W{k})/max(nu(k) - N - 1, 1e-6);
    trcov(k) = trace(Sigk);
end
[~,ks] = max(trcov);

% Guard: skip if component has too little effective mass or too few points
if Nk(ks) < max(5, 0.01*size(X,2)) || nnz(R(ks,:) > 0) < 3
    return;
end

% Proposed split: duplicate component ks by bisecting its responsibilities
Rnew = R;
Rnew(ks,:) = 0.5 * R(ks,:);
Rnew = [Rnew; Rnew(ks,:)];

% Build α0 of the correct length (K+1)
a0 = alpha0(:);
if isempty(a0), a0 = 1; end
if numel(a0) ~= K
    a0 = repmat(mean(double(a0)), K, 1);
end
a0_new = [a0; a0(ks)];

% Update posteriors and test ELBO
[alpha_n, m_n, beta_n, W_n, nu_n] = posterior_from_R(X,Rnew,a0_new,m0,beta0,W0,nu0,diag_cov);

Fcur = elbo_fun(X,R,alpha,m,beta,W,nu,a0,m0,beta0,W0,nu0,diag_cov);
Fnew = elbo_fun(X,Rnew,alpha_n,m_n,beta_n,W_n,nu_n,a0_new,m0,beta0,W0,nu0,diag_cov);

if Fnew > Fcur
    improved = true;
    R2=Rnew; alpha2=alpha_n; m2=m_n; beta2=beta_n; W2=W_n; nu2=nu_n; F2=Fnew;
end
end



function [R2,alpha2,m2,beta2,W2,nu2,F2,improved] = try_one_merge( ...
    X,R,alpha,m,beta,W,nu,alpha0,m0,beta0,W0,nu0,diag_cov,merge_thresh,elbo_fun)

K = size(R,1);
improved=false; R2=R; alpha2=alpha; m2=m; beta2=beta; W2=W; nu2=nu; F2=-inf;
if K<2, return; end

% Normalise α0 to current K
a0 = alpha0(:);
if isempty(a0), a0 = 1; end
if numel(a0) ~= K
    a0 = repmat(mean(double(a0)), K, 1);
end

[muHat, SigHat] = posterior_point_estimates_local(m,beta,W,nu);
best_gain = 0; best = [];
Fcur = elbo_fun(X,R,alpha,m,beta,W,nu,a0,m0,beta0,W0,nu0,diag_cov);

for i=1:K-1
    for j=i+1:K
        d = bhattacharyya_gauss(muHat{i}, SigHat{i}, muHat{j}, SigHat{j});
        if d < merge_thresh
            Rm = R; Rm(i,:) = R(i,:)+R(j,:); Rm(j,:) = [];

            % Merge α0 consistently
            a0m = a0;
            a0m(i) = a0m(i) + a0m(j);
            a0m(j) = [];
            a0m = a0m(:);

            [alpha_m, m_m, beta_m, W_m, nu_m] = posterior_from_R(X,Rm,a0m,m0,beta0,W0,nu0,diag_cov);
            Fm = elbo_fun(X,Rm,alpha_m,m_m,beta_m,W_m,nu_m,a0m,m0,beta0,W0,nu0,diag_cov);
            if Fm - Fcur > best_gain
                best_gain = Fm - Fcur;
                best = {Rm,alpha_m,m_m,beta_m,W_m,nu_m,Fm,a0m};
            end
        end
    end
end

if ~isempty(best)
    R2=best{1}; alpha2=best{2}; m2=best{3}; beta2=best{4}; W2=best{5}; nu2=best{6}; F2=best{7};
    improved=true;
end
end


% ---------- background densities ----------
function logp = log_student_t(X, mu, Sigma, df)
N = size(X,1); T = size(X,2);
Sigma = symm(Sigma);
U = chol(Sigma + 1e-12*eye(N),'upper');
logdetS = 2*sum(log(diag(U)));
Q = U \ (X - mu);
Q = sum(Q.^2,1);
logZ = gammaln(0.5*(df+N)) - gammaln(0.5*df) - 0.5*N*log(df*pi) - 0.5*logdetS;
logp = logZ - 0.5*(df+N)*log(1 + Q/df);
logp = reshape(logp, 1, T);
end

function logp = log_uniform_box(X)
xmin = min(X,[],2); xmax = max(X,[],2);
vol  = prod(max(xmax - xmin, 1e-12));
logp = -log(vol) * ones(1,size(X,2));
end

% ---------- utilities ----------
function M = unwhiten_model(Mw, Wmat, xbar)
% Mw: model in whitened space; return in original space
if isempty(Wmat), Mout = Mw; M = Mout; return; end
A = pinv(Wmat);   % original = A * whitened + xbar
K = numel(Mw.m);
m = cell(1,K); W = cell(1,K);
for k=1:K
    m{k} = A * Mw.m{k} + xbar;
    W{k} = (Wmat') * Mw.W{k} * Wmat;
end
M = Mw; M.m = m; M.W = W;
end

function R = seed_soft_resp(X, mu0)
K = size(mu0,2); T = size(X,2);
R = zeros(K,T);
for k=1:K
    Dk = sum((X - mu0(:,k)).^2,1);
    R(k,:) = -0.5*Dk;
end
R = exp(bsxfun(@minus, R, max(R,[],1)));
R = bsxfun(@rdivide, R, sum(R,1));
end

function [m1,m2] = two_means(Xc)
[~,C] = lloyd_kmeans(Xc, 2, 3, 50);
m1 = C(:,1); m2 = C(:,2);
end

function [idx_best, C_best] = lloyd_kmeans(X, K, replicates, maxIter)
[N,T] = size(X);
best_inertia = inf; idx_best = []; C_best = [];
for rep=1:replicates
    C = kmeanspp_seed(X, K);
    idx = zeros(1,T);
    for it=1:maxIter
        % assign
        for t=1:T
            x = X(:,t); dmin = inf; imin = 1;
            for k=1:K
                d = sum((x - C(:,k)).^2);
                if d<dmin, dmin=d; imin=k; end
            end
            idx(t) = imin;
        end
        % update
        C_new = zeros(N,K); counts = zeros(1,K);
        for t=1:T
            C_new(:,idx(t)) = C_new(:,idx(t)) + X(:,t);
            counts(idx(t))  = counts(idx(t)) + 1;
        end
        for k=1:K
            if counts(k)>0, C_new(:,k) = C_new(:,k)/counts(k);
            else, C_new(:,k) = X(:,randi(T)); end
        end
        if max(vecnorm(C_new - C,2,1)) < 1e-6, C = C_new; break; end
        C = C_new;
    end
    inertia = 0; for t=1:T, inertia = inertia + sum((X(:,t) - C(:,idx(t))).^2); end
    if inertia < best_inertia, best_inertia = inertia; idx_best = idx; C_best = C; end
end
end

function [mu, Sigma] = posterior_point_estimates_local(m,beta,W,nu)
K = numel(m); N = numel(m{1});
mu = cell(1,K); Sigma = cell(1,K);
for k=1:K
    mu{k} = m{k};
    Sigma{k} = inv(W{k}) / max(nu(k) - N - 1, 1e-6);
    Sigma{k} = symm(Sigma{k});
end
end

function alpha0 = optimise_dirichlet_alpha_vec(alpha_post, alpha0, alpha0_min, maxit)
% Newton-like updates on α0(k) with positivity and damping.
alpha0 = max(alpha0(:), alpha0_min);
for it=1:maxit
    S  = sum(alpha_post);
    g  = psi(S) - psi(sum(alpha0)) + psi(alpha_post) - psi(alpha0);
    H  = -psi(1,alpha0) - psi(1,sum(alpha0));   % diagonal approx
    step = g ./ max(-H, 1e-6);
    alpha0_new = max(alpha0 + step, alpha0_min);
    if max(abs(alpha0_new - alpha0)) < 1e-6*max(1,max(alpha0)), alpha0 = alpha0_new; break; end
    alpha0 = 0.7*alpha0 + 0.3*alpha0_new;     % damping
end
end

function [Xn, N, T] = normalise_shape(X)
if size(X,2) >= size(X,1)
    Xn = X; N = size(X,1); T = size(X,2);
else
    Xn = X'; N = size(Xn,1); T = size(Xn,2);
end
end

function S = cov_fallback(X)
S = cov(X');
if ~isfinite(sum(S(:))) || any(isnan(S(:)))
    Xc = X - mean(X,2);
    S = (Xc*Xc')/max(size(X,2)-1,1);
end
S = symm(S);
end

function S = regularise_cov(S, epsv)
S = symm(S);
e = eig(S);
emin = min(e);
if emin < epsv
    S = S + (epsv - emin) * eye(size(S));
end
S = S + 1e-12*eye(size(S));
S = symm(S);
end

function A = symm(A), A = (A + A')/2; end
function L = logdet_spd(A), U = chol(A + 1e-12*eye(size(A)),'upper'); L = 2*sum(log(diag(U))); end
function L = logdet_diag(A), L = sum(log(max(diag(A), realmin))); end
function C = kmeanspp_seed(X,K)
[N,T] = size(X);
C = zeros(N,K);
idx = randi(T); C(:,1) = X(:,idx);
D = sum((X - C(:,1)).^2,1);
for k=2:K
    p = D./sum(D); p = max(p,realmin); cdf = cumsum(p);
    r = rand; j = find(cdf>=r,1,'first');
    C(:,k) = X(:,j);
    Dj = sum((X - C(:,k)).^2,1);
    D = min(D, Dj);
end
end
function Ainv = inv_chol(A)
U = chol(A + 1e-12*eye(size(A)),'upper');
Ainv = U \ (U' \ eye(size(A)));
end
function L = dirichlet_log_norm(a)
L = gammaln(sum(a)) - sum(gammaln(a));
end
function H = wishart_entropy(W,nu,N)
H = -wishart_logZ(W,nu,N) - 0.5*(nu - N - 1)*ElogLambda_single(W,nu,N,false) + 0.5*nu*N;
end
function Z = wishart_logZ(W,nu,N)
Z = -(nu/2)*logdet_spd(W) - (nu*N/2)*log(2) - mvgammaln(0.5*nu,N);
end

% ---------- RESTORED: NIW prior expected log-probability sum ----------
function E = niw_E_log_p_sum(m,beta,W,nu,m0,beta0,W0,nu0,diag_cov)
K = numel(m); N = numel(m{1}); E = 0; iW0 = inv(W0);
for k=1:K
    if diag_cov
        Wk = diag(diag(W{k}));
        ElogLam = ElogLambda_single(Wk,nu(k),N,true);
        tr_term = trace(iW0 * (nu(k)*Wk));
    else
        Wk = W{k};
        ElogLam = ElogLambda_single(Wk,nu(k),N,false);
        tr_term = trace(iW0 * (nu(k)*Wk));
    end
    term_L = wishart_logZ(W0,nu0,N) + 0.5*(nu0 - N - 1)*ElogLam - 0.5*tr_term;
    dmu = m{k} - m0;
    quad_mu = beta0 * ( nu(k) * (dmu'*(Wk*dmu)) + N/beta(k) );
    term_mu = 0.5*N*log(beta0/(2*pi)) + 0.5*ElogLam - 0.5*quad_mu;
    E = E + term_L + term_mu;
end
end

function e = ElogLambda_single(W,nu,N,diag_cov)
if diag_cov
    e = sum(psi(0.5*(nu - (0:(N-1))'))) + N*log(2) + logdet_diag(W);
else
    e = sum(psi(0.5*(nu - (0:(N-1))'))) + N*log(2) + logdet_spd(W);
end
end
function val = mvgammaln(a,p)
val = (p*(p-1)/4)*log(pi) + sum(gammaln(a + (1 - (1:p))/2));
end
function d = bhattacharyya_gauss(m1,S1,m2,S2)
S = 0.5*(S1+S2); dm = m2 - m1;
d = 0.125*(dm'*(S\dm)) + 0.5*log(max(det(S),realmin)/sqrt(max(det(S1),realmin)*max(det(S2),realmin)));
end
function v = getf(s,field,default)
if isstruct(s) && isfield(s,field) && ~isempty(s.(field)), v = s.(field); else, v = default; end
end
