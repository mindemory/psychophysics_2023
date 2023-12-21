function [estP, nll, mod_fit] = modelOptimization(psyfuncGenerator, cohLevs, nR, ntrials, initP, lb, ub, modelType, sign_mu, attn_mu, sig_change)
options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display', 'off');
eps = 1e-6; % A small value to avoid float-point errors
nLogL = @(NR, NT, p) -sum(NR.*log(psyfuncGenerator(cohLevs, [p(1) p(2)])+eps) + ...
                         (NT-NR).*log(1-psyfuncGenerator(cohLevs, [p(1) p(2)])+eps));
p_range = linspace(lb, ub, res); % model parameter ranges
computeEvidence = @(NLL, prior, c) sum(exp(NLL+prior-c) .* exp(c));

switch modelType
    case 1
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [0 p]), 1:4));
    case 2
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx) p(2)]), 1:4));
    case 3
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)]), 1:4));
    case 4
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)+p(4)*sig_change(idx)]), 1:4));
end

% Get model evidence using Bayesian model-fitting
p_range = linspace(lb, ub, res);
switch modelType
    case 1
        prior_sig = 1./p_range;
    case 2
        prior_mu = 1./ones(1, length(p_range));
        prior_sig = 1./p_range;
    case 3
        prior_mu1 = 1./ones(1, length(p_range));
        prior_mu2 = 1./ones(1, length(p_range));
        prior_sig = 1./p_range;
    case 4
        prior_mu1 = 1./ones(1, length(p_range));
        prior_mu2 = 1./ones(1, length(p_range));
        prior_sig1 = 1./p_range;
        prior_sig2 = 1./p_range;
end

% Optimization using fmincon
[estP, nll] = fmincon(mod_nLogNL, initP, [], [], [], [], lb, ub, [], options);
mod_fit = NaN(4, length(cohLevs));
for idx = 1:4
    switch modelType
        case 1
            mod_fit(idx, :) = psyfuncGenerator(cohLevs, [0 estP]);
        case 2
            mod_fit(idx, :) = psyfuncGenerator(cohLevs, [estP(1)*sign_mu(idx) estP(2)]);
        case 3
            mod_fit(idx, :) = psyfuncGenerator(cohLevs, [estP(1)*sign_mu(idx)+estP(2)*attn_mu(idx) estP(3)]);
        case 4
            mod_fit(idx, :) = psyfuncGenerator(cohLevs, [estP(1)*sign_mu(idx)+estP(2)*attn_mu(idx), estP(3)+estP(4)*sig_change(idx)]);
    end
end
end