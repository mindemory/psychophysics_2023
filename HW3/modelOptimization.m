function [estP, nll, mod_evi, mod_fit] = modelOptimization(psyfuncGenerator, cohLevs, nR, ntrials, initP, lb, ub, modelType, sign_mu, attn_mu, sig_change)
options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display', 'off');
eps = 0; % A small value to avoid float-point errors
nLogL = @(NR, NT, p) -sum(NR.*log(psyfuncGenerator(cohLevs, [p(1) p(2)])+eps) + ...
    (NT-NR).*log(1-psyfuncGenerator(cohLevs, [p(1) p(2)])+eps));
res = 20;
computeEvidence = @(NLL, prior, c) sum(exp(-NLL+log(prior)-c)) * exp(c);
pr_lb = 0.1; pr_ub = 1;
% Negative log-likelihood
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

% Get model evidence using Bayesian model-fitting
switch modelType
    case 1
        param_space = linspace(pr_lb, pr_ub, res);
        numcombs = length(param_space);
        th_record = NaN(numcombs, 2);
        for ii = 1:numcombs
            this_param = param_space(ii);
            this_prior = 1./this_param;
            this_nll = mod_nLogNL(this_param);
            th_record(ii, :) = [this_nll, this_prior];
        end
    case 2
        p1 = linspace(pr_lb, pr_ub, res);
        p2 = linspace(pr_lb, pr_ub, res);
        [P1, P2] = ndgrid(p1, p2);
        param_space = [P1(:) P2(:)];
        numcombs = length(param_space);
        th_record = NaN(numcombs, 2);
        for ii = 1:numcombs
            this_param = param_space(ii, :);
            this_prior = 1/res * 1/this_param(2);
            this_nll = mod_nLogNL(this_param);
            th_record(ii, :) = [this_nll, this_prior];
        end
    case 3
        p1 = linspace(pr_lb, pr_ub, res);
        p2 = linspace(pr_lb, pr_ub, res);
        p3 = linspace(pr_lb, pr_ub, res);
        [P1, P2, P3] = ndgrid(p1, p2, p3);
        param_space = [P1(:) P2(:) P3(:)];
        numcombs = length(param_space);
        th_record = NaN(numcombs, 2);
        for ii = 1:numcombs
            this_param = param_space(ii, :);
            this_prior = 1/res^2 * 1/this_param(3);
            this_nll = mod_nLogNL(this_param);
            th_record(ii, :) = [this_nll, this_prior];
        end
    case 4
        p1 = linspace(pr_lb, pr_ub, res);
        p2 = linspace(pr_lb, pr_ub, res);
        p3 = linspace(pr_lb, pr_ub, res);
        p4 = linspace(pr_lb, pr_ub, res);
        [P1, P2, P3, P4] = ndgrid(p1, p2, p3, p4);
        param_space = [P1(:) P2(:) P3(:) P4(:)];
        numcombs = length(param_space);
        th_record = NaN(numcombs, 2);
        for ii = 1:numcombs
            this_param = param_space(ii, :);
            this_prior = 1/res^2 * 1/this_param(3) * 1/this_param(4);
            this_nll = mod_nLogNL(this_param);
            th_record(ii, :) = [this_nll, this_prior];
        end
end

th_record(:, 2) = th_record(:, 2) ./ sum(th_record(:, 2)); % Occam's razor
% th_record(:, 1) = th_record(:, 1) ./ min(th_record(:, 1));
c = max(-th_record(:, 1) + log(th_record(:, 2)));
mod_evi = computeEvidence(th_record(:, 1), th_record(:, 2), c);

end