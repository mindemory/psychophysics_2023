function modelEvidence(psyfuncGenerator, cohLevs, modelType)
eps = 1e-6; % A small value to avoid float-point errors
nLogL = @(NR, NT, p) -sum(NR.*log(psyfuncGenerator(cohLevs, [p(1) p(2)])+eps) + ...
                         (NT-NR).*log(1-psyfuncGenerator(cohLevs, [p(1) p(2)])+eps));
computeEvidence = @(NLL, prior, c) sum(exp(NLL+prior-c) .* exp(c));


switch modelType
    case 1
        prior = 
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [0 p]), 1:4));
    case 2
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx) p(2)]), 1:4));
    case 3
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)]), 1:4));
    case 4
        mod_nLogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)+p(4)*sig_change(idx)]), 1:4));
end

end