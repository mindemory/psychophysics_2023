clear; close all; clc;

%% Initialization and Data Generation:
% Sets coherence levels, number of trials, subjects, and initializes the psychometric function generator.
% Generates simulated data for each model with varying Point of Subjective Equality (PSE) and slope values.
cohLevs = linspace(-1, 1, 20);
Ncoh = length(cohLevs);
ntrials = 1e2;
nsubs = 5;
gt_models = 4;
fit_models = 4;


psyfuncGenerator = @(x, p) (0.5*0)+(1-0).*normcdf(x, p(1), p(2));
PSEs = [0 0 0 0;
    -0.3 -0.3 0.3 0.3;
    -0.6 -0.3 0.3 0.6;
    -0.6 -0.3 0.3 0.6];

slp = 4;
slopes = [slp slp slp slp;
          slp slp slp slp;
          slp slp slp slp;
          slp*2.5 slp slp slp*2.5];

% Initializes variables for sign, attention, and slope change, needed for different model conditions.
sign_mu = [-1 -1 1 1];
attn_mu = [-1 0 0 1];
sig_change = [1 0 0 1];

data = NaN(gt_models, fit_models, Ncoh, nsubs, ntrials);
color_vect = ["r*", "b*", "k*", "m*"];
color_vectfits = ["r-", "b-", "k-", "m-"];

legnames = {'Adapt L: Attend yes', ...
            'Adapt L: Attend no', ...
            'Adapt R: Attend yes', ...
            'Adapt R: Attend no'};

figure();
for sub = 1:nsubs
    for modelN =  1:gt_models
        subplot(2, 2, modelN)
        hold on;
        for ttype = 1:fit_models
            this_PSE = PSEs(modelN, ttype);% + randn(1, 1)*PSEs(modelN, ttype)/10;
            this_slope = slopes(modelN, ttype);% + randn(1, 1)*slopes(modelN, ttype)/10;
            this_psyfunc = psyfuncGenerator(cohLevs, [this_PSE, 1/this_slope]);
            this_data = rand(Ncoh, ntrials) < repmat(this_psyfunc', 1, ntrials);
            data(modelN, ttype, :, sub, :) = this_data;
            plot(cohLevs, squeeze(mean(this_data, 2)), color_vect(ttype));
        end
        xlabel('Coherence Level')
        ylabel('Prob correct for Right')
        title(['Model: ' num2str(modelN)])
    end
end

%% Estimating optimal parameters using fmincon
% Shape is (subjects * model_groudtruth * model_estim)        
NLLs = NaN(nsubs, gt_models, fit_models); 
mfit_curves = NaN(nsubs, gt_models, fit_models, 4, length(cohLevs));

% Loops over subjects and models, fitting each model to the data and calculating the negative log-likelihood for each fit.
% For each model, uses fmincon to minimize the negative log-likelihood and find the best-fitting parameters.
lb = [0.01; 0.01; 0.01; 0.01];
ub = [1; 1; 1; 1];
numParams = [1, 2, 3, 4];

for sub = 1:nsubs
    for modelN = 1:gt_models
        nR = squeeze(sum(data(modelN, :, :, sub, :), 5));
        for modIdx = 1:fit_models
            initP = rand(numParams(modIdx), 1).*(ub(1:numParams(modIdx))-lb(1:numParams(modIdx))) + lb(1:numParams(modIdx));
            [estP, NLLs(sub, modelN, modIdx), mfit_curves(sub, modelN, modIdx, :, :)] = modelOptimization(psyfuncGenerator, cohLevs, nR, ntrials, initP, lb(1:numParams(modIdx)), ...
                ub(1:numParams(modIdx)), modIdx, sign_mu, attn_mu, sig_change);
        end
    end
end

figure();
for gtModelIdx = 1:gt_models
    [~, bestFitIdx] = min(NLLs(1, gtModelIdx, :));
    subplot(gt_models, fit_models + 1, (gtModelIdx - 1) * (fit_models + 1) + 1);
    hold on;
    for conditionIdx = 1:fit_models
        scatterData = squeeze(data(gtModelIdx, conditionIdx, :, 1, :));
        meanData = mean(scatterData, 2);
        scatter(cohLevs, meanData, color_vect(conditionIdx));
    end
    xlabel('Coherence Level');
    ylabel('Prob correct for Right');
    title(['GT Model: ' num2str(gtModelIdx)]);
    hold off;

    for fitModelIdx = 1:fit_models
        ax = subplot(gt_models, fit_models + 1, (gtModelIdx - 1) * (fit_models + 1) + fitModelIdx + 1);
        hold on;
        for conditionIdx = 1:fit_models
            scatterData = squeeze(data(gtModelIdx, conditionIdx, :, 1, :));
            meanData = mean(scatterData, 2);
            scatter(cohLevs, meanData, 'k*'); 
        end

        for conditionIdx = 1:fit_models
            fitCurve = squeeze(mfit_curves(1, gtModelIdx, fitModelIdx, conditionIdx, :));
            plot(cohLevs, fitCurve, color_vectfits(conditionIdx), 'LineWidth', 2);
        end

        xlabel('Coherence Level');
        ylabel('Prob correct for Right');
        nll_val = NLLs(1, gtModelIdx, fitModelIdx);
        title(['Fit Model: ' num2str(fitModelIdx) ', NLL: ' num2str(nll_val, '%.2f')]);

        if fitModelIdx == bestFitIdx
            set(ax, 'LineWidth', 2);
        end

        hold off;
    end
end
%% AIC, BIC, Bayes Factor Estimation
% Calculate AIC, BIC, and BF estimations
k = 1; n = 4 * ntrials;
AICc = 2*k + 2*NLLs;
BICc = k*log(n) + 2*NLLs;
BF_estim = NaN(size(BICc));

for ss = 1:nsubs
    for ii = 1:4
        this_min = min(BICc(ss, ii, :));
        BF_estim(ss, ii, :) = exp(-0.5.*(BICc(ss, ii, :)-this_min));
    end
end

% Calculate means and medians
NLL_mean = squeeze(mean(NLLs, 1));
AIC_mean = squeeze(mean(AICc, 1));
BIC_mean = squeeze(mean(BICc, 1));
BF_median = squeeze(median(BF_estim, 1));

% Plotting the log of NLL, AIC, BIC, and BF
figure();
for ii = 1:4
    subplot(4, 4, 4*(ii-1)+1);
    plotMetrics(NLLs, NLL_mean, 'NLL', ii);

    subplot(4, 4, 4*(ii-1)+2);
    plotMetrics(AICc, AIC_mean, 'AIC', ii);

    subplot(4, 4, 4*(ii-1)+3);
    plotMetrics(BICc, BIC_mean, 'BIC', ii);

    subplot(4, 4, 4*(ii-1)+4);
    plotMetrics(BF_estim, BF_median, 'BF estim', ii);
end


%% Doing model-fitting using Bayesian model evidence
% Shape is (subjects * model_groudtruth * model_estim)        
MEs = NaN(nsubs, gt_models, fit_models);
res = 1e2; % resolution with which to sample the parameter space

% Loops over subjects and models, fitting each model to the data and calculating the negative log-likelihood for each fit.
% For each model, uses fmincon to minimize the negative log-likelihood and find the best-fitting parameters.
lb = [0.01; 0.01; 0.01; 0.01];
ub = [1; 1; 1; 1];
numParams = [1, 2, 3, 4];

for sub = 1:nsubs
    for modelN = 1:gt_models
        nR = squeeze(sum(data(modelN, :, :, sub, :), 5));
        for modIdx = 1:fit_models
            initP = rand(numParams(modIdx), 1).*(ub(1:numParams(modIdx))-lb(1:numParams(modIdx))) + lb(1:numParams(modIdx));
            [estP, NLLs(sub, modelN, modIdx), mfit_curves(sub, modelN, modIdx, :, :)] = modelOptimization(psyfuncGenerator, cohLevs, nR, ntrials, initP, lb(1:numParams(modIdx)), ...
                ub(1:numParams(modIdx)), modIdx, sign_mu, attn_mu, sig_change);
        end
    end
end
%%
     
res = 1e2;
lb_M1 = 0.01; ub_M1 = 1;
init_M1 = rand*(ub_M1-lb_M1) + lb_M1;
sig1_range = linspace(lb_M1, ub_M1, res);
prior_M1 = 1./(1:length(sig1_range));
D1 = NaN(nsubs, size(PSEs, 2), res);

lb_M2 = [0.01 0.01]; ub_M2 = [1 1];
init_M2 = rand*(ub_M2-lb_M2) + lb_M2;
mu2_range = linspace(lb_M2(1), ub_M2(1), res);
sig2_range = linspace(lb_M2(2), ub_M2(2), res)';
flat_prior = 1/length(mu2_range)*ones(1,length(mu2_range));
M2_jointprior = (1./sig2_range) .* flat_prior;
M2_jointprior = M2_jointprior';
M2_jointpriornorm = M2_jointprior/sum(M2_jointprior, 'all', 'omitnan');
figure
set(gcf,'position',[10,10,1000,400])
%subplot(1,2,1)
imagesc(sig2_range, flat_prior, M2_jointpriornorm)
box off
title('Model Two Prior')
xlabel('\sigma')
ylabel('\mu')
colorbar
[MU2,SIGMA2]= meshgrid(mu2_range,sig2_range);
D2 = NaN(nsubs, size(PSEs, 2), res);

lb_M3 = [0.01 0.01 0.01]; ub_M3 = [1 1 1];
init_M3 = rand*(ub_M3-lb_M3) + lb_M3;
mu3_1_range = linspace(lb_M3(1), ub_M3(1), res);
mu3_2_range = linspace(lb_M3(2), ub_M3(2), res);
sig3_range = linspace(lb_M3(3), ub_M3(3), res);
D3 = NaN(nsubs, size(PSEs, 2), res, res, res);

lb_M4 = [0.01 0.01 0.01 0.01]; ub_M4 = [1 1 1 1];
init_M4 = rand*(ub_M4-lb_M4) + lb_M4;
mu4_1_range = linspace(lb_M4(1), ub_M4(1), res);
mu4_2_range = linspace(lb_M4(2), ub_M4(2), res);
sig4_1_range = linspace(lb_M4(3), ub_M4(3), res);
sig4_2_range = linspace(lb_M4(4), ub_M4(4), res);
D4 = NaN(nsubs, size(PSEs, 2), res, res, res, res);

options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display', 'off');
mfit_curves = NaN(nsubs, gt_models, fit_models, length(cohLevs));

for sub = 1:nsubs
    for modelN =  1:size(PSEs, 2)
        nR = squeeze(sum(data(modelN, :, :, sub, :), 5));
        
        M1_nlogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [0 p]), 1:4));
        [estP_M1, NLLs(sub, modelN, 1)] = fmincon(M1_nlogNL, init_M1, [], [], [], [], lb_M1, ub_M1, [], options);
        parfor ii = 1:res
            D1(sub, modelN, ii) = M1_nlogNL(sig1_range(ii));
        end
        idx1 = find(D1(sub, modelN, :) == min(D1(sub, modelN, :)));
        bestNLL1 = D1(sub, modelN, idx1);
        interim1 = -D1(sub, modelN, :) + log(prior_M1);
        c1 = max(interim1(:));
        normalized_interim1 = interim1 - c1;
        c1_norm = max(normalized_interim1(:));
        MEs(sub, modelN, 1) = sum(exp(normalized_interim1(:)))*exp(c1_norm);
        

        
        M2_nlogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx) p(2)]), 1:4));
        [estP_M2, NLLs(sub, modelN, 2)] = fmincon(M2_nlogNL, init_M2, [], [], [], [], lb_M2, ub_M2, [], options);
        parfor ii = 1:res
            D2(sub, modelN, ii) = M2_nlogNL([mu2_range(ii), sig2_range(ii)]);
        end
        idx2 = find(D2(sub, modelN, :) == min(D2(sub, modelN, :)));
        bestNLL2 = D2(sub, modelN, idx2);
        interim2 = -D2(sub, modelN, :) + log(M2_jointpriornorm);
        c2 = max(interim2(:));
        normalized_interim2 = interim2 - c2;
        c2_norm = max(normalized_interim2(:));
        MEs(sub, modelN, 2) = sum(exp(normalized_interim2(:)))*exp(c2_norm);
        

        M3_nlogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)]), 1:4));
        [estP_M3, NLLs(sub, modelN, 3)] = fmincon(M3_nlogNL, init_M3, [], [], [], [], lb_M3, ub_M3, [], options);
%         parfor ii = 1:res
%             for jj = 1:res
%                 for kk = 1:res
%                     D3(sub, modelN, ii, jj, kk) = M3_nlogNL([mu3_1_range(ii), mu3_2_range(jj), sig3_range(kk)]);
%                 end
%             end
%         end
        
        
        M4_nlogNL = @(p) sum(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)+p(4)*sig_change(idx)]), 1:4));
        [estP_M4, NLLs(sub, modelN, 4)] = fmincon(M4_nlogNL, init_M4, [], [], [], [], lb_M4, ub_M4, [], options);
    end
end
