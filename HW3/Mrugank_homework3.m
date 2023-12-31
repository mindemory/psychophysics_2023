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


psyfuncGenerator = @(x, p) (0.5*0.05)+(1-0.05).*normcdf(x, p(1), p(2));
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
MEs = NaN(nsubs, gt_models, fit_models); 
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
            [estP, NLLs(sub, modelN, modIdx), MEs(sub, modelN, modIdx), ...
                mfit_curves(sub, modelN, modIdx, :, :)] = modelOptimization(psyfuncGenerator, cohLevs, nR, ...
                ntrials, initP, lb(1:numParams(modIdx)), ub(1:numParams(modIdx)), modIdx, sign_mu, attn_mu, sig_change);
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
BF_mean = squeeze(mean(BF_estim, 1));
evi_mean = squeeze(mean(MEs, 1));

% Plotting the log of NLL, AIC, BIC, BF, and model evidence
figure();
for ii = 1:4
    subplot(4, 5, 5*(ii-1)+1);
    plotMetrics(NLLs, NLL_mean, 'NLL', ii);

    subplot(4, 5, 5*(ii-1)+2);
    plotMetrics(AICc, AIC_mean, 'AIC', ii);

    subplot(4, 5, 5*(ii-1)+3);
    plotMetrics(BICc, BIC_mean, 'BIC', ii);

    subplot(4, 5, 5*(ii-1)+4);
    plotMetrics(BF_estim, BF_mean, 'BF estim', ii);

    subplot(4, 5, 5*(ii-1)+5);
    plotMetrics(MEs, evi_mean, 'model evidence', ii);
end