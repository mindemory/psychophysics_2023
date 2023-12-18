clear; close all; clc;

%% Initialization and Data Generation:

% Sets coherence levels, number of trials, subjects, and initializes the psychometric function generator.
% Generates simulated data for each model with varying Point of Subjective Equality (PSE) and slope values.
cohLevs = linspace(-1, 1, 20);
Ncoh = length(cohLevs);
ntrials = 1e2;
nsubs = 7;

psyfuncGenerator = @(x, p) (0.5*0)+(1-0).*normcdf(x, p(1), p(2));
PSEs = [0 0 0 0;
    -0.3 -0.3 0.3 0.3;
    -0.5 -0.3 0.3 0.5;
    -0.6 -0.3 0.3 0.6];

slp = 4;
slopes = [slp slp slp slp;
          slp slp slp slp;
          slp slp slp slp;
          slp*1.5 slp slp slp*1.5];

data = zeros(size(PSEs, 2), size(PSEs, 1), Ncoh, nsubs, ntrials);
color_vect = ["r*", "b*", "k*", "m*"];
legnames = {'Adapt L: Attend yes', ...
            'Adapt L: Attend no', ...
            'Adapt R: Attend yes', ...
            'Adapt R: Attend no'};

figure();
for modelN =  1:size(PSEs, 2)
    subplot(2, 2, modelN)
    hold on;
    for ttype = 1:size(PSEs, 1)
        for sub = 1:nsubs
            this_PSE = PSEs(modelN, ttype) + randn(1, 1)*PSEs(modelN, ttype)/6;
            this_slope = slopes(modelN, ttype) + randn(1, 1)*slopes(modelN, ttype)/6;
            this_psyfunc = psyfuncGenerator(cohLevs, [this_PSE, 1/this_slope]);
            this_data = rand(Ncoh, ntrials) < repmat(this_psyfunc', 1, ntrials);
            data(modelN, ttype, :, sub, :) = this_data;
            plot(cohLevs, squeeze(mean(this_data, 2)), color_vect(ttype));%, 'DisplayName', legnames{ttype})
        end
    end
    xlabel('Coherence Level')
    ylabel('Prob correct for Right')
    title(['Model: ' num2str(modelN)])
end

% Initializes variables for sign, attention, and slope change, needed for different model conditions.
sign_mu = [-1 -1 1 1];
attn_mu = [-1 0 0 1];
sig_change = [1 0 0 1];
% Defines the negative log-likelihood function that will be minimized.
nLogL = @(NR, NT, p) -sum(NR.*log(psyfuncGenerator(cohLevs, [p(1) p(2)])+0.01) + ...
             (NT-NR).*log(1-psyfuncGenerator(cohLevs, [p(1) p(2)])+0.01));
        
NLLs = zeros(nsubs, size(PSEs, 2), size(PSEs, 2));
% Loops over subjects and models, fitting each model to the data and calculating the negative log-likelihood for each fit.
% For each model, uses fmincon to minimize the negative log-likelihood and find the best-fitting parameters.
for sub = 1:nsubs
    for modelN =  1:size(PSEs, 2)
%         disp([sub, modelN])
        nR = squeeze(sum(data(modelN, :, :, sub, :), 5));

        M1_nlogNL = @(p) sum(cell2mat(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [0 p]), 1:4, 'UniformOutput', false)));
        options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display', 'off');
        lb_M1 = 0.01; ub_M1 = 2;
        init_M1 = rand*(ub_M1-lb_M1) + lb_M1;
        [estP_M1, NLLs(sub, modelN, 1)] = fmincon(M1_nlogNL, init_M1, [], [], [], [], lb_M1, ub_M1, [], options);
        
        M2_nlogNL = @(p) sum(cell2mat(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx) p(2)]), 1:4, 'UniformOutput', false)));
        lb_M2 = [0.01 0.01]; ub_M2 = [2 2];
        init_M2 = rand*(ub_M2-lb_M2) + lb_M2;
        [estP_M2, NLLs(sub, modelN, 2)] = fmincon(M2_nlogNL, init_M2, [], [], [], [], lb_M2, ub_M2, [], options);
        
        M3_nlogNL = @(p) sum(cell2mat(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)]), 1:4, 'UniformOutput', false)));
        lb_M3 = [0.01 0.01 0.01]; ub_M3 = [2 2 2];
        init_M3 = rand*(ub_M3-lb_M3) + lb_M3;
        [estP_M3, NLLs(sub, modelN, 3)] = fmincon(M3_nlogNL, init_M3, [], [], [], [], lb_M3, ub_M3, [], options);
        
        M4_nlogNL = @(p) sum(cell2mat(arrayfun(@(idx) nLogL(nR(idx, :), ntrials, [p(1)*sign_mu(idx)+p(2)*attn_mu(idx), p(3)+p(4)*sig_change(idx)]), 1:4, 'UniformOutput', false)));
        lb_M4 = [0.01 0.01 0.01 0.01]; ub_M4 = [2 2 2 2];
        init_M4 = rand*(ub_M4-lb_M4) + lb_M4;
        [estP_M4, NLLs(sub, modelN, 4)] = fmincon(M4_nlogNL, init_M4, [], [], [], [], lb_M4, ub_M4, [], options);

    end
end

%% Calculating AIC and BIC:
% AIC and BIC for each model and subject, which helps in model comparison. Lower AIC/BIC values indicate a better model fit considering the complexity.
k = 1; 
n = 4 * ntrials;
AICc = 2*k + 2*NLLs;
BICc = k*log(n) + 2*NLLs;

% Plotting the log of NLL, AIC, and BIC across different models for each data set. This visual representation aids in comparing how well each model fits the data.
figure();
for ii = 1:4
    subplot(4, 3, 3*(ii-1)+1)
    plot([1 2 3 4], log(squeeze(NLLs(:, ii, :))'), 'o')
    xlim([0.5 4.5])
    xlabel('Model')
    ylabel('Negative log-likelihood')
    title(['Data generated using Model ' num2str(ii)])
    
    subplot(4, 3, 3*(ii-1)+2)
    plot([1 2 3 4], log(squeeze(AICc(:, ii, :))'), 'o')
    xlim([0.5 4.5])
    xlabel('Model')
    ylabel('AIC')
    title(['Data generated using Model ' num2str(ii)])
    
    subplot(4, 3, 3*(ii-1)+3)
    plot([1 2 3 4], log(squeeze(BICc(:, ii, :))'), 'o')
    xlim([0.5 4.5])
    xlabel('Model')
    ylabel('BIC')
    title(['Data generated using Model ' num2str(ii)])
end


% Calculate Bayes Factors
bayesFactor = zeros(nsubs, size(PSEs, 2), size(PSEs, 2), size(PSEs, 2));
for sub = 1:nsubs
    for dataModel = 1:size(PSEs, 2)
        for model1 = 1:size(PSEs, 2)
            for model2 = 1:size(PSEs, 2)
                bayesFactor(sub, dataModel, model1, model2) = exp((BICc(sub, dataModel, model2) - BICc(sub, dataModel, model1))/2);
            end
        end
    end
end

% Plotting Bayes Factors
figure();
for ii = 1:size(PSEs, 2)
    for jj = 1:size(PSEs, 2)
        subplot(size(PSEs, 2), size(PSEs, 2), (ii-1)*size(PSEs, 2) + jj)
        imagesc(squeeze(bayesFactor(:, ii, jj, :)))
        colorbar
        title(['BF Model ' num2str(jj) ' over Model ' num2str(ii)])
        xlabel('Model')
        ylabel('Subject')
    end
end

