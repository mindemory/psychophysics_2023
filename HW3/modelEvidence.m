plotFlag_modelEvi = 0;

get_evi = @(nLL, prior, c) sum(exp(-nLL + log(prior) - c)) * exp(c);

nPSE = 50; PSE_all = linspace(lb_PSE, ub_PSE, nPSE);
nslope = 50; slope_all = linspace(lb_slope, ub_slope, nslope);

switch imodel_fit
    case 1, param_space = allcomb(slope_all);
    case 2, param_space = allcomb(PSE_all, slope_all);
    case 3, param_space = allcomb(PSE_all, PSE_all, slope_all);
    case 4, param_space = allcomb(PSE_all, PSE_all, slope_all, slope_all);
end

nparamComb  = size(param_space, 1);
record = nan(nparamComb, 2);

% parpool(4)
parfor ip = 1:nparamComb
    param = param_space(ip,:); % a given pair of params
    
    % get nLL
    nLL = get_nLL(imodel_fit, coh, data_new, param);
    
    % get prior
    PSE_prior = 1/nPSE;
    switch imodel_fit
        case 1, prior = 1/param;
        case 2, prior = PSE_prior * 1/param(2);
        case 3, prior = PSE_prior^2 * 1/param(3);
        case 4, prior = PSE_prior^2 * 1/param(3) * 1/param(4);
    end
    
    record(ip, :) = [nLL,prior];
end

% normalize
record(:,2) = record(:,2) / sum(record(:,2));

%%
if plotFlag_modelEvi 
    modelEvi_titles = {'nLL space', 'prior space'};
    figure('Position', [0 200 800 300])
    for n = 1:2
        subplot(1,2,n), hold on
        imagesc(PSE_all, slope_all, (reshape(record(:, n), nPSE, nslope)))
        if n==1
            [~ , min_nLL_ind] = min(record(:, 1));
            min_nLL_y = mod(min_nLL_ind, nPSE);
            min_nLL_x = round(min_nLL_ind/nPSE);
            plot(PSE_all(min_nLL_x), slope_all(min_nLL_y), 'r*', 'MarkerSize', 10)
            text(PSE_all(min_nLL_x) - .15, slope_all(min_nLL_y) + .1, sprintf('PSE = %.4f\nslope = %.4f', PSE_all(min_nLL_x), slope_all(min_nLL_y)), 'Color', 'w', 'FontSize', 12)
            plot(PSE_att, slope0, 'c+', 'MarkerSize', 10)
            legend('Optimal parameter', 'True parameter', 'Location', 'northeast')
        end
        colorbar
        xlim([PSE_all(1), PSE_all(end)])
        ylim([slope_all(1), slope_all(end)])
        xlabel(sprintf('PSE (%d bins)', nPSE), 'FontSize', sz_label)
        ylabel(sprintf('slope (%d bins)', nslope), 'FontSize', sz_label)
        axis square
        title(modelEvi_titles{n}, 'FontSize', sz_title)
    end
    suptitle(sprintf('Figure 2. Fitting M2 to the true model'))
    
    saveas(gcf, 'Fig3_nLL_prior_M2M2.jpg')
end


%%
c = max(-record(:,1) + log(record(:,2)));

%%
evi = get_evi(record(:,1), record(:,2), c);