function plotMetrics(metric, metric_mean, ylabel_text, ii)
    plot([1 2 3 4], squeeze(metric(:, ii, :))', 'o');
    hold on;
    plot([1 2 3 4], metric_mean(ii, :)', 'ks--', 'LineWidth', 1.5);
    xlim([0.5 4.5]);
    xticks([1 2 3 4]);
    xticklabels({'M1', 'M2', 'M3', 'M4'});
    xlabel('Model');
    ylabel(ylabel_text);
    title(['Data generated using Model ' num2str(ii)]);
end