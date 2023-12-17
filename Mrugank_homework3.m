N = 100; 
coherenceLevels = -1:0.1:1; 
conditions = 4; 
data = zeros(N, length(coherenceLevels), conditions);

for i = 1:N
    PSEs = rand(1, conditions) * 0.4 - 0.2; 
    slopes = rand(1, conditions) * 2 + 1;  

    for cond = 1:conditions
        for j = 1:length(coherenceLevels)
            coh = coherenceLevels(j);
            data(i, j, cond) = normcdf(coh, PSEs(cond), 1/slopes(cond));
        end
    end
end

figure();
plot(coherenceLevels, squeeze(mean(data, 1)))