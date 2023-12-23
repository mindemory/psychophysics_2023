clear; close all; clc;
%%
% The experiment here is a 2AFC method of constant stimuli. There are 10
% stimuli levels and a noise level that is chosen randomly from one of the
% stimulus levels. At each trial, one of the stimuli levels are chosen and
% the stimulus is presented either on the left or right randomly, and the
% task of the subject is to determine if the stimulus is on the left or on
% the right. 
% Initialization
nStim = 7;  % Number of stimuli
nTrials = 1e3;
totalTrials = nStim * nTrials;  
stimuli = linspace(0, 10, nStim);  % Stimulus - levels selected
noiseLvl = 0;  % Noise level
% If the noise is lower than signal its a detection task, if the noise is
% in between, its a discrimination task
constNoise = 0.6; % Constant noise
intervalBias = 0.5; % Bias value, get interesting plots when bias = 1 and -1 so this is correct!

% Plot stimulus and noise distributions
% Visualziing stimuli and noise ditributions to make sure the task is hard
% enough but not too hard
figure;
x = linspace(-5, 15, 1000); 
for i = 1:nStim
    y = normpdf(x, stimuli(i), 1); 
    fill(x, y, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); 
    hold on;
end
fill(x, normpdf(x, noiseLvl, 1), 'r', 'FaceAlpha', 0.5);
hStimulus = plot(NaN, NaN, 'b', 'DisplayName', 'Stimulus');
hNoise = plot(NaN, NaN, 'r', 'DisplayName', 'Noise');
xlabel('Stimulus/Noise Level');
ylabel('Probability Density');
title('Distributions for Stimuli and Noise');
legend([hStimulus, hNoise], 'Location', 'northeast');

% Simulate side where the signal is on each trial uniformly
% Seq is the sequence of stimuli levels, chosen randomly. One step close to
% adding history effects :/
side = randsample([-1, 1], totalTrials, true);
seq = repmat(stimuli, 1, nTrials);
seq = seq(randperm(length(seq)));

% Psychometric function for performing the decision making 
psyFunc = @(x) normcdf(x, 1, 0);

% Sample noise and signal from a standard normal distribution given the
% mean which is the signal and the noise levels
noiseSamp = normrnd(noiseLvl, 1, [1, totalTrials]);
signalSamp = normrnd(seq, 1, [1, totalTrials]);

% Get the probablility of making response given the signal and noise for
% different conditions
prob = psyFunc(signalSamp - noiseSamp);
probWithNoise = psyFunc(signalSamp - noiseSamp + constNoise);
probWithIntervalBias = psyFunc(signalSamp - noiseSamp);
probWithIntervalBias(side == 1) = min(probWithIntervalBias(side == 1) + intervalBias/2, 1);
probWithIntervalBias(side == -1) = max(probWithIntervalBias(side == -1) - intervalBias/2, 0);

% Get ther respones by comparing probablity to a random number. Turnint it
% into -1 or 1 depending on the side of the stimulus
resp = 2 * (rand(1, totalTrials) < prob) - 1; 
resp = resp .* side; 
respWithNoise = 2 * (rand(1, totalTrials) < probWithNoise) - 1; 
respWithNoise = respWithNoise .* side; 
respWithIntervalBias = 2 * (rand(1, totalTrials) < probWithIntervalBias) - 1; 
respWithIntervalBias = respWithIntervalBias .* side; 

