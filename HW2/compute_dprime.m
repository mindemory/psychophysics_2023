function dP = compute_dprime(stimuli, seq, side, noiseStatus, biasVal, historyEffect)
% Created by Mrugank (12/21/2023): This function performs a decision making
% task for the given set of stimuli, seq which is the stimuli scrambled and
% organized into trials, side which determines which side the stimulus is
% presented on.
% Additonally, noiseStatus can be constant Noise or signal-dependent noise
% which is Weber's law noise.
% biasValue when 0 implies no bias and decision is based on if evidence >
% 0, however biasValue can be changed either by intervalBias thereby
% favoring one side more over the other, and/or by historyEffects.
% In this scenari, historyEffect is dependent on previous five-trials with
% a decaying impact of trials further in the past.
if nargin < 6
    historyEffect = 0;
end

totalTrials = length(seq);
baseNoiseLevel = 0.1;
evidenceFunc = @(noiseScale) arrayfun(@(idx) normrnd(seq(idx), ...
    baseNoiseLevel + sqrt(noiseScale*(seq(idx)/max(stimuli))^2)) + normrnd(0,1), ...
    1:totalTrials);

% Computing evidence based on noiseStatus
switch noiseStatus
    case 'constant'
        evidence = evidenceFunc(0); % Constant noise, no additional scaling
    case 'signal-dependent'
        evidence = evidenceFunc(1); % Weber's law noise, scale with stimulus
    otherwise
        error('Invalid noiseStatus');
end

% Initialize response and history effect variables
response = zeros(totalTrials, 1);
historyBiasAdjustment = 0; % Starting with no adjustment

% Compute response for each trial, including history effect
for ii = 1:totalTrials
    if historyEffect > 0 && ii > 6
        historyBiasAdjustment = 0;
        for jj = 1:5
            historyBiasAdjustment = historyBiasAdjustment + (1/jj) * (response(ii-jj) == side(ii-jj));
        end
    end
    % Compute response with adjusted bias
    currentBiasVal = biasVal + historyBiasAdjustment;
    response(ii) = side(ii).*(evidence(ii)>currentBiasVal) + (-side(ii)).*(evidence(ii)<=currentBiasVal);
end

% d-prime
dP = zeros(length(stimuli), 1);
for ii = 1:length(stimuli)
    sLev = stimuli(ii);
    [~,~,~,~, dP(ii)] = calcDPrime(response(seq == sLev), side(seq == sLev));
end

end