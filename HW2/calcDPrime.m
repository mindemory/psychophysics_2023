function [H, M, FA, CR, dP] = calcDPrime(responses, groundTruth)
    H = sum(responses(groundTruth == 1)  == 1);
    M = sum(responses(groundTruth == 1)  == -1);
    FA = sum(responses(groundTruth == -1)  == 1);
    CR = sum(responses(groundTruth == -1)  == -1);
%     hitRate = max(H/(H+M), 1e-6);
%     fAlarmRate = max(FA/(FA+CR), 1e-6);
    hitRate = max(min(H/(H+M),1-1e-4),1e-4);
    fAlarmRate = max(min(FA/(FA+CR),1-1e-4),1e-4);
    dP = norminv(hitRate) - norminv(fAlarmRate);
end