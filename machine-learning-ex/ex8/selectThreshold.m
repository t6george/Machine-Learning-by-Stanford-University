function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    pred = pval < epsilon;
    predAnomaly = pred == 1;
    actualAnomaly = yval == 1;
    
    fp = sum(predAnomaly & (yval == 0));
    fn = sum((pred == 0) & actualAnomaly);
    tp = sum(predAnomaly & actualAnomaly);

    prec = tp / (tp + fp);
    recall = tp / (tp + fn);

    F1 = (2 * prec * recall) / (prec + recall);
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end

