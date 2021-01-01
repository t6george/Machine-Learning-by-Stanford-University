function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
minErr = 1;

for c = 1:length(vals)
    for s = 1:length(vals)
        predictions = svmPredict(svmTrain(X, y, vals(c), @(x1, x2) gaussianKernel(x1, x2, vals(s))) , Xval);
        err = mean(double(predictions ~= yval));
        if err < minErr
            minErr = err;
            C = vals(c);
            sigma = vals(s);
        end
    end
end

end

