function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];

for i = 1:num_labels
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    all_theta(i,:) = (fmincg (@(t)(lrCostFunction(t, X, (y == mod(i,10)), lambda)), all_theta(i,:)', options))';
end

end

