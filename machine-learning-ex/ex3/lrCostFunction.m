function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

J = -(y' * log(h) + (1-y)' * log(1-h)) + (ones(1, length(theta)) * (theta .^ 2) - theta(1) ^ 2) * lambda / 2;

J = J / m;

h = h - y;

theta(1) = 0;

grad = (X' * h) ./ m + lambda * theta / m;


grad = grad(:);

end

