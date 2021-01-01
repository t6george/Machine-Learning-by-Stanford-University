function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = X * theta;
hy = h - y;

J = (sum((hy) .^ 2, 'all') + lambda * (sum(theta .^ 2, 'all') - theta(1) ^ 2)) / (2 * m);

grad = (X' * hy + lambda * [0; theta(2:end)]) / m;

grad = grad(:);

end

