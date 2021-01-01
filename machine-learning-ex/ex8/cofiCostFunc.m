function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
          
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

diff = (X * Theta' - Y) .* R;

J = (sum(diff .^ 2,'all') / 2) + lambda * (sum(Theta .^ 2, 'all') + sum(X .^ 2, 'all')) / 2;

Theta_grad = (X' * diff)' + lambda * Theta;

X_grad = (diff * Theta) + lambda * X;

grad = [X_grad(:); Theta_grad(:)];

end

