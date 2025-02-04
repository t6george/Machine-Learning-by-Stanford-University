function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_tmp = theta;

for iter = 1:num_iters
    for j = 1:length(theta)
        J = 0;
        for i = 1:m
            J = J + (X(i,:) * theta - y(i)) * X(i,j);
        end
        
        theta_tmp(j) = theta(j) - alpha * J / m;
    end
    
    theta = theta_tmp;

    J_history(iter) = computeCostMulti(X, y, theta);

end

end
