function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

o = ones(m, 1);

z2 = [o, X] * Theta1';
a2 = [o, sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

logh = log(a3);
log1mh = log(1-a3);
one_hot = zeros(num_labels, 1);

for i = 1:m
    one_hot(y(i)) = 1;
    J = J + logh(i,:) * one_hot + log1mh(i,:) * (1 - one_hot);    
    one_hot(y(i)) = 0;
end

J = J / (-m);

J = J + lambda * (sum((Theta1(:,2:end) .^ 2),'all') + sum((Theta2(:,2:end) .^ 2), 'all')) / (2 * m);

sigGrad = sigmoidGradient(z2);

for i = 1:m
    one_hot(y(i)) = 1;
    
    delta3 = a3(i,:)' - one_hot;
    
    delta2 = (Theta2' * delta3) .* [1; sigGrad(i,:)'];
    delta2 = delta2(2:end);
    
    Theta1_grad = Theta1_grad + delta2 * [1, X(i,:)];
    Theta2_grad = Theta2_grad + delta3 * a2(i,:);
    
    one_hot(y(i)) = 0;
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

Theta1_grad = Theta1_grad + (lambda / m) * ([zeros(size(Theta1,1),1), Theta1(:, 2:end)]);
Theta2_grad = Theta2_grad + (lambda / m) * ([zeros(size(Theta2,1),1), Theta2(:, 2:end)]);

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

