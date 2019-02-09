function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% compute cost
tmp_theta = theta([2 : size(theta, 1)] , :);
tmp_theta = tmp_theta .^ 2;
J = costFunction(theta, X, y) + ((lambda/(2*m)) * sum(tmp_theta));


% compute gradient
g = sigmoid(X * theta);
grad_0 = (1/m) * sum((g - y));

grad_tmp = (1/m) * (((g - y)' * X)' + (lambda * theta));

grad = [grad_0; grad_tmp([2 : size(theta, 1)] , :)];

% =============================================================

end
