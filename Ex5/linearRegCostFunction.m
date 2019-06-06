function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
error = h - y;
square.error = sum(error .^ 2);
regTerm1 = (1/(2 * m)) * square.error;

thetatemp = theta(2:end);
sumtheta  = sum(thetatemp .^ 2);
regTerm2 = (lambda / (2 * m)) * sumtheta;

J = regTerm1 + regTerm2;

grad = (1/m) * X' * error;
grad(2:end) += (lambda / m) * thetatemp;










% =========================================================================

grad = grad(:);

end
