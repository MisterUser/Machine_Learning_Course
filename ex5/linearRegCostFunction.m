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

%X, y, and theta are column vectors
%X = mxn, theta = nx1, y = mx1
J = 1/(2*m)*sum((X*theta - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);

%X*theta-y = mx1 , transverse is 1xm. Multiply by X (mxn) = 1xn. (one for each theta)
%  no need for the sum term, because the matrix mult does it automatically
%  Then take transverse so that can add to column vector of l/m *theta
grad = ((1/m) .* ((X*theta - y)'*X))';
%add regularization to all grad terms except theta0
grad(2:end) = grad(2:end) + (lambda/m).*theta(2:end);

% =========================================================================

%grad = grad(:);

end
