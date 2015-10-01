function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
inv_m = 1/m;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

    %h_theta(x) = g(theta'*x) = 1/(1+exp(-theta'*X)
    h_theta = sigmoid(X*theta);
    J = inv_m * sum(-1 * y' * log(h_theta) - (ones(m,1)-y)' * log(ones(m,1)-h_theta));
    %h_theta is an mx1 vector of constants which resulted from g()
    %  subtracting y gives an mx1 vector with elements h(xi) - y(i)
    %  invert this vector to get 1xm
    %  X is mx(length(theta)
    %  multiplying 1xm vector by mx(l_theta) gives 1 x (l_theta)
    %  grad is l_thetax1 -> so need to inverse
    grad = (inv_m * (h_theta - y)'*X)'; 

% =============================================================

end
