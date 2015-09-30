function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
%m = length(y); % number of training examples
K = alpha/length(y); %alpha/m for coefficient -> constant
theta_len = length(theta);
theta_temp = zeros(theta_len,1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    for t_par = 1:theta_len
       theta_temp(t_par) = theta(t_par) - K*sum((X*theta - y)'*X(:,t_par));
    end

    %theta1_temp = theta(1) - K*sum(X*theta - y)
    %theta2_temp = theta(2) - K*sum((X*theta - y)'*X(:,2))

    theta = theta_temp;

    %theta(1) = theta1_temp;
    %theta(2) = theta2_temp;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
