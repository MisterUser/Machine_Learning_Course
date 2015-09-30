function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
num_features = size(X,2);
mu = zeros(1, num_features);
sigma = zeros(1, num_features);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
    mvDiv = @(A,B) A ./ B;
    mu = mean(X); %calculates means for all features
    sigma = std(X); %all std's for all features
    X_norm = bsxfun(@minus,X_norm, mu); %make mean of each feature 0 by subtracting mean
    X_norm = bsxfun(mvDiv,X_norm, sigma);


% ============================================================

end
