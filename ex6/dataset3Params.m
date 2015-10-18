function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
%sigma = 0.3;
sigma = 0.1;
return;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%define list of values to try
listVals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

%do first iteration to get first predError value
model= svmTrain(X, y, 0.01, @(x1, x2) gaussianKernel(x1, x2, 0.01));
predictions = svmPredict(model,Xval);
predError = mean(double(predictions~=yval));        

for cval = listVals
    fprintf('trying cval = %f\n',cval);

    for sigval = listVals
        fprintf('trying sigma = %f',sigval);
        model= svmTrain(X, y, cval, @(x1, x2) gaussianKernel(x1, x2, sigval));
        predictions = svmPredict(model,Xval);
        predError_new = mean(double(predictions~=yval));        
        fprintf('old prediction error: %f | new: %f\n',predError, predError_new);
        if predError_new < predError
            predError = predError_new;
            C = cval;
            fprintf('C:%f\n',C);
            sigma = sigval;
            fprintf('sigma:%f\n',sigma);
        end
    end
end

% =========================================================================

end
