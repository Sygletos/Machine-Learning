function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%mOnes = ones(size(z));  
%mOnes ./ (mOnes + exp(-z)) % is this necessary? no! See below

g = 1 ./ (1+exp(-z));

% =============================================================

end
