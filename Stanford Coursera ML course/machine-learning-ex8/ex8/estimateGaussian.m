function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% X is m x n. sum(X,1) is 1 x n, sum(X,1)' is n x 1
mu = (1/m)*sum(X,1)';
% mu is n x 1, mu' is 1 x n. Make a m x n matrix where mu is repeated in m rows
%size(mu)
mu_matrix = repmat(mu',[m,1]);

% Second argument equal to 0 (1) uses N-1 (N) normalization, 
% 3rd argument is dimension of variance sum. See "help var"
sigma2 = var(X-mu_matrix,1, 1);
% sigma2 should be 1 x n so we transpose it
sigma2 = sigma2';



% =============================================================


end
