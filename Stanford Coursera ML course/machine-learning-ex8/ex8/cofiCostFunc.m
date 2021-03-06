function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% See the equation of section 2.2.1 in the ex8.pdf file
%size(Theta') % 3x4
%size(X)    % 5x3
%size(Y)    % 5x4
%size(R)    % 5x4
J = (1/2) * sum(sum(R.*((X*Theta'-Y).^2))) ...
    + (lambda/2)*sum(sum(Theta.^2))   ...
    + (lambda/2)*sum(sum(X.^2));

% See page 10 in ex8.pdf

% X_grad must be same dimensions as X, which is 5x3
% So below we have 
% 5x4 .* ( ( 5x3 * 3x4 - 5x4 ) ) * 4x3  = 5x4 * 4x3 = 5x3
X_grad = R.*((X*Theta'-Y)) * Theta   + lambda*X;

% Theta_grad must be 4x3
% 5x4 .* ( ( 5x3 * 3x4 - 5x4 ) ) * 3x5  = 4x5 * 5x3 = 4x3
Theta_grad = X' * (R.*((X*Theta'-Y)));
Theta_grad = Theta_grad' + lambda*Theta;

%size(X_grad)
%size(Theta_grad)

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
