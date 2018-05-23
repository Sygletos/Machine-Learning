function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

addpath(pwd)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regTheta = Theta1;
regTheta(1) = 0;

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
%fprintf("\nsize of h2 in the beginning is %s:", mat2str(size(h2)))

% Make a 5000 x 10 matrix from y using logical vectors [0,...,1,...] instead of number labels

v_y = repmat(y(:),1,num_labels);

v_logical = repmat(1:num_labels,m,1);

Y = bsxfun(@eq, v_y, v_logical);

%%%%%%%%%%%%%% DEBUG BLOCK %%%%%%%%%%%%%
%y(1:3,1)
%v_y(1:3,:)
%v_logical(1:3,:)
%Y(1:3,:)

%fprintf("size of Y' is %s:", mat2str(size(Y')))
%fprintf("\nsize of log(h2) is %s:", mat2str(size(log(h2))))
%fprintf("\nsize of y'*log(h2) is %s:", mat2str(size(y'*log(h2))))
%fprintf("\nsize of Y'*log(h2) is %s:", mat2str(size(Y'*log(h2))))
%fprintf("\nsize of (1-h2) is %s:", mat2str(size(1 - h2)))
%fprintf("\nsize of (1-y') is %s:", mat2str(size(1 - y')))

regTheta1 = Theta1;
regTheta1(:,1) = 0;
regTheta2 = Theta2;
regTheta2(:,1) = 0;

% The k-sum is essentially a diagonal of the k x k matrix produced when
% multiplying Y'*h2
J = (1/m)*( sum(diag( -(Y')*log(h2) - (1-Y')*log(1 - h2) )) )...
  + (lambda/(2*m))*( sum(diag(regTheta1'*regTheta1)) + sum(diag(regTheta2'*regTheta2)) );

%fprintf("\nsum(diag(Jk)) is %s:", sum(diag(Jk)) )
% -------------------------------------------------------------

% =========================================================================

% Doing backpropagation 
a1 = [ones(m, 1) X];
a2 = sigmoid([ones(m, 1) X] * Theta1');
z2 = [ones(m, 1) X] * Theta1';
z2 = [ones(m, 1) z2];
a3 = sigmoid([ones(m, 1) a2] * Theta2');

Delta1=zeros(size(Theta1));
%Delta1=Delta1(:,2:end);
Delta2=zeros(size(Theta2));
%Delta2=Delta2(:,2:end);

for t = (1:m) 
    a1_t = a1(t,:);
    a2_t = a2(t,:);
    a2_t = [1 a2_t];
    z2_t = z2(t,:);
    a3_t = a3(t,:);
    y_t = y(t,:);
    Y_t = Y(t,:);
    
    delta_3 = (a3_t - Y_t);   % size num_labels x 1
%    fprintf("\nSize of Theta2: %s", mat2str(size(Theta2)))
%    fprintf("\nSize of delta_3: %s", mat2str(size(delta_3)))
%    fprintf("\nSize of sigmoidGradient(z2_t): %s", mat2str(size(sigmoidGradient(z2_t))))
%    fprintf("\nSize of (delta_3*Theta2): %s", mat2str(size(delta_3*Theta2)))

    delta_2 = (delta_3*Theta2) .* sigmoidGradient(z2_t); 
    delta_2 = delta_2(2:end);
%{
    fprintf("\nSize of delta_2: %s", mat2str(size(delta_2)))
    fprintf("\nSize of a1_t: %s", mat2str(size(a1_t)))
    fprintf("\nSize of delta_2*a1_t: %s", mat2str(size(delta_2'*a1_t)))
    fprintf("\nSize of Delta1: %s", mat2str(size(Delta1)))
    fprintf("\nSize of delta_3: %s", mat2str(size(delta_3)))
    fprintf("\nSize of a2_t: %s", mat2str(size(a2_t)))
%}
    Delta1 = Delta1 + delta_2'*a1_t;
    Delta2 = Delta2 + delta_3'*a2_t;
    
end
%Delta1 = [zeros(1,input_layer_size); Delta1];
%Delta2 = [zeros(1,hidden_layer_size); Delta2];

fprintf("\nSize of Theta1_grad: %s", mat2str(size(Theta1_grad)))
fprintf("\nSize of regTheta1: %s", mat2str(size(regTheta1)))
Theta1_grad = (1/m)*Delta1 + (lambda/m)*regTheta1;
Theta2_grad = (1/m)*Delta2 + (lambda/m)*regTheta2;

%fprintf("\nSize of Theta2_grad: %s", mat2str(size(Theta2_grad)))

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
