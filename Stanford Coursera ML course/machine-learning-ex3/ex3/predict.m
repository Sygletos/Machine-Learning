function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

fprintf("size of X before bias inputs: %s\n",mat2str(size(X)))
X = [ones(m, 1) X];
fprintf("size of X after bias inputs: %s\n",mat2str(size(X)))
fprintf("size of Theta1: %s\n", mat2str(size(Theta1)))
a1 = X*Theta1';

fprintf("size of a1 b4 adding bias unit: %s\n", mat2str(size(a1)))
a1 = [ ones(m,1) a1];
fprintf("size of a1 with bias unit: %s\n", mat2str(size(a1)))

fprintf("size of Theta2: %s\n", mat2str(size(Theta2)))
a2 = sigmoid(a1*Theta2');
fprintf("size of output layer a2: %s\n", mat2str(size(a2)))
fprintf("m = %d\n", m)

[Y,I] = max(a2,[],2);
sprintf("%f, %f, %f, %f", a2(1,1),a2(1,2),a2(1,3),a2(1,I))
size(Y)
size(I)
p = I;

% =========================================================================


end
