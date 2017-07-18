function [selectedFeatures] = semiJMI(X_data, Y_proxy, topK, prior_y)
% Summary 
%    Semi-JMI algorithm for feature selection (Sechidis Brown, 2017) 
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_proxy: n x 1 vector \tilde{Y}, with values 1 and 0 for the positevely
%             labelled and negatively labelled examples respectively (labelled data),
%             and NaN for the examples with missing label (unlabelled data)
%    topK: Number of features to be selected
%    prior_y: Our bilief over the marginal probabilty p(y=1), default value
%             is the probability in the labelled set p(Y_proxy=1)


if nargin<3
    error('Not enough input arguments');
else if nargin<4
        prior_y = mean(Y_proxy==1);
    end
end

% Step 1: Initialise
n = sum(Y_proxy==0); % number of negatives supplied with labels
p = sum(Y_proxy==1); % number of positives supplied with labels
m = sum(isnan(Y_proxy)); % number of missing labels

% Step 2: Create surrogate variables
Y_proxy_0 = Y_proxy; Y_proxy_0(isnan(Y_proxy)) = 0;
Y_proxy_1 = Y_proxy; Y_proxy_1(isnan(Y_proxy)) = 1;

% Step 3: Calculate switching threshold
a = sqrt(p*(p+m));
b = sqrt(n*(n+m));
phi = a/(a+b);

% Step 4: Decide optimal surrogate (Theorem 8) and use it in IAMB to derive MB
if prior_y < phi
    Y_labels = Y_proxy_0;
else
    Y_labels = Y_proxy_1;
end
selectedFeatures = JMI(X_data,Y_labels, topK);