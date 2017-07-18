% Load one of the provided dataset, e.g.
load('./Datasets/krvskp.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pre-process the data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transform multi-class problems to binary in 1-vs-all strategy
Y_labels(Y_labels~=1) =0;
% Discretise continuous features via an equal-width-strategy, for example using 5 bins
X_data = disc_dataset_equalwidth(X_data,5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate semi-supervised datasets %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let's assume that we want 25% of the examples to be labelled
pS1 = 0.25; % probability p(s=1)
% In this example we will generate data under MAR-C, as we did in Section 6
% We generate the  semi-supervised datasets under the class-prior-change 
% scenario by randomly  over-sampling the positive class, such that the 
% probability of  a labelled example being positive
% p(y=1|s=1) = c*p(y=1), which means that:
% p(s=1|y=1) = c*p(s=1) and p(s=1|y=0) = (1-1.5*p(y=1))*p(s=1)/p(y=0)
% So, in order to generate the  labelled set we should sample the positive
% examples with probability p(s=1|y=1) and the negative examples with
% p(s=1|y=0).
c = 1.25; % For c=1. we are in the MCAR scenario, while for c>1 in MAR-C
pY1 = mean(Y_labels); % probability p(y=1)
pS1givenY1 = c*pS1; % probability p(s=1|y=1)
pS0givenY0 = (1-c*pY1)*pS1/ (1-pY1); % probability p(s=1|y=0)

% Label some positive examples
ypos_indices = find(Y_labels==1);%Find the positive indeces
positiveSet = binornd(1,pS1givenY1,1,length(ypos_indices))';% Find which examples will be labelled
while sum(positiveSet~=0) == 0 && pS1givenY1>0 % Check if you have empty labelled set, if yes re-sample
    positiveSet = binornd(1,pS1givenY1,1,length(ypos_indices))';
end; 
S1Y1_indices = ypos_indices(find(positiveSet==1));  %update

% Label some negative examples
yneg_indices = find(Y_labels==0);%Find the positive indeces
negativeSet = binornd(1,pS0givenY0,1,length(yneg_indices))';% Find which examples will be labelled
while sum(negativeSet~=0) == 0 && pS0givenY0>0 % Check if you have empty labelled set, if yes re-sample
    negativeSet = binornd(1,pS0givenY0,1,length(yneg_indices))';
end; 
S1Y0_indices = yneg_indices(find(negativeSet==1));  %update

% So the proxy variable that we observe in the semi-supervised scenario is:
Y_proxy = NaN(size(X_data,1),1); % Initialize with all values missing, NaN
Y_proxy(S1Y1_indices) = 1; % The positvely-labelled examples
Y_proxy(S1Y0_indices) = 0; % The negatively-labelled examples


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Select the features using our suggested algorithms %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we will select the top-10 features with Semi-JMI and we will compare
% these subsets with the idea, which is the one that we have if we use JMI 
% in the unobserved class labels Y
topK=10;
Selected_with_semiJMI = semiJMI(X_data, Y_proxy, topK, pY1);
disp('Returned subset using our Semi-JMI:')
disp(Selected_with_semiJMI)
Selected_with_ideal_JMI = JMI(X_data, Y_labels, topK);
disp('Returned subset using JMI with unobserved class labels Y:')
disp(Selected_with_ideal_JMI)
% We repeat the same for MIM
Selected_with_semiMIM = semiMIM(X_data, Y_proxy, topK, pY1);
disp('Returned subset using our Semi-MIM:')
disp(Selected_with_semiMIM)
Selected_with_ideal_MIM = MIM(X_data, Y_labels, topK);
disp('Returned subset using MIM with unobserved class labels Y:')
disp(Selected_with_ideal_MIM)
% And the same with IAMB
alpha = 0.05;
Selected_with_semiIAMB = semiIAMB(X_data, Y_proxy, alpha, pY1);
disp('Returned subset using our Semi-IAMB:')
disp(Selected_with_semiIAMB)
Selected_with_ideal_IAMB = IAMB(X_data, Y_labels, alpha);
disp('Returned subset using IAMB with unobserved class labels Y:')
disp(Selected_with_semiIAMB)