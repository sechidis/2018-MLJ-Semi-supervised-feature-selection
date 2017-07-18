function [selectedFeatures] = JMI(X_data,Y_labels, topK)
% Summary 
%    JMI algorithm for feature selection
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_labels: n x 1 vector with the labels
%    topK: Number of features to be selected


numFeatures = size(X_data,2);

mi_score = zeros(1,numFeatures);

for index_feature = 1:numFeatures
    score_per_feature(index_feature) = mi(X_data(:,index_feature),Y_labels);
end
[val_max,selectedFeatures(1)]= max(score_per_feature);
mi_score(selectedFeatures(1)) = val_max;
not_selected_features = setdiff(1:numFeatures,selectedFeatures);

%%% Efficient implementation of the second step, at this point I will store
%%% the score of each feature. Whenever I select a feature I put NaN score
score_per_feature = zeros(1,numFeatures);
score_per_feature(selectedFeatures(1)) = NaN;
count = 2;
while count<=topK

    for index_feature_ns = 1:length(not_selected_features)

            score_per_feature(not_selected_features(index_feature_ns)) = score_per_feature(not_selected_features(index_feature_ns))+mi([X_data(:,not_selected_features(index_feature_ns)),X_data(:, selectedFeatures(count-1))], Y_labels);
      
    end
    
    [val_max,selectedFeatures(count)]= nanmax(score_per_feature);

    
   score_per_feature(selectedFeatures(count)) = NaN;
    not_selected_features = setdiff(1:numFeatures,selectedFeatures);
    count = count+1;
end


