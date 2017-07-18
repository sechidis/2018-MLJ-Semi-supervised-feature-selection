function new_data = disc_dataset_equalwidth( X_data, bins )
% Summary: 
%    This function discretizes each feature of a given dataset in an
%    equal-width manner
% Inputs:
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    bins: the number of categories

for fnum = 1:size(X_data,2)
    if length(unique(X_data(:,fnum)))<=bins
        [~,~,new_data(:,fnum)] = unique(X_data(:,fnum)); % To have as many categories as the alphabet
    else
        feat = X_data(:,fnum);
        minval = min(feat);
        width = abs(max(feat)-min(feat))/bins;
        
        %create boundaries for equal width
        boundaryend=0;
        for n=1:bins
            boundaryend(n) = minval + n*width;
        end
        boundaryend(bins) = boundaryend(bins) + 1; 
        
        
        
        lastboundaryend = minval;
        newfeature=0;
        for n=1:bins
            indices = find( feat>=lastboundaryend & feat<boundaryend(n) );
            newfeature(indices) = n;
            lastboundaryend = boundaryend(n);
        end
        
         [~,~,new_data(:,fnum)] = unique(newfeature); % To have as many categories as the alphabet
        
    end
end
