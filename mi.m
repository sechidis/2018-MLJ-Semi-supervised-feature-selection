function MI= mi( X, Y)
% Summary
%    Estimate mutual information I(X;Y) between two categorical variables X,Y
%    X,Y can be matrices which are converted into a joint variable before computation

[~,~,X] = unique(X,'rows');
[~,~,Y] = unique(Y,'rows');
[~,ar,X]=unique(X); arity_X=length(ar);
[~,ar,Y]=unique(Y); arity_Y=length(ar);
n = length(Y);
table_dim = [arity_X arity_Y ];
p_XY = accumarray([X Y ],1,table_dim)/n;

p_X_p_Y=sum(p_XY,2) * sum(p_XY,1);

id_non_zero = intersect(find(p_XY~=0),find(p_X_p_Y~=0));
MI = sum(sum( p_XY(id_non_zero) .* log(p_XY(id_non_zero) ./  p_X_p_Y(id_non_zero))));


