function CMI= cmi(  X, Y, Z)
% Summary 
%    Estimate conditional mutual information I(X;Y|Z) between categorical variables X,Y,Z 
%    X,Y,Z can be matrices which are converted into a joint variable before computation

epsilon=10^(-50);

if length(Z)==0 || length(unique(Z))==1
    CMI = mi(X,Y);
    return
end

[~,~,X] = unique(X,'rows');
[~,~,Y] = unique(Y,'rows');
[~,~,Z] = unique(Z,'rows');

arity_x = length(unique(X));arity_y = length(unique(Y));arity_z = length(unique(Z));

n = length(Y);
table_dim = [arity_x arity_y arity_z];
k = prod(table_dim);
%%% Find probabilities
p_xyz = accumarray([X Y Z],1,table_dim)/n;
p_z = squeeze(sum(sum(p_xyz,1),2));
p_xz = squeeze(sum(p_xyz,2));
p_yz = squeeze(sum(p_xyz,1));

CMI=0;
for index_z = 1:arity_z
    CMI =CMI+ sum(sum( squeeze(p_xyz(:,:,index_z)) .* log(epsilon + squeeze(p_xyz(:,:,index_z))*p_z(index_z) ./ (epsilon +  p_xz(:,index_z)* p_yz(:,index_z)' ) )));
end

