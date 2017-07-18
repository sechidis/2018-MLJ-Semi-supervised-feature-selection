function [cmb] = IAMB( X_data, Y_labels, alpha)
% Summary 
%    IAMB algorithm for feature selection  
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_labels: n x 1 vector with the labels
%    alpha: Significance level for the tests

numf = size(X_data,2);
N = size(X_data,1);

cmb = [];

features=1:numf; 

finished = false;
while ~finished
    cmbVector = joint(X_data(:,cmb));
    association=[];dof=[];pvalue=[];
    for fIndex = 1:length(features)
        association(fIndex) = 2*N*cmi(X_data(:,features(fIndex)), Y_labels, cmbVector);
        dof(fIndex) = max(1,(length(unique(Y_labels))-1) *(length(unique(X_data(:,features(fIndex))))-1) * max(1,length(unique(cmbVector))));       
        pvalue(fIndex) = chi2cdf(association(fIndex),dof(fIndex),'upper');        
    end
    
    [values fidx] = sortrows([pvalue' association'],[1 -2]);

    minPvalue = values(1,1);
    minidx = fidx(1);
    if  minPvalue> alpha
        finished = true;
    else        
        cmb = [ cmb features(minidx) ];
        features(minidx) =[];
    end
end

finished = false;
while ~finished && ~isempty(cmb)
    association = [];
    dof =[];
    Pvalue=[];
    for n = 1:length(cmb)
        cmbwithoutn = cmb;
        cmbwithoutn(n)=[];
        cmbwithoutnVector = joint(X_data(:,cmbwithoutn));
        association(n) = 2*N*cmi( X_data(:,cmb(n)), Y_labels, cmbwithoutnVector);
        dof(n)=   max(1,(length(unique(Y_labels))-1) *(length(unique(X_data(:,cmb(n))))-1) * max(1,length(unique(cmbwithoutnVector))));        
        Pvalue(n) = 1-chi2cdf(association(n),dof(n));
    end
    [values fidx] = sortrows([Pvalue' association'],[-1 2]);
    maxval = values(1,1);
    maxidx = fidx(1);
    
    if maxval < alpha
        finished = true;
    else
        cmb(maxidx) = [];
    end
end





