function [TestPredict, post] = LDApredict(TestData, Wg, Cg, movs, varargin)
% given the parameters from LDA train, this function takes new data and provides 
% a classification using an LDA. 
% function

% inputs:   - TestData = new data to be classified (rows = features,
%                        columns = the individual data points)
%           - Wg = the multiplicative LDA weight
%           - Cg = the additive LDA weight
%           - movs = the unique numeric movement labels used
%           - varargin = parameters that can be used to calculate a
%                        posterior probability
%                               If want to calculate directly:
%                               1) Wp = multiplicative constants needed
%                               2) Cp = additive constants needed
%                               If want to fit Gaussian models:
%                               3) mu = means of of the classes
%                               4) sig = the pooled covariance matrix

%%

post = [];

Ptest = size(TestData,2);
%%-- Compute the decision functions --%%
Ate = TestData'*Wg + ones(Ptest,1)*Cg;
exAte = exp(Ate)';

errte = 0;
AAte = compet(Ate');
% errte = errte + sum(sum(abs(AAte-ind2vec(TestClass))))/2;
% nete = errte/Ptest;
% PeTest = 1-nete;

TestInd = vec2ind(AAte);
TestPredict = movs(TestInd);
TestPost = sum((AAte .* exAte),1) ./ sum(exAte,1);

percent=[];

%% calculating the posterior probability
if size(varargin,1)>0
    Wp = varargin{1};
    Cp = varargin{2};
    mu = varargin{3};
    sig = varargin{4};
    
    
    % Calculating using constants -- temp is also the posterior
    temp = exp(TestData'*Wg + ones(Ptest,1)*Cg + diag(TestData'*Wp*TestData)*ones(1,size(Cg,2)) + Cp*ones(size(TestData,2),size(Cg,2)));
    temp = temp ./ repmat(sum(temp,2),1,size(temp,2));
    
    
    for i = 1:size(temp,2)
        temp2(:,i) = mvnpdf(TestData', mu(:,i)', sig);
    end
    ind = find(sum(temp2,2)==0);

    post =  temp2 ./ repmat(sum(temp2,2),1,size(temp2,2));
    post(ind,:) = 0;
end

return;
