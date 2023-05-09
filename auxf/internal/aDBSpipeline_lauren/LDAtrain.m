function [Wg,Cg, movs, V, Wpost, Cpost, Mi, C, W, T] = LDAtrain(TrainData,TrainClass)
% this function takes trains an LDA classifier. Based off englehart's
% function

% inputs: - TrainData = features of data. Rows = the features, Columns = the
%                     # of exemplars
%         - TrainClass = the training data labels for each exemplar as a
%                        row vector. labels must be numeric, but do not
%                        need to be consecutive

% outputs:  - Wg = the multiplicative LDA weights
%           - Cg = the additive LDA weights
%           - movs = a list of the unique values used as class labels
%           - Wpost = the multiplicative weights used to calculate the
%                     posterior probability
%           - Cpost = the additive weights used to calculate the
%                     posterior probability
%           - Mi = the mean for each class (rows = the features, columns
%                  are the different classes)
%           - C = the pooled covariance matrix

%%
N = size(TrainData,1); % N = number of features, # of rows
Ptrain = size(TrainData,2);
% Ptest = size(TestData,2);

sc = repmat(std(TrainData')',1,size(TrainData,2));
TrainData =  TrainData + sc./10000.*randn(size(TrainData)); 

movs = unique(TrainClass);
K = size(movs,2);

%%-- Compute the means and the pooled covariance matrix --%%
C = zeros(N,N);
totMean = mean(TrainData')';
Sb = zeros(length(totMean),length(totMean));
for l = 1:K;
	idx = find(TrainClass==movs(l));
	Mi(:,l) = mean(TrainData(:,idx)')';
    Sb = Sb+((Mi(:,l)-totMean)*(Mi(:,l)-totMean)')*size(idx,2);
	 C = C + cov((TrainData(:,idx)-Mi(:,l)*ones(1,length(idx)))');
end

C = C./K;
Pphi = 1/K; 

Cinv = inv(C);


% Sb = between class scatter
% C = within class scatter / # of classes
[V D] = eig(Cinv.*K*Sb);

%%-- Compute the LDA weights --%%
for i = 1:K
	Wg(:,i) = Cinv*Mi(:,i);
	Cg(:,i) = -1/2*Mi(:,i)'*Cinv*Mi(:,i) + log(Pphi)';
end

Wpost = -1/2*Cinv;
Cpost = -N/2*log(2*pi) - 1*2*log(det(C));


%calculating LD weights and thresholds -- greater than this threshold
%predicts class 2, less than this threshold predicts class 1
%%% needs to be updated if want to do a multi-class problem
W = diff(Wg');
T = -diff(Cg);

