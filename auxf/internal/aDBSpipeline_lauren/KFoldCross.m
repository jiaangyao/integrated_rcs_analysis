function [ca, cm, cmCell, AUC, AUCstd] = KFoldCross(data, lab, K)
% data (organized by cell array for same label) 
% lab = vector of labels for each cell in data
% K = number for cross validation

for i = 1:length(lab)
   idx = randperm(size(data{i},1));
   N = floor(size(data{i},1)/K);
   rem = size(data{i},1) - N*K;
   for k = 1:K
       if k <= rem
           dataK{k}{i} = data{i}(idx((k-1)*N+(1:(N+1))),:);
       else
           dataK{k}{i} = data{i}(idx((k-1)*N+(1:(N))),:);
       end
       labK{k}{i} = repmat(lab(i),size(dataK{k}{i},1),1);
   end
   
end

indCa = zeros(K,1);
indCm = zeros(length(unique(lab)), length(unique(lab)), K);
for k = 1:K
    datTest = cell2mat(dataK{k}');
    datTrain = cell2mat(cellfun(@(x)(cell2mat(x')), dataK(setdiff(1:K,k)), 'uniformoutput', 0)');
    labTest = cell2mat(labK{k}');
    labTrain = cell2mat(cellfun(@(x)(cell2mat(x')), labK(setdiff(1:K,k)), 'uniformoutput', 0)');

    [Wg,Cg, cl, ~, ~, ~, ~, ~, W, T] = LDAtrain(datTrain',labTrain');  
    
    [labPred] = LDApredict(datTest', Wg, Cg, cl)';
    indCa(k) = sum(labPred == labTest)/length(labPred);
    indCm(:,:,k) = confusionmat(labTest, labPred);
    
    for i = 1:length(lab)
       labPredCell{i} =  LDApredict(dataK{k}{i}', Wg, Cg, cl)';
       indCmCell{i}(:,:,k) = ConfMat(repmat(lab(i),length(labPredCell{i}),1), labPredCell{i}, unique(lab));
    end
    
    scores = W*datTest';
    [~,~,~,AUCInd(k)] = perfcurve(labTest, scores', 2);
end
ca = mean(indCa);
cm = squeeze(mean(indCm,3));
cmCell = cellfun(@(x)(squeeze(mean(x,3))), indCmCell, 'uniformoutput', 0);
AUC = mean(AUCInd);
AUCstd = std(AUCInd);

function cm = ConfMat(group, grouphat, lab)

cm = zeros(length(lab), length(lab));

for i = 1:length(lab)
    boolTrueLab = group == lab(i);
    cm(i,:)=cell2mat(arrayfun(@(x)(sum(grouphat(boolTrueLab) == x)), lab, 'uniformoutput', 0));

end

