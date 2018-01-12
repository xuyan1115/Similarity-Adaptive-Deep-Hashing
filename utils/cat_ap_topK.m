function [p,r] = cat_ap_topK(cateTrainTest,HammingRank, M_set)
% evaluate top-K precision
% Input:
%    cateTrainTest = relevant  train documents for each test document
%    HammingRank = rank of retrieved train documents for each test document
% Output:
%    p   = macro-averaged precision


numTest = size(cateTrainTest,2);

p = zeros(length(M_set),1);
r = zeros(length(M_set),1); 
for ix_k = 1:length(M_set)
    K = M_set(ix_k);
    precisions = zeros(1,numTest);
    recalls    = zeros(1,numTest);

    topK = HammingRank(1:K,:);

    for j = 1:numTest
        retrieved = topK(:,j);
        rel = cateTrainTest(retrieved,j);
        retrieved_relevant_num = sum(rel);
        real_releant_num = sum(cateTrainTest(:,j));
        precisions(j) = retrieved_relevant_num/K;
        recalls(j) = retrieved_relevant_num/real_releant_num;
    end

    p(ix_k) = mean(precisions);
    r(ix_k) = mean(recalls);
end

end