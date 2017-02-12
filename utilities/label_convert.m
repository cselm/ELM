function [trainLabelNew, testLabelNew] = label_convert(trainLabel, testLabel, type)

if nargin < 3
    type = '2';
end

classes    = unique(trainLabel);
nClasses   = numel(classes);
nTrainData = numel(trainLabel);
nTestData  = numel(testLabel);

trainLabelNew = -ones(nClasses,nTrainData,'single');
testLabelNew  = -ones(nClasses,nTestData,'single');

for i = 1 : nClasses
    trainLabelNew(i,trainLabel==classes(i)) = 1;
    testLabelNew(i,testLabel==classes(i))   = 1;
end

if ~strcmp(type,'2')
    trainLabelNew = (trainLabelNew+1)/2;
    testLabelNew  = (testLabelNew+1)/2;
end


end

