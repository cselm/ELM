

% =========================================================================
%%%    Authors:    Kai Zhang AND DR Jiuwen Cao
%%%    Hangzhou Dianzi University, CHINA
%%%    EMAIL:      jwcao@hdu.edu.cn
%%%    DATE:       June 2016
%%% Please cite the paper: "J. Cao, K. Zhang, M. Luo, C. Yin and X. Lai, Extreme learning machine and adaptive sparse representation for image classification, Neural networks (2016), http://dx.doi.org/10.1016/j.neunet.2016.06.001"
%--------------------------------------------------------------------------


clear;clc;
addpath(genpath('./.'));
rng('default');
%%%%%%%%%-----Load data--------------------------------------------------

load(fullfile('data','usps.mat'));

nn.label = trainlabel;
[trainlabel, testlabel] = label_convert(trainlabel, testlabel,'2');
%-----------Setting----------------------------------------------------
method            = {'ELM','RELM'};


nn.hiddensize     = 3000;
nn.activefunction = 's';
nn.inputsize      = size(traindata,1);
nn.method         = method{2};
nn.type           = 'classification';
%-----------Initializzation-----------

nn                = elm_initialization(nn);
fprintf('      method      |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');

%--------ELM--------------------
nn.method         = method{1};
[nn, acc_train]   = elm_train(traindata, trainlabel, nn);
[nn1, acc_test]    = elm_test(testdata, testlabel, nn);

fprintf('      %6s      |     %0.5f     |      %.3f      |      %.5f      |      %.5f      \n',nn.method,nn.C_opt,acc_train,acc_test,nn.time_train);

%--------RELM-LOO--------------------
nn.method         = method{2};
nn.C              = exp(-4:0.2:6);
[nn, acc_train]   = elm_train(traindata, trainlabel, nn);
[nn2, acc_test]    = elm_test(testdata, testlabel, nn);

fprintf('      %6s      |     %2.5f    |      %.3f      |      %.5f      |      %.5f      \n',nn.method,nn.C_opt,acc_train,acc_test,nn.time_train);






