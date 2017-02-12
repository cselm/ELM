

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

load(fullfile('data','AR.mat'));

traindata  =  traindata./( repmat(sqrt(sum(traindata.*traindata)), [size(traindata,1),1]) );
testdata   =  testdata./(repmat(sqrt(sum(testdata.*testdata)), [size(testdata,1),1]) );

% [traindata,PS] = mapminmax(traindata,-1,1);%
% testdata = mapminmax('apply',testdata,PS);

% [traindata,PS] = mapstd(traindata);%
% testdata = mapstd('apply',testdata,PS);


[trainlabel1, testlabel1] = label_convert(trainlabel, testlabel,'2');

%-----------Setting----------------------------------------------------
alpha             = 0.1;
kclass            = length(unique(trainlabel))/2;  % for adaptive class domain selection
nn.hiddensize     = 1000;       % hidden nodes number

method            = {'ELM','RELM'};
nn.activefunction = 's';
nn.inputsize      = size(traindata,1);
nn.method         = method{2};
nn.type           = 'classification';
%-----------Initializzation-----------

nn                = elm_initialization(nn);

%--------RELM-LOO--------------------
nn.method         = method{2};
nn.C              = exp(-4:0.2:4);
[nn, acc_train]   = elm_train(traindata, trainlabel1, nn);
%[nn, acc_test]   = elm_test(testdata, testlabel, nn);

ID      = [];
IDe     = [];
lamda   = 5e-4;
tol     = 1e-2;
tic;
f = 0;
j=1;

%--------RELM-LOO---SRC--------------------
for i = 1 : size(testdata,2)
    
    [nn, acc_test]   = elm_test(testdata(:,i), [], nn);
    O = nn.testlabel;
    [Tf, id] = max(O);
    O(id) = -inf;
    [Ts, id2] = max(O);
    Tdiff = Tf-Ts;
    IDe   = [IDe, id];
    
    if Tdiff > alpha
        ID      =   [ID, id];
%         if id ~= testlabel(i)
%             fprintf('Wrong classification for %1.0f th testing sample by ELM criterion (|T_first - T_second| = %1.2f)   \n ', i, Tdiff);
%         end
    else
        f = f + 1;
        [sim, slabel] = sort(nn.testlabel, 'descend');
        newtrainlabel = trainlabel(ismember(trainlabel,slabel(1:kclass)));
        newtraindata  = traindata(:,ismember(trainlabel,slabel(1:kclass)));
        y = testdata(:,i);
        s = l1_ls(newtraindata, y, lamda, tol, 1);
        newlabel = unique(newtrainlabel);
        
%         if ~ismember(testlabel(i), newlabel)
%             fprintf('Wrong classification for %1.0f th testing sample by ELM criterion (|Adaptive class domain|) \n ', i);
%         end
        
        gap = [];
        
        for indClass  =  1 : length(newlabel)
            coef_c    =  s(newtrainlabel==newlabel(indClass));
            Dc        =  newtraindata(:,newtrainlabel==newlabel(indClass));
            gap(indClass) = norm(y-Dc*coef_c)^2;
        end
        
        wgap3  = gap ;
        index3 = find(wgap3==min(wgap3));
        id3    = index3(1);
        id     = newlabel(id3);
        
        fprintf('%1.0f / %1.0f  %1.0f   %1.3f    %1.0f    %1.2f    %1.0f \n', i, size(testdata,2), f, sum(s), find(slabel==testlabel(i)), Tdiff, id==testlabel(i));
        
        ID      =   [ID, id];
    end
end

Rec_ELM      =   sum(IDe==testlabel)/length(testlabel);
Rec_EASRC    =   sum(ID==testlabel)/length(testlabel); % recognition rate


disp(['RELM_LOO     recogniton rate is    ' num2str(Rec_ELM)]);

disp(['EASRC       recogniton rate is     ' num2str(Rec_EASRC)]);



