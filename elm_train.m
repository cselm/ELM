
% =========================================================================
%%%    Authors:    Kai Zhang AND DR Jiuwen Cao
%%%    Hangzhou Dianzi University, CHINA
%%%    EMAIL:      jwcao@hdu.edu.cn
%%%    DATE:       June 2016
%%% Please cite the paper: "J. Cao, K. Zhang, M. Luo, C. Yin and X. Lai, Extreme learning machine and adaptive sparse representation for image classification, Neural networks (2016), http://dx.doi.org/10.1016/j.neunet.2016.06.001"
%--------------------------------------------------------------------------

function [nn, acc_train] = elm_train(X, Y, nn)

% beta f(Wx+b) = y

tic;

ndata = size(X,2);
tempH = nn.W*X + repmat(nn.b,1,ndata);

switch lower(nn.activefunction)
    case{'s','sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case{'t','tanh'}
        H = tanh(tempH);
end

clear tempH;

switch(nn.method)
    case 'ELM'
        [beta,C_opt,LOO] = regressor(H', Y', 0);
    case 'RELM'
        [beta,C_opt,LOO] = regressor(H', Y', nn.C);
end

nn.time_train = toc;
nn.C_opt = C_opt;
nn.LOO   = LOO;

nn.beta  = beta';
Y_hat    = nn.beta*H;


if ismember(nn.type,{'c','classification','Classification'})
    [~,label_actual]  = max(Y_hat,[],1);
    [~,label_desired] = max(Y,[],1);
    acc_train = sum(label_actual==label_desired)/ndata;
else
    normfro   = norm(Y-Y_hat,'fro');
    acc_train = sqrt(normfro^2/ndata);
end


nn.trainlabel  = Y_hat;
nn.acc_train   = acc_train;



