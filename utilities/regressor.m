function [beta, optLamda, LOO] = regressor(H, Y, lamdas)

% Y = H * beta;
% beta' * H' = Y'

if nargin == 2
    lamdas = exp(-7:1:7);
end

nData = size(H,1);

if numel(lamdas) == 1
    optLamda = lamdas; LOO = Inf;
    if nData < size(H,2)
        beta = H'*pinv(H*H'+optLamda*eye(nData))*Y;
    else
        beta = pinv(H'*H+optLamda*eye(size(H,2)))*H'*Y;
    end
else
    LOO  = inf(1,numel(lamdas));
    if nData < size(H,2)
        HH     = H*H';
        [U, S] = svd(HH);
        S      = diag(S)';
        A      = HH*U;
        B      = U'*Y;
        for iLamda = 1 : length(lamdas)
            lamdaCur   = lamdas(iLamda);
            temp    = A.*repmat(1./(S+lamdaCur),length(S),1);
            HAT     = sum(temp.*U,2);
            Y_hat   = temp*B;
            errDiff = (Y-Y_hat)./repmat((1-HAT),1,size(Y,2));
            normFro = norm(errDiff,'fro');
            errLoo  = normFro^2/nData;
            LOO(iLamda)  = errLoo;
        end
        [~,ind]  = min(LOO);
        optLamda = lamdas(ind(1));
        beta     = H'*(U.*repmat(1./(S+optLamda),length(S),1))*B;
    else
        [U, S] = svd(H'*H);
        S      = diag(S)';
        A      = H*U;
        B      = A'*Y;
        for iLamda = 1 : length(lamdas)
            lamdaCur= lamdas(iLamda);
            temp    = A.*repmat(1./(S+lamdaCur),size(A,1),1);
            HAT     = sum(temp.*A,2);
            Y_hat   = temp*B;
            errDiff = (Y-Y_hat)./repmat((1-HAT),1,size(Y,2));
            normFro = norm(errDiff,'fro');
            errLoo  = normFro^2/nData;
            LOO(iLamda)  = errLoo;
        end
        [~,ind]  = min(LOO);
        optLamda = lamdas(ind(1));
        beta     = U.*repmat(1./(S+optLamda),length(S),1)*B;
    end
    
%     if ind(1) == 1 && optLamda>1e-6
%         fprintf('You are suggested to choose a lamda smaller than %f \n',optLamda) ;
%     end
%     if ind(1) == numel(lamdas) && optLamda<1e6
%         fprintf('You are suggested to choose a lamda larger than %f \n',optLamda) ;
%     end
end

%fprintf('Optimal C is %f \n',optLamda);




