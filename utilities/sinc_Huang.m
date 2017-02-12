function [traindata,trainlabel,testdata,testlabel] = sinc_Huang
%Obtain Random P, T
s = 20;

X1=s*rand(1,5000)-s/2;

X1(X1 == 0) = 1e-6;

Y1 = [sin(X1).*X1.^(-1); X1];

for i=1:size(X1,2)
    Y1(1, i)=Y1(1, i)+0.4*rand(1,1)-0.2;
end
traindata  = Y1(2,:);
trainlabel = Y1(1,:);
%     fid = fopen('sinc_train','w');
%     fprintf(fid,'%2.8f %2.8f\n',Y1);
%     fclose(fid);

X2=sort(s*rand(1,5000)-s/2);

X2(X2 == 0) = 1e-6;


Y2 = [sin(X2).*X2.^(-1); X2];

testdata = Y2(2,:);
testlabel= Y2(1,:);




%     fid = fopen('sinc_test','w');
%     fprintf(fid,'%2.8f %2.8f\n',Y2);
%     fclose(fid);
