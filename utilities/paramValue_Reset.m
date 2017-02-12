function [opts] = paramValue_Reset(opts, paramValues)

%paramValues = {{param1,value1},{param2,value2},...,{paramN,valueN}} 

optNames = fieldnames(opts);

i = 1 ;

while i <= numel(paramValues)
    paramValueCur = paramValues{i};
    param = paramValueCur(1) ;
    value = paramValueCur(2) ;
    field = optNames{strcmpi(param, optNames)} ;
    opts.(field) = value;
    i = i + 1;
end


