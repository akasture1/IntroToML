function [percenErr] = calcError(M,wOpt,c)
% Empirically calculate the test error
% Assumes true classifier -> x2 = 1*x1 + c;
tol = 1e-4;
errors = 0;
for j = 1:M
    x = rand(2,1);
    if(x(2) - x(1) - c < 0)
        class = -1;
    else
        class = 1;
    end
    
    % because sign maps 0 -> 0
    if( class * (wOpt' * [1;x]) < tol )    
        errors = errors + 1;
%     if( sign(wOpt' * [1;x]) ~= class )
%         errors = errors + 1;
%     end
    end
end
percenErr = errors/M;
end