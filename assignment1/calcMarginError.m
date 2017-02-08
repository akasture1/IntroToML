function [percenErr] = calcMarginError(M,wOpt,c,gamma)
% Empirically calculate the test error
% Assumes true classifier1 -> x2 = 1*x1 + c + gamma;
%         true classifier2 -> x2 = 1*x1 + c - gamma;

tol = 1e-4;
errors = 0;
j=0;
while j ~= M
    x = rand(2,1);
    if(x(2) - x(1) - c > gamma)
        class = 1;
        j = j + 1;
    elseif(x(2) - x(1) - c < -gamma)
        class = -1;
        j = j + 1;
    else
        continue
    end
    
    % because sign maps 0 -> 0
    if( class * (wOpt' * [1;x]) < tol )    
        errors = errors + 1;
    end
    
%     if( sign(wOpt' * [1;x]) ~= class )
%         errors = errors + 1;
%     end

end
percenErr = errors/M;
end
