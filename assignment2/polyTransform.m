function Z = polyTransform(X, q)

    % Polynomials (degree q)
    q0=@(x1,x2,c)[c x2];
    q1=@(x1,x2,c)[x1 c x2];
    q2=@(x1,x2,c)[x1.^2 x1 c x2];
    q3=@(x1,x2,c)[x1.^3 x1.^2 x1 c x2];
    q4=@(x1,x2,c)[x1.^4 x1.^3 x1.^2 x1 c x2];
    
    x1 = X(:,1);
    x2 = X(:,2);
    d = length(x1);
    % Transform!
    if(q==0)
        Z = q0(x1,x2,ones(d,1))';
    elseif(q==1)
        Z = q1(x1,x2,ones(d,1))';
    elseif(q==2)
        Z = q2(x1,x2,ones(d,1))';
    elseif(q==3)
        Z = q3(x1,x2,ones(d,1))';
    elseif(q==4)
        Z = q4(x1,x2,ones(d,1))';
    end
end