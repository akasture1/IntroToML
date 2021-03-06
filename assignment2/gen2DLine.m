function [x1,x2]=gen2DLine(w)
    
    x1=0:0.001:2.5;
    
    if(length(w)==1)
        x2=zeros(1,length(x1));
    
    elseif(length(w)==2)
        x2=(-x1.*w(1))./w(end);
    
    elseif(length(w)==3)
        x2=(-x1.^2.*w(1) -x1.*w(2))./w(end);
    
    elseif(length(w)==4)
        x2=(-x1.^3.*w(1) -x1.^2.*w(2) - x1.*w(3))./w(end);
    
    elseif(length(w)==5)
        x2=(-x1.^4.*w(1) -x1.^3.*w(2) -x1.^2.*w(3) -x1.*w(4))./w(end);
    end
end