function [perf1,perf2,xFit,yFit] = corr_fun(p1,p2)

    perf1 = p1;
    perf2 = p2;
    perf1 = perf1(~isnan(p1) & ~isnan(p2));
    perf2 = perf2(~isnan(p1) & ~isnan(p2));
    
    if any(~isnan(perf1)) && any(~isnan(perf2))
        coefficients = polyfit(perf1,perf2, 1);
        xFit = linspace(floor(min([perf1;perf2])), ceil(max([perf1;perf2])), 1000);
        yFit = polyval(coefficients , xFit);
    end

end

