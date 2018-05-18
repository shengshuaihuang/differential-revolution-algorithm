%DE Implement of Differential Evolution algorithm
%   x:      vector to be optimized
%   pSize:  population size
%   gSize:  generation size
%   xL:     low bound of x
%   xU:     up bound of x
%   papas:  struct of relevant control parameters, including crossover
%   factor cr, amplification factor f
%
%
%   more detail see 
%   Storn R, Price K. Differential evolution–a simple and efficient heuristic for 
%   global optimization over continuous spaces[J]. Journal of global optimization, 1997, 11(4): 341-359.
%
%   $Author: Hsshuai $  $Date: 2018/5/18 19:12:41 $
%  


function [xF, o] = DE(d, pSize, gSize, xL, xU, paras)
    f = paras.f;
    cr = paras.cr;
    
    xAll = zeros(d, pSize, gSize);  % all x, all population, all generation
    resultGmin = zeros(gSize, 1);   % resultGmin(g) means the minimum objective value of the g^th generation
    resultGmax = zeros(gSize, 1);   % resultGmax(g) means the maximum objective value of the g^th generation
    resultP = zeros(pSize, 1);

    for p=1:pSize
        xAll(:, p, 1) = xL + rand(d,1).*(xU - xL);
        resultP(p) = costFunc(xAll(:, p, 1));    % costFunc() should be defined in advance according to your task
    end
    
    resultGmin(1) = min(resultP);
    resultGmax(1) = max(resultP);
    for g=1:gSize-1
        randIndex = randi(pSize,4, pSize);
        xAllG = xAll(:,:,g);
        for p=1:pSize
            randIndex_ = randIndex(:,p);
            randIndex_(randIndex_==p) = randIndex(4,p);
            oldP = xAllG(:,p);
            
            %% Mutation and Crossover
            crIndex = (rand(d,1) <= cr);
            hP = oldP;
            hP(crIndex) = xAllG(crIndex, randIndex_(1)) + f*(xAllG(crIndex, randIndex_(2)) - xAllG(crIndex,randIndex_(3)));
            
            %%
            uIndex =(hP>xU);
            lIndex = (hP<xL);
            hP(uIndex) = xU(uIndex);
            hP(lIndex) = xL(lIndex);
            
            %%  Selection
            oOld = costFunc(oldP);
            oNew = costFunc(hP);
            if oOld > oNew
                xAll(:, p, g+1) = hP;
                resultP(p) = oNew;
            else
                xAll(:, p, g+1) = oldP;
                resultP(p) = oOld;
            end
        end
        [resultGmin(g+1), argIndex] = min(resultP);
        resultGmax(g+1) = max(resultP);
    end
    xF = xAll(:, argIndex, gSize);
    o = resultGmin(gSize);
    plot(resultGmin);hold on
    plot(resultGmax)
end
