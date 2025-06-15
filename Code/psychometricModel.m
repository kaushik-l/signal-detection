function [xfit,yfit,fitresult,labels] = psychometricModel(Choice,Evidence)
%% fits a psychometric curve
% inputs (required)
% - data: a table containing a vector evidence and a vector with choices
% inputs (optional)
% 'genModel': specifies the generative model that is used for fitting
%             'gaussian' (default) - 2-parameter Gaussian cdf (mu, sigma)
%             'gaussianZerobias' - 1-parameter Gaussian cdf (sigma)
%             'guessLapseGaussian' - 4-parameter Gaussian cdf (mu, sigma)
%             with assymmetric lapse rates (lamda, gamma)
%             'symmetricLapseGaussian' - 3-parameter Gaussian cdf (mu,
%             sigma) with symmetric lapse rates (lamda) 'guessGaussian' -
%             3-parameter Gaussian cdf (mu, sigma) with guess rate only
%             (gamma). This is commonly used for detection tasks with a
%             baseline guess rate.
% outputs
% fitresult - Matlab fit object with fitted parameters etc.
% gof       - goodness of fit
% ft        - formula used for fit

%% prepare data
x = Evidence;
y = Choice;

%% psychometric fit

% Asymmetric Lapse Gaussian
ft = fittype( 'gamma+(1-gamma-lamda).*(normcdf(x,m,sigma))','independent', 'x', 'dependent', 'y' );
opts = fitoptions( ft );
opts.Display = 'Off';
opts.Lower = [0 0 1*min(x),.01];
opts.StartPoint = [0.25,0.25,mean(x),1*range(x)./2];
opts.Upper = [0.5,0.5,1*max(x),3*range(x)];
[fitresult,gof] = fit( x(:), double(y(:)), ft, opts );

fitBinEdges=[prctile(x,[0:1:100]) max(x)+eps];
xfit=linspace(min(fitBinEdges)-.1*range(fitBinEdges),max(fitBinEdges)+.1*range(fitBinEdges),1E5);
yfit=feval(fitresult,xfit);
labels = {'\gamma','\lambda','\mu','\sigma'};


% % Gaussian
% ft = fittype( 'normcdf(x,m,sigma)', 'independent', 'x', 'dependent', 'y');
% opts = fitoptions( ft );
% opts.Display = 'Off';
% opts.Lower = [1*min(x),.01];
% opts.StartPoint = [mean(x),1*range(x)./2];
% opts.Upper = [1*max(x),3*range(x)];
% [fitresult,gof] = fit( x(:), double(y(:)), ft, opts );
% labels = {'\mu','\sigma'};
% 
% fitBinEdges=[prctile(x,[0:1:100]) max(x)+eps];
% xfit=linspace(min(fitBinEdges)-.1*range(fitBinEdges),max(fitBinEdges)+.1*range(fitBinEdges),1E5);
% yfit=feval(fitresult,xfit);

end








