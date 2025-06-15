function [gradient] = generateScaledColors(color,scaleMagnitude,numColors,plotCheck)

[~,idx] = max(color);

% adjVector = zeros(1,3);
% adjVector(idx) = 1;

darkerColor = scaleMagnitude*color;
lighterColor = 1-(scaleMagnitude*(1-color));

gradient = [linspace(darkerColor(1),lighterColor(1),numColors)',...
    linspace(darkerColor(2),lighterColor(2),numColors)',...
    linspace(darkerColor(3),lighterColor(3),numColors)'];

if plotCheck == 1
   figure;
    b = bar(1:length(gradient),ones(1,length(gradient)),'FaceColor','flat',...
        'BarWidth',1,'EdgeAlpha',0);
    b.CData = gradient;
end
end

