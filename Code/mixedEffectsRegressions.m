% This code plots block effects for different response variables from the
% auditory signal detection task data

%% Settings
clear;
clc;
close all
 
options.rootFolder = '/Users/justin/Documents/HORGA LAB/Research/Projects/SignalDetection/CodeForGrace'; %%% SET THIS PATH TO WHERE YOU SAVE THE CodeForGrace FOLDER
addpath(genpath(options.rootFolder));

%% Load Combined dataset
dataFolder = fullfile(options.rootFolder,'Data');
load([dataFolder filesep 'Combined Samples.mat'],'SDTstructure')

%% Exclude participants with poor behavior and format data for plotting

SDTsubStructure = [];
SDTsubStructureTbl = [];
for sub = 1:length(SDTstructure)
    if SDTstructure(sub).exclusion == 0 % only use included participants
        % Build data structure with included participants
        SDTsubStructure = [SDTsubStructure;SDTstructure(sub)];

        % Rename variables for convenience
        SDTstructure(sub).behaviorStructure.mainTask.choice = SDTstructure(sub).behaviorStructure.mainTask.choiceBinary;
        SDTstructure(sub).behaviorStructure.mainTask.PC = rescale(SDTstructure(sub).behaviorStructure.mainTask.perceptualConf);
        SDTstructure(sub).behaviorStructure.mainTask.RC = rescale(SDTstructure(sub).behaviorStructure.mainTask.RCAdj);

        % Record dataset membership
        SDTstructure(sub).behaviorStructure.mainTask.dataset = repmat({SDTstructure(sub).dataset},height(SDTstructure(sub).behaviorStructure.mainTask),1);

        % Save to full table
        SDTsubStructureTbl = [SDTsubStructureTbl;SDTstructure(sub).behaviorStructure.mainTask];
    end
end

%% Generate regression plots 

% Select which type of data you would like to visualize
options.data = {'RC'}; % choice, PC, RC

datasets = {'HighCAPS_2023','HighCAPS_2024'};
datasetColors = flip(generateScaledColors([172,104,180]./256,0.5,length(datasets),0));
offset = 0.1;
signs = [-1,1];

modelspec = [options.data{:} ' ~ empSigProb + empRewProb + SNR + (empSigProb + empRewProb + SNR|id)'];
labels = {'P(Sig.)','P(Rew.)','SNR'};
titleText = ['Predicting ' options.data{:}];
labelText = {'P(Sig.)','P(Rew.)','SNR'};

% Select appropriate regression settings
switch options.data{:}
    case 'choice'
        distribution = 'Binomial';
    case {'RC','PC'}
        distribution = 'Normal';
        
end

figure('Position',[100 100,400,400]);
hold on
clear e numParticipants
for d = 1:length(datasets)

    tbl = SDTsubStructureTbl(strcmp(SDTsubStructureTbl.dataset,datasets{d}),:);
    numParticipants(d) = length(unique(tbl.id));
    mdl = fitglme(tbl,modelspec,'Distribution',distribution);

    estimates = mdl.Coefficients.Estimate(2:end);
    lowerData = estimates - mdl.Coefficients.Lower(2:end);
    upperData = mdl.Coefficients.Upper(2:end) - estimates;

    e(d) = errorbar((1:length(labels))+ (0.2*signs(d)), estimates, lowerData, upperData,...
        'o','CapSize',0,'MarkerFaceColor', datasetColors(d,:),'LineWidth', 2, 'Color', datasetColors(d,:),'LineStyle','none');

    display(['Dataset ' num2str(d) ' pValues', newline...
        'P(Sig.): ' num2str(mdl.Coefficients.pValue(2)), newline...
        'P(Rew.): ' num2str(mdl.Coefficients.pValue(3)), newline...
        'SNR: ' num2str(mdl.Coefficients.pValue(4))])

end


xlim([0.5,length(labels)+0.5])
xticks(1:length(labels))
set(gca,'xticklabel',labelText)
title(titleText)
yline(0,'k--','LineWidth',1.5)
ylabel('Fixed Effects')
set(gca,'FontName','Arial','FontSize',20,'TickDir','out','LineWidth',3,...
    'color', [238,238,238]./255);
axis square
box off

legend(e,['Exp. 1 (N = ' num2str(numParticipants(1)) ')'],...
    ['Exp. 2 (N = ' num2str(numParticipants(2)) ')'],'color',...
    [238,238,238]./255,'Location','northwest')
