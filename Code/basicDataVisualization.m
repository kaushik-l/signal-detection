% This code plots block effects for different response variables from the
% auditory signal detection task data

%% Settings
clear;
clc;
close all
 
options.rootFolder = '/Users/justin/Documents/HORGA LAB/Research/Projects/SignalDetection/CodeForGrace'; %%% SET THIS PATH TO WHERE YOU SAVE THE CodeForGrace FOLDER
addpath(genpath(options.rootFolder));

%% Load dataset
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

        % Save to full table
        SDTsubStructureTbl = [SDTsubStructureTbl;SDTstructure(sub).behaviorStructure.mainTask];
    end
end

%% Plotting responses by block contingency

% Select which type of data you would like to visualize
options.data = {'choice'}; % choice, PC, RC

figure('Position',[100 100,1000,500]);
SNRs = [0,0.25,0.5,0.75];

% Generate Colors
colorOpts(:,:,1) = flipud(generateScaledColors([222,36,59]./255,0.7,2,0)); % signal colors
colorOpts(:,:,2) = flipud(generateScaledColors([1,135,255]./255,0.7,2,0)); % reward colors

% Identify blocks with data of interest
blockIdx(1,1) = {SDTsubStructureTbl.blockSigProb <0.5 & SDTsubStructureTbl.blockRewProb == 0.5}; % LowSig
blockIdx(2,1) = {SDTsubStructureTbl.blockSigProb > 0.5 & SDTsubStructureTbl.blockRewProb == 0.5}; % HighSig
blockIdx(1,2) = {SDTsubStructureTbl.blockRewProb < 0.5 & SDTsubStructureTbl.blockSigProb == 0.5}; % LowRew
blockIdx(2,2) = {SDTsubStructureTbl.blockRewProb > 0.5 & SDTsubStructureTbl.blockSigProb == 0.5}; % HighRew

% Labels
labels = {'P(Sig.)','P(Rew.)'};

for b = 1:size(blockIdx,2) % signal & reward blocks
    
    nexttile;
    clear e;
    hold on
    for c = 1:size(blockIdx,1) % high vs low probability
        
        % Select trials from the desired block(s)
        blockTbl = SDTsubStructureTbl(blockIdx{c,b},:);
        groupData = groupsummary(blockTbl,["id","SNR"],{"mean"},options.data{:});
        [realMeans,realSems,~] = grpstats(groupData.(['mean_',options.data{:}]),...
            {groupData.SNR},{'mean','sem','gname'});
        e(c) = errorbar(SNRs,realMeans,realSems, 'o','CapSize',0,'MarkerFaceColor', colorOpts(c,:,b),'LineWidth', 2, 'Color', colorOpts(c,:,b),'LineStyle','none');

        % Fit psychometric curve
        [xfit,yfit,~,~] = psychometricModel(groupData.(['mean_',options.data{:}]),...
            groupData.SNR);
        % Plot psychometric curve
        plot(xfit,yfit,'LineWidth',2,'Color',colorOpts(c,:,b));

        xlabel('SNR')
        ylabel(options.data{:})
        xlim([min(SNRs)-0.1,max(SNRs)+0.1])
        xticks(SNRs)
        xticklabels({'No Signal','25%','50%','75%'})
    end
    
    l = legend(e,{'Low','High'}, 'color', [238,238,238]./255,'Location','southeast','LineWidth',2);
    title(l,labels{b})
end

% Format axes
h = findobj('type','axes');
for ax = 1:length(h)
    ylim(h(ax),[0,1])
    set(h(ax),'FontName','Arial','FontSize',24,'TickDir','out',...
        'LineWidth',5,'color', [238,238,238]./255,'box','off');
    axis(h(ax),'square')
end

%% Collect Demographic Info

for sub = 1:length(SDTstructure)
    age(sub,1) = SDTstructure(sub).screenerStructure.demographics.age;
    sex(sub,1) = SDTstructure(sub).screenerStructure.demographics.sex_male1_female2_other3;
    BDI(sub,1) = SDTstructure(sub).BDI;
    CAPS(sub,1) = SDTstructure(sub).CAPS;
    RAVENS(sub,1) = SDTstructure(sub).RAVENS;
    education(sub,1) = SDTstructure(sub).screenerStructure.demographics.educationNum;
end

groups = [SDTstructure.group]';

[r,p] = corr(groups,CAPS);

