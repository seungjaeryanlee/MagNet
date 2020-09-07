%% 
clear all;clc;close all;

set(0,'DefaultAxesFontName', 'LM Roman 12')
set(0,'DefaultAxesFontSize', 12)
set(0,'DefaultAxesFontWeight', 'Bold')
% Change default text fonts.
set(0,'DefaultTextFontname', 'LM Roman 12')
set(0,'DefaultTextFontSize', 12)
set(0,'DefaultTextFontWeight', 'Bold')

alpha = 1;
%Read the data from Haoran
M1=load('D:\Dropbox (Princeton)\MagNet_TestData_hiB\data(sin_5k_hiB).mat');
data1=M1.data;
%Combining with other dataset if necessary
M2=load('D:\Dropbox (Princeton)\MagNet_TestData_hiB\data(tri_5k_hiB).mat');
data2=M2.data;
M3=load('D:\Dropbox (Princeton)\MagNet_TestData_hiB\data(trap_5k_hiB).mat');
data3=M3.data;

% M2=load('D:\Dropbox (Princeton)\MagNet_TestData_bias\data(sin_5k_bias).mat');
% data2=M2.data;

data = [data1;data2(2:length(data2(:,1)),:);data3(2:length(data3(:,1)),:)]; %%%%%%3
% data = [data1;data2(2:length(data2(:,1)),:)]; %%%%%%2
% data=data2; %%%%%%%1
% data=data1; %%%%%%%1

TotalLength=length(data(:,1));
volt=data(2:TotalLength,2);
curr=data(2:TotalLength,1); %used for NN training
loss=volt.*curr;

%% Prepare Data ===========================================================
%Data and model parameters
imscale=24;%24                         %size of the rescaled image
SampleLength=data(1,2);             %samples per waveform
SampleRate=data(1,1);               %samples per waveform
DataLength=(TotalLength-1)/SampleLength;%number of the waveform data points
TrainSize=floor(DataLength*0.7);    %70% of the data is used for training
TestSize=floor(DataLength*0.1);     %10% of the data is used for testing
ValidSize=floor(DataLength*0.2);    %20% of the data is used for validating

%Reshape the matrix
t = 0:SampleRate:SampleRate*(SampleLength-1);
currm=reshape(curr,[SampleLength,DataLength])';%Each row is one excitation;
% lossm=mean(reshape(loss,[SampleLength,DataLength]))';%Each colume is the loss for each excitation;
lossm=trapz(t,(reshape(loss,[SampleLength,DataLength])))'/SampleRate/SampleLength;%Each colume is the loss for each excitation;

TransformLength = 1000;

fb = cwtfilterbank('SignalLength',TransformLength,'VoicesPerOctave',8,'Wavelet','Morse'); % 'Morse' | 'amor' | 'bump'
for i = 1:DataLength
    cfs = abs(fb.wt(currm(i,1:TransformLength)));               %create scalogram
%     cfs = abs(fb.wt(currm(i,:)));               %create scalogram
    image = imresize(cfs,[imscale imscale]);    %resize scalogram
    scalogram(:,:,1,i)=image;                   %save scalogram
    fprintf('[%d / %d]\n',i,DataLength);
end

% N = round(rand*DataLength);figure,surf(abs(scalogram(:,:,1,N)));

%% Training ===============================================================
close all; clc;

%Everytime rearrange the data
IndexTrain = randperm(DataLength,TrainSize);
IndexValid = randperm(DataLength,ValidSize);
IndexTest  = randperm(DataLength,TestSize);
%Training Data
XTrain=scalogram(:,:,:,IndexTrain);
YTrain=lossm(IndexTrain);
%Validation Data
XValid=scalogram(:,:,:,IndexValid);
YValid=lossm(IndexValid);
%Test Data
XTest=scalogram(:,:,:,IndexTest);
YTest=lossm(IndexTest);

%Set up CNN Layers
layers = [
    imageInputLayer([imscale imscale 1],'Normalization', 'zscore')
    
    convolution2dLayer(3,32,'Padding','same')%32
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')%32
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
%     maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')%16
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
%     maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(8)%8
    tanhLayer
%     dropoutLayer(0.95)
    
    fullyConnectedLayer(1)
    regressionLayer];

miniBatchSize  = 2*2^floor(log2(DataLength/50)); %1*...
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...  %adam
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...%100
    'InitialLearnRate',1e-2, ...%1e-2
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...  %0.5
    'LearnRateDropPeriod',20, ...%20
    'Shuffle','every-epoch', ...
    'ValidationData',{XValid,YValid}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

%Run Neural Network
net = trainNetwork(XTrain,YTrain,layers,options);

%Test the Neural Network and estimate the expected error
YPred = predict(net,XTest);
% figure,plot(1:TestSize,YPred,'r-',1:TestSize,YTest,'b-');
% xlabel('Case Number #');ylabel('Measured Loss [W]');
% legend('Predicted Loss','Measured Loss');

% h= findall(groot,'Type','Figure');
% h.MenuBar = 'figure';

%% Theoretical Calculation by Steinmetz ===================================
close all; clc;
% Core Shape
N1 = 8;
N2 = 8;
A = 95.75*1e-6; % m
l = 103*1e-3;
V = 9862/1e9; % m3
f = 5; % kHz
period = 1/(f*1e3)/SampleRate;
t1 = 3200;
t2 = t1 + period ;

% Calculate B
B = zeros((length(data)-1)/SampleLength,SampleLength);
dB = zeros((length(data)-1)/SampleLength,1);
dBdt = zeros((length(data)-1)/SampleLength,SampleLength);
for i = 1:(length(data)-1)/SampleLength
    flux = 0;
    count = 0;
    Bmin = 1e10;
    Bmax = -1e10;

    for k = 1:SampleLength
        if(count<=period)
            flux = flux + data((i-1)*SampleLength+k,1)/N2/A*SampleRate;
            B(i,k) = flux;
            if (flux<Bmin) Bmin = flux; end
            if (flux>Bmax) Bmax = flux; end
            count = count+1;            
        else
            flux = flux - (Bmin+Bmax)/2;
            Bmin = 1e10;
            Bmax = -1e10;
            count = 0;
        end
    end
    dB(i) = max(B(i,3200-25:3300+25))-min(B(i,3200-25:3300+25));
    dBdt(i,:) = data((i-1)*SampleLength+1:i*SampleLength,1)'/N2/A;
end   

% Regression fitting for the parameters
K = 108;
a = 1.05;  % 1.41
b = 2.2;    % 2.3
xdata = zeros(3*(t2-t1+1),(length(data)-1)/SampleLength);
for i = 1:(length(data)-1)/SampleLength
    xdata(:,i) = [dB(i)*ones(1,t2-t1+1) dBdt(i,t1:t2) t(t1:t2)]';
end
param = lsqcurvefit(@iGSE, [K a b], xdata(:,[IndexTrain IndexValid]), lossm([IndexTrain IndexValid])');
K = param(1);
a = param(2);
b = param(3);
theta = 0:0.0001:2*pi;
m = (abs(cos(theta))).^a*2^(b-a);
ki = K / (2*pi)^(a-1) / trapz(theta,m);

% Calculate iGSE
for i = 1:(length(data)-1)/SampleLength
    loss_theo(i) = V* trapz(t(t1:t2),ki*(abs(dBdt(i,t1:t2)).^a)*dB(i).^(b-a)) /(SampleRate*period) ;
end

    %sin
%     if i<=350    f = 1;
%     else if i<=700  f = 2;
%         else  f = 5; end  end
%     Bmax(i) = max(data((i-1)*SampleLength+1:i*SampleLength,1))/(f*1000*2*pi)/N2/A;

%     loss_theo(i) = V*K*(f^a)*(Bmax(i).^b)/1000;


% figure
% plot(B(2500,:))

YTheo=loss_theo(IndexTest)';

%% Print the results ======================================================
close all; clc;
fprintf('=============Accuracy Summary=============\n')

set(0,'DefaultTextFontSize', 16)
set(0,'DefaultAxesFontSize', 16)

% Regression and Correlation
figure;
% plot(YTest,YTheo,'r*') %%
xlabel('Measured Loss [W]');ylabel('Predicted Loss [W]');
hold on; plot(YTest,YPred,'b*') 
hold on; plot([0:0.01:0.6],[0:0.01:0.6],'k--','LineWidth',1.5);
% legend('by iGSE','by MagNet');%,'Ideally, {\ity_p}={\ity_m}'); %%
axis([0 0.6 0 0.6]);
grid on;
Correlation1 = corr(YPred,YTest);
Correlation2 = corr(YTheo,YTest);%%
fprintf('Correlation = %.5f(NN) and %.5f(SE)\n',Correlation1,Correlation2);

% Relative Error
Error_re1 = abs(YPred-YTest)./abs(YTest)*100;
x = find(Error_re1>500);
Error_re1(x)=500;
Error_re2 = abs(YTheo-YTest)./abs(YTest)*100;
x = find(Error_re2>500);
Error_re2(x)=500;
figure;
% plot(YTest,Error_re2,'r*'); %%
hold on;plot(YTest,Error_re1,'b*');
xlabel('Measured Loss [W]');ylabel('Relative Error [%]');
% legend('by iGSE','by MagNet'); %%
axis([-inf inf -10 100]);
grid on;
Error_re_max1 = max(Error_re1);
Error_re_avg1 = mean(Error_re1);
Error_re_max2 = max(Error_re2);
Error_re_avg2 = mean(Error_re2);
fprintf('Relative Error (NN): Max = %.4f%%, Avg = %.4f%%\n',Error_re_max1,Error_re_avg1);
% fprintf('Relative Error (SE): Max = %.4f%%, Avg = %.4f%%\n',Error_re_max2,Error_re_avg2); %%

% % Error Distribution
% figure,histogram(Error_re1,0:5:100,'FaceColor','b','Normalization','probability'); grid on;
% xlabel('Relative Error [%]');
% ylabel('Percentage Ratio [%]');
% title('Error Distribution of Testing Points');
% ytix = get(gca, 'YTick');
% set(gca, 'YTick',ytix, 'YTickLabel',ytix*100)

% % Absolute Error
% Error_ab1 = YPred-YTest;
% Error_ab2 = YTheo-YTest;
% figure;
% % plot(YTest,Error_ab2,'r*'); %%
% hold on;plot(YTheo,Error_ab1,'b*');
% xlabel('Measured Loss [W]');ylabel('Absolute Error [W]');
% % legend('by iGSE','by MagNet'); %%
% grid on;
% Error_ab_max1 = max(abs(Error_ab1));
% Error_ab_avg1 = mean(abs(Error_ab1));
% Error_ab_max2 = max(abs(Error_ab2));
% Error_ab_avg2 = mean(abs(Error_ab2));
% fprintf('Absolute Error (NN): Max = %.4f, Avg = %.4f\n',Error_ab_max1,Error_ab_avg1);
% % fprintf('Absolute Error (SE): Max = %.4f, Avg = %.4f\n',Error_ab_max2,Error_ab_avg2); %%

% Mean-squared Error and R-square
MSE1 = sum((YPred-YTest).^2) / length(YTest);
MSE2 = sum((YTheo-YTest).^2) / length(YTest);
Rsquare1 = 1 - MSE1 / var(YTest);
Rsquare2 = 1 - MSE2 / var(YTheo);
fprintf('R_square = %.5f(NN) and %.5f(SE)\n',Rsquare1,Rsquare2);
% % 
%%
% close all;
% I1 = deepDreamImage(net,2,1:16,'Verbose',false);figure;I1 = imtile(I1);imshow(I1);
% I2 = deepDreamImage(net,6,1:16,'Verbose',false);figure;I2 = imtile(I2);imshow(I2);
% I3 = deepDreamImage(net,10,1:16,'Verbose',false);figure;I3 = imtile(I3);imshow(I3);%figure;pcolor(I3);
% I4 = deepDreamImage(net,14,1,'Verbose',false);figure;I4 = imtile(I4);imshow(I4);
% 
% % h= findall(groot,'Type','Figure');
% % h.MenuBar = 'figure';
