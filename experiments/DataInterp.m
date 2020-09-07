close all; clear all; clc;

N = 4; %How many curves
B = [25 50 100 200]; %Labels for B (mT)

%==========================================================================
data = cell(N,1);
fmin = inf;
fmax = -inf;

for i = 1:N    % Load and save original data
    FileName = ['D:\Dropbox (Princeton)\Raw Data_hiB\dummy\',num2str(B(i)),'mT.txt'];
    temp = load(FileName); % Input Format: [f Pv]
    
    fmin = min([fmin,min(temp(:,1))]);
    fmax = max([fmax,max(temp(:,1))]);
    
    temp = [temp(:,2), ones(size(temp(:,1)))*B(i)*1e-3, temp(:,1)]; % Output Format: [Pv B f]
    temp = sortrows(temp,[2,3]); % Sort Pv as B then f
    data{i} = temp;
end

%==========================================================================
f = 10.^(linspace(log10(fmin),log10(fmax),100));
data2 = [];

for i = 1:N    % Interp over freq
    temp = data{i}; % Input Format: [f Pv]
    
    ftemp = f(f<=max(temp(:,3)) & f>=min(temp(:,3)));
    Ptemp = interp1(temp(:,3),temp(:,1),ftemp,'spline');
    
    temp = [Ptemp', ones(size(Ptemp'))*B(i)*1e-3, ftemp']; % Output Format: [Pv B f]
    data2 = [data2;temp];
end

%==========================================================================
% Bscale = 10.^(linspace(log10(min(B)*1e-3),log10(max(B)*1e-3),100));
Bscale = linspace((min(B)*1e-3),(max(B)*1e-3),100);
data3 = [];

for i = 1:length(f)    % Interp over flux
    temp = data2(data2(:,3)==f(i),:);
    Bmax = max(temp(:,2));
    Bmin = min(temp(:,2));
    
%     Btemp = Bscale(Bscale<=Bmax & Bscale>=Bmin);
    Btemp = Bscale(Bscale>=Bmin);
%     Btemp = Bscale;

    sz = size(temp);
    if (sz(1)==1)
        data3 = [data3;temp];
    else
        poly = polyfit(log10(temp(:,2)),log10(temp(:,1)),1);
        Ptemp = 10.^polyval(poly,log10(Btemp));
        temp = [Ptemp', Btemp', ones(size(Btemp'))*f(i)]; % Output Format: [Pv B f]
        data3 = [data3;temp];
    end
end

%==========================================================================
DUMMY = data3; % Interpolation finished. Output Format: [Pv B f]
% writematrix(DUMMY,'DUMMY.csv');
%==========================================================================

% Generate Sinusoidal Voltage Waveforms

f = DUMMY(:,3)*1e3;
Bmax = DUMMY(:,2);

N1 = 8;
N2 = 8;
A = 95.75*1e-6; % m
l = 103*1e-3;
V = 9862/1e9; % m3

Vmax = 2*pi*f*N2*A.*Bmax;

DataLength = length(f);
% SampleRate = 1/(fmax*1e3)/100;  %100 = number of sample per period for fmax
% SampleLength = round(1/(fmin*1e3)/SampleRate*1);   %1 = number of total period for fmin

SampleRate = 2e-8;
SampleLength = 2000;

t = 0:SampleRate:(SampleLength-1)*SampleRate;
VOLT = zeros(DataLength,SampleLength);
for i = 1:length(f)
    VOLT(i,:) = Vmax(i)*sin(2*pi*f(i)*t);
end

% writematrix(VOLT,'VOLT.csv');

LOSS = DUMMY(:,1);
% LOSS = DUMMY(:,1)*V/1000;
% writematrix(LOSS,'LOSS.csv');

scatter3(DUMMY(:,3),DUMMY(:,2),log10(DUMMY(:,1)));