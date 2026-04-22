%% =========================================================
%  RUN_PROJ_FP_SUBSPACE_SNR6_PROJENERGY_PERCLASS_SAVEACC_MC.m
%
%  Projection energy ratio (FAST):
%   A) per-harmonic rho_h, aggregate by median (robust)
%   B) per-harmonic block normalization (each 4-d block)
%   C') projection energy ratio:
%        rho_h = ||PB*y||^2 / (||PB*y||^2 + ||PI*y||^2)
%
%  Noise grid:
%   - complex AWGN by SNR levels: [50 45 40 35 30 25] dB
%   - total 6 points; each point MC=800
%
%  NEW OUTPUT (per-class):
%   - ACC_overall_mean/std (6x1), ACC_overall_mc{6} (800x1)
%   - ACC_C1_mean/std      (6x1), ACC_C1_mc{6}
%   - ACC_C2_mean/std      (6x1), ACC_C2_mc{6}
%   - ACC_C3_mean/std      (6x1), ACC_C3_mc{6}
%   - plus tL,tH,rho_clean,sIV, dict paths, etc.
%
%  (Also writes a CSV summary table: SNR, Acc(C1),Acc(C2),Acc(C3),Overall)
%% =========================================================
clc; clear; close all;

%% ===================== USER PATHS =====================
dataDir = 'E:\桌面\乔方昱课题论文\研究内容1\实验3\数据\数据';

% dict dir = script folder by default
dictDir = fileparts(mfilename('fullpath'));

dictIntFile = fullfile(dictDir,'DICT_Aint_region_(8_9_10_11_12_13_14_15)_L7_R16.mat');
dictBndFile = fullfile(dictDir,'DICT_Abnd_pair_L7_R16.mat');

assert(exist(dataDir,'dir')==7, 'dataDir not found: %s', dataDir);
assert(exist(dictIntFile,'file')==2, 'dictIntFile not found: %s', dictIntFile);
assert(exist(dictBndFile,'file')==2, 'dictBndFile not found: %s', dictBndFile);

%% ===================== SNR GRID =====================
SNRdB_list = [50 45 40 35 30 25];
MC = 800;

%% ===================== CONSTANTS / FP FORMAT =====================
% FP16: 4 harmonic blocks, each length 4:
% [I_L I_R V_L V_R] for h=3,5,7,9
Hblk = 4;
blkLen = 4;
d = Hblk*blkLen; % 16
assert(d==16);

% indices for I/V within each block
idxI = []; idxV = [];
for b=1:Hblk
    base = (b-1)*blkLen;
    idxI = [idxI, base+1, base+2]; %#ok<AGROW>
    idxV = [idxV, base+3, base+4]; %#ok<AGROW>
end
EPSN = 1e-12;

%% ===================== LOAD DICTS =====================
DB = load_dict_matrix(dictBndFile, [2 16], 'DB');  DB = DB.';  % ->16x2
DI = load_dict_matrix(dictIntFile, [8 16], 'DI');  DI = DI.';  % ->16x8
fprintf('[DICT] DB=%dx%d, DI=%dx%d\n', size(DB,1), size(DB,2), size(DI,1), size(DI,2));

%% ===================== LOAD DATA CASES =====================
L = dir(fullfile(dataDir, '*.mat'));
assert(~isempty(L), 'No case MAT files found in this folder.');

Y = {}; cls = [];
for i=1:numel(L)
    S = load(fullfile(dataDir, L(i).name));

    c = pick_field_first(S, {'cls','class','label','ycls','Class'});
    y = pick_field_first(S, {'fp16','FP16','y','y_fp','F','Fin','fin','fingerprint','fp'});

    if isempty(c) || isempty(y), continue; end
    c = double(c);
    y = y(:);

    if ~isfinite(c) || ~ismember(c,[1 2 3]), continue; end
    if numel(y) ~= 16, continue; end
    if ~isnumeric(y), continue; end

    Y{end+1,1} = complex(y); %#ok<AGROW>
    cls(end+1,1) = c; %#ok<AGROW>
end

N = numel(Y);
assert(N>0, 'No valid cases loaded (all skipped). Check fp16/y field and cls field.');

nC1 = sum(cls==1);
nC2 = sum(cls==2);
nC3 = sum(cls==3);

fprintf('[DATA] Loaded %d cases. C1=%d C2=%d C3=%d\n', N, nC1, nC2, nC3);

% pack to matrix (faster) : 16 x N
Ymat = complex(zeros(16,N));
for i=1:N
    Ymat(:,i) = Y{i};
end
clear Y;

%% ===================== WHITEN I/V (estimate from clean data) =====================
tmpRat = zeros(min(N,50),1);
for i=1:min(N,50)
    y0 = Ymat(:,i);
    tmpRat(i) = norm(y0(idxI)) / max(norm(y0(idxV)), EPSN);
end
sIV = median(tmpRat);
fprintf('[WHITEN] median ||I||/||V|| = %.6f -> scale V by %.6f\n', sIV, sIV);

Swh = ones(16,1);
Swh(idxV) = sIV;

DBw = diag(Swh) * DB;
DIw = diag(Swh) * DI;

%% ===================== PRECOMPUTE PER-HARMONIC PROJECTORS =====================
rowsH = cell(Hblk,1);
PBh = cell(Hblk,1);
PIh = cell(Hblk,1);

for h=1:Hblk
    rows = (h-1)*blkLen + (1:blkLen);
    rowsH{h} = rows;

    DBh = DBw(rows,:);     % 4x2
    DIh = DIw(rows,:);     % 4x8

    [QB, ~] = qr(DBh, 0);  % 4x2
    [QI, ~] = qr(DIh, 0);  % 4x4 (rank<=4)

    PBh{h} = QB * QB';
    PIh{h} = QI * QI';
end

%% ===================== LEARN THRESHOLDS ON CLEAN =====================
rho_clean = zeros(N,1);
for i=1:N
    y = Ymat(:,i);
    y(idxV) = y(idxV) * sIV;                 % whitening
    y = block_norm_4x4(y, blkLen, EPSN);     % block norm
    rho_clean(i) = rho_median_proj_energy(y, PBh, PIh, rowsH, EPSN);
end

[tL, tH, acc0] = learn_tL_tH(rho_clean, cls);
fprintf('[THR] learned tL=%.3f, tH=%.3f  clean overall acc=%.3f\n', tL, tH, acc0);

%% ===================== RUN SNR x MC (PER-CLASS) =====================
nS = numel(SNRdB_list);

ACC_overall_mean = zeros(nS,1);
ACC_overall_std  = zeros(nS,1);
ACC_overall_mc   = cell(nS,1);

ACC_C1_mean = zeros(nS,1);  ACC_C1_std = zeros(nS,1);  ACC_C1_mc = cell(nS,1);
ACC_C2_mean = zeros(nS,1);  ACC_C2_std = zeros(nS,1);  ACC_C2_mc = cell(nS,1);
ACC_C3_mean = zeros(nS,1);  ACC_C3_std = zeros(nS,1);  ACC_C3_mc = cell(nS,1);

rng(20260120,'twister');
tStart = tic;

for is = 1:nS
    snrdb = SNRdB_list(is);

    acc_over_mc = zeros(MC,1);
    acc1_mc = zeros(MC,1);
    acc2_mc = zeros(MC,1);
    acc3_mc = zeros(MC,1);

    for mci = 1:MC
        % ---- add AWGN to ALL cases (vectorized) ----
        Ynoisy = add_awgn_snr_colwise(Ymat, snrdb);

        correct_all = 0;
        correct1 = 0; correct2 = 0; correct3 = 0;

        for i=1:N
            yM = Ynoisy(:,i);

            % whitening on V
            yM(idxV) = yM(idxV) * sIV;

            % block normalization
            yM = block_norm_4x4(yM, blkLen, EPSN);

            rho = rho_median_proj_energy(yM, PBh, PIh, rowsH, EPSN);
            pred = predict_from_rho(rho, tL, tH);

            if pred == cls(i)
                correct_all = correct_all + 1;
                if cls(i)==1, correct1 = correct1 + 1; end
                if cls(i)==2, correct2 = correct2 + 1; end
                if cls(i)==3, correct3 = correct3 + 1; end
            end
        end

        acc_over_mc(mci) = correct_all / N;

        % per-class acc (avoid div-by-zero)
        acc1_mc(mci) = correct1 / max(nC1,1);
        acc2_mc(mci) = correct2 / max(nC2,1);
        acc3_mc(mci) = correct3 / max(nC3,1);
    end

    ACC_overall_mean(is) = mean(acc_over_mc);
    ACC_overall_std(is)  = std(acc_over_mc);
    ACC_overall_mc{is}   = acc_over_mc;

    ACC_C1_mean(is) = mean(acc1_mc);  ACC_C1_std(is) = std(acc1_mc);  ACC_C1_mc{is} = acc1_mc;
    ACC_C2_mean(is) = mean(acc2_mc);  ACC_C2_std(is) = std(acc2_mc);  ACC_C2_mc{is} = acc2_mc;
    ACC_C3_mean(is) = mean(acc3_mc);  ACC_C3_std(is) = std(acc3_mc);  ACC_C3_mc{is} = acc3_mc;

    fprintf('[SNR=%2.0f dB] Overall=%.3f±%.3f | C1=%.3f±%.3f C2=%.3f±%.3f C3=%.3f±%.3f (%.1fs)\n', ...
        snrdb, ACC_overall_mean(is), ACC_overall_std(is), ...
        ACC_C1_mean(is), ACC_C1_std(is), ...
        ACC_C2_mean(is), ACC_C2_std(is), ...
        ACC_C3_mean(is), ACC_C3_std(is), toc(tStart));
end

%% ===================== SAVE OUTPUT (NEW NAME) =====================
% <<< NEW output name prefix >>>
outBase = sprintf('ACC_PROJENERGY_PERCLASS_SAVEACC_MC_SNR_%s_MC%d_%s', ...
    strrep(num2str(SNRdB_list),'  ','_'), MC, datestr(now,'yyyymmdd_HHMMSS'));

outFile = fullfile(dataDir, [outBase '.mat']);

save(outFile, ...
    'SNRdB_list','MC', ...
    'ACC_overall_mean','ACC_overall_std','ACC_overall_mc', ...
    'ACC_C1_mean','ACC_C1_std','ACC_C1_mc', ...
    'ACC_C2_mean','ACC_C2_std','ACC_C2_mc', ...
    'ACC_C3_mean','ACC_C3_std','ACC_C3_mc', ...
    'tL','tH','rho_clean','sIV', ...
    'dictBndFile','dictIntFile','dataDir', ...
    'N','nC1','nC2','nC3');

fprintf('DONE. Saved MAT:\n%s\n', outFile);

%% ===================== WRITE CSV SUMMARY TABLE (means) =====================
csvFile = fullfile(dataDir, [outBase '_TABLE.csv']);
try
    T = table(SNRdB_list(:), ACC_C1_mean(:), ACC_C2_mean(:), ACC_C3_mean(:), ACC_overall_mean(:), ...
        'VariableNames', {'SNR_dB','Acc_C1','Acc_C2','Acc_C3','Overall'});
    writetable(T, csvFile);
    fprintf('Saved CSV:\n%s\n', csvFile);
catch
    % fallback if writetable unavailable
    fid = fopen(csvFile,'w');
    fprintf(fid,'SNR_dB,Acc_C1,Acc_C2,Acc_C3,Overall\n');
    for k=1:numel(SNRdB_list)
        fprintf(fid,'%.0f,%.6f,%.6f,%.6f,%.6f\n', ...
            SNRdB_list(k), ACC_C1_mean(k), ACC_C2_mean(k), ACC_C3_mean(k), ACC_overall_mean(k));
    end
    fclose(fid);
    fprintf('Saved CSV (fallback):\n%s\n', csvFile);
end

%% =========================================================
%  FUNCTIONS
%% =========================================================
function Ynoisy = add_awgn_snr_colwise(Y, SNRdB)
% Add complex AWGN to achieve per-column SNR (dB).
    [d, N] = size(Y);
    Ps = mean(abs(Y).^2, 1);              % 1 x N
    snrLin = 10.^(SNRdB/10);
    Pn = Ps ./ snrLin;                    % 1 x N
    sigma = sqrt(Pn/2);                   % 1 x N
    sigmaMat = repmat(sigma, d, 1);       % d x N
    Nmat = sigmaMat .* (randn(d,N) + 1j*randn(d,N));
    Ynoisy = Y + Nmat;
end

function yN = block_norm_4x4(y, blkLen, EPSN)
yN = y;
nBlk = numel(y)/blkLen;
for b = 1:nBlk
    rows = (b-1)*blkLen + (1:blkLen);
    nb = norm(yN(rows));
    yN(rows) = yN(rows) / max(nb, EPSN);
end
end

function rho = rho_median_proj_energy(y, PBh, PIh, rowsH, EPSN)
Hblk = numel(rowsH);
rho_h = zeros(Hblk,1);
for h=1:Hblk
    yy = y(rowsH{h});
    yB = PBh{h} * yy;
    yI = PIh{h} * yy;
    EB = real(yB' * yB);
    EI = real(yI' * yI);
    rho_h(h) = EB / max(EB + EI, EPSN);
end
rho = median(rho_h);
end

function pred = predict_from_rho(rho, tL, tH)
if rho >= tH
    pred = 1; % C1
elseif rho <= tL
    pred = 2; % C2
else
    pred = 3; % C3
end
end

function [bestTL, bestTH, bestAcc] = learn_tL_tH(rho, cls)
grid = linspace(0,1,401);
bestAcc = -inf;
bestTL = 0.2; bestTH = 0.8;
for i=1:numel(grid)
    tL = grid(i);
    for j=1:numel(grid)
        tH = grid(j);
        if tH <= tL, continue; end
        pred = arrayfun(@(r)predict_from_rho(r, tL, tH), rho);
        acc = mean(pred(:) == cls(:));
        if acc > bestAcc
            bestAcc = acc;
            bestTL = tL;
            bestTH = tH;
        end
    end
end
end

function val = pick_field_first(S, names)
val = [];
for k=1:numel(names)
    if isfield(S, names{k})
        val = S.(names{k});
        return;
    end
end
end

function A = load_dict_matrix(matFile, targetSize, tag)
S = load(matFile);
fn = fieldnames(S);
A = [];
for i=1:numel(fn)
    v = S.(fn{i});
    if isnumeric(v) && ismatrix(v) && all(size(v) == targetSize)
        A = complex(v);
        return;
    end
end
for i=1:numel(fn)
    v = S.(fn{i});
    if isnumeric(v) && ismatrix(v) && size(v,1)==targetSize(1) && size(v,2)==targetSize(2)
        A = complex(v);
        return;
    end
end
error('Cannot find %s matrix of size %dx%d inside %s', tag, targetSize(1), targetSize(2), matFile);
end