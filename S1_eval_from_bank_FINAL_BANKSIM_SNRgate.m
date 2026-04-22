%% =========================================================
%  S1_eval_from_bank_FINAL_BANKSIM_SNR_STORE.m
%
%  BANK-SIM with LR reject using VOLTAGE-balance rhoV gate
%  Noise model: complex AWGN by SNRdB levels
%
%  Saves:
%    results_S1_bank_BANKSIM_SNR\
%      - STAMP_*.txt
%      - S1_bank_results_BANKSIM_SNR.mat   (optionally includes DIST)
%      - Fig_S1_bank_curves_BANKSIM_SNR.{png,pdf,emf}
%% =========================================================
clc; clear; close all;

%% ====== PATHS ======
bankRoot = 'E:\桌面\乔方昱课题论文\研究内容1\实验2\banks';
savefold = fullfile(fileparts(bankRoot), 'results_S1_bank_BANKSIM_SNR');
if ~exist(savefold,'dir'); mkdir(savefold); end

disp("BANK ROOT: " + bankRoot);
disp("SAVEFOLD : " + savefold);

%% =========================================================
%  READ BANKS
%% =========================================================
caseDirs = dir(fullfile(bankRoot, 'case*'));
caseDirs = caseDirs([caseDirs.isdir]);
assert(~isempty(caseDirs), 'No case* folders found under bankRoot.');

FL_bank    = [];
FR_bank    = [];
FLR_bank   = [];
FIN_bank   = [];
FLIN_bank  = [];
FRIN_bank  = [];
FLRIN_bank = [];

for i = 1:numel(caseDirs)
    cfold = fullfile(bankRoot, caseDirs(i).name);

    a = load(fullfile(cfold,'F_phy_boundary_L.mat'));
    FL_bank = [FL_bank; a.F_boundary(:).']; %#ok<AGROW>

    b = load(fullfile(cfold,'F_phy_boundary_R.mat'));
    FR_bank = [FR_bank; b.F_boundary(:).']; %#ok<AGROW>

    c = load(fullfile(cfold,'F_phy_boundary_LR.mat'));
    FLR_bank = [FLR_bank; c.F_boundary(:).']; %#ok<AGROW>

    d = load(fullfile(cfold,'F_phy_boundary_IN_8_15.mat'));
    FIN_bank = [FIN_bank; d.F_boundary_all]; %#ok<AGROW>

    e = load(fullfile(cfold,'F_phy_boundary_LIN_8_15.mat'));
    FLIN_bank = [FLIN_bank; e.F_boundary_all]; %#ok<AGROW>

    f = load(fullfile(cfold,'F_phy_boundary_RIN_8_15.mat'));
    FRIN_bank = [FRIN_bank; f.F_boundary_all]; %#ok<AGROW>

    g = load(fullfile(cfold,'F_phy_boundary_LRIN_8_15.mat'));
    FLRIN_bank = [FLRIN_bank; g.F_boundary_all]; %#ok<AGROW>
end

other_pool = [FLR_bank; FIN_bank; FLIN_bank; FRIN_bank; FLRIN_bank];

fprintf('\n=== BANK SIZES ===\n');
fprintf('FL_bank    : %d x 16\n', size(FL_bank,1));
fprintf('FR_bank    : %d x 16\n', size(FR_bank,1));
fprintf('FLR_bank   : %d x 16\n', size(FLR_bank,1));
fprintf('Other_pool : %d x 16\n\n', size(other_pool,1));

%% =========================================================
%  SETTINGS
%% =========================================================
mode = 'IV';

% --- thresholds ---
tau_sim    = 0.9970;
tau_gap    = 0.0000;
tau_rhoI   = 0.010;     % rhoI gate (current-balance)
tau_LR     = 0.87;      % LR similarity threshold (BANKSIM)
rhoV_min   = 0.15;      % LR reject only when rhoV is balanced enough
tau_E      = 1e-12;

% SNR grid (dB)
SNRdB_list = [50 45 40 35 30 25];

MC = 800;
rng(1);

% >>> store distributions per SNR (each has length MC)
STORE_DIST = true;

%% ====== PRECOMPUTE TEMPLATE BANK MATRICES ======
TLB  = template_pack_bank(FL_bank,  mode);
TRB  = template_pack_bank(FR_bank,  mode);
TLRB = template_pack_bank(FLR_bank, mode);

%% ====== STAMP ======
disp('=== THRESHOLDS (EFFECTIVE) ===');
fprintf('tau_sim=%.4f, tau_gap=%.4f, tau_rhoI=%.4f, tau_LR=%.4f, rhoV_min=%.4f, tau_E=%.1e, mode=%s\n', ...
    tau_sim, tau_gap, tau_rhoI, tau_LR, rhoV_min, tau_E, mode);

stampfile = fullfile(savefold, sprintf('STAMP_%s.txt', datestr(now,'yyyymmdd_HHMMSS')));
fid=fopen(stampfile,'w');
fprintf(fid,'SCRIPT: %s\n', mfilename('fullpath'));
fprintf(fid,'tau_sim=%.6f\n tau_gap=%.6f\n tau_rhoI=%.6f\n tau_LR=%.6f\n rhoV_min=%.6f\n tau_E=%.3e\n mode=%s\n', ...
    tau_sim, tau_gap, tau_rhoI, tau_LR, rhoV_min, tau_E, mode);
fprintf(fid,'SNRdB_list=%s\n', mat2str(SNRdB_list));
fprintf(fid,'MC=%d\n', MC);
fprintf(fid,'STORE_DIST=%d\n', STORE_DIST);
fclose(fid);
fprintf('Wrote stamp: %s\n', stampfile);

%% =========================================================
%  SANITY CHECK at highest SNR (C1 samples)
%% =========================================================
MCcheck = 5000;
labels = (rand(MCcheck,1) < 0.5);
Ftrue = complex(zeros(MCcheck,16));
idxL = randi(size(FL_bank,1), sum(labels), 1);
idxR = randi(size(FR_bank,1), sum(~labels),1);
Ftrue(labels,:)  = FL_bank(idxL,:);
Ftrue(~labels,:) = FR_bank(idxR,:);

Y0 = add_awgn_snr_mat(Ftrue, SNRdB_list(1));  % highest SNR

[dec0, gap0, rhoI0, rhoV0, sBest0, sLR0, reason0] = decide_S1_vec_bank_rhoVgate( ...
    Y0, TLB, TRB, TLRB, tau_sim, tau_gap, tau_rhoI, tau_LR, rhoV_min, tau_E);

fprintf('\n=== CHECK @ SNR=%.1f dB (C1 samples) ===\n', SNRdB_list(1));
fprintf('Fail=%.4f, Cov=%.4f\n', mean(dec0==0), mean(dec0~=0));
fprintf('reason breakdown: rhoI=%.4f LR=%.4f sim=%.4f gap=%.4f\n', ...
    mean(reason0==1), mean(reason0==2), mean(reason0==3), mean(reason0==4));
fprintf('sBest percentiles [1 5 50 95 99] = %s\n', mat2str(prctile(sBest0,[1 5 50 95 99]),4));
fprintf('rhoI percentiles  [1 5 50 95 99] = %s\n', mat2str(prctile(rhoI0,[1 5 50 95 99]),4));
fprintf('rhoV percentiles  [1 5 50 95 99] = %s\n', mat2str(prctile(rhoV0,[1 5 50 95 99]),4));
fprintf('sLR  percentiles  [1 5 50 95 99] = %s\n', mat2str(prctile(sLR0,[1 5 50 95 99]),4));
fprintf('gap  percentiles  [1 5 50 95 99] = %s\n\n', mat2str(prctile(gap0,[1 5 50 95 99]),4));

%% =========================================================
%  MAIN LOOP over SNR
%% =========================================================
nS = numel(SNRdB_list);

Fail_C1     = zeros(nS,1);
Fail_Others = zeros(nS,1);
AccC1_cond  = nan(nS,1);
CovC1       = zeros(nS,1);
GapC1_mean  = zeros(nS,1);

FailRho = zeros(nS,1);
FailLR  = zeros(nS,1);
FailSim = zeros(nS,1);
FailGap = zeros(nS,1);

% --- store per-sample vectors (length=MC) ---
if STORE_DIST
    DIST = struct();

    DIST.C1.dec         = cell(nS,1);
    DIST.C1.true_dec    = cell(nS,1);
    DIST.C1.gapLR       = cell(nS,1);
    DIST.C1.rhoI        = cell(nS,1);
    DIST.C1.rhoV        = cell(nS,1);
    DIST.C1.sBest       = cell(nS,1);
    DIST.C1.sLR         = cell(nS,1);
    DIST.C1.fail_reason = cell(nS,1);

    DIST.Others.dec         = cell(nS,1);
    DIST.Others.gapLR       = cell(nS,1);
    DIST.Others.rhoI        = cell(nS,1);
    DIST.Others.rhoV        = cell(nS,1);
    DIST.Others.sBest       = cell(nS,1);
    DIST.Others.sLR         = cell(nS,1);
    DIST.Others.fail_reason = cell(nS,1);
end

worstFailOthers = inf;
worstSNR = NaN;

for s = 1:nS
    snrdb = SNRdB_list(s);

    % ---- C1 from banks ----
    labels = (rand(MC,1) < 0.5); % 1=L, 0=R
    true_dec = -ones(MC,1); true_dec(labels) = +1;

    Ftrue = complex(zeros(MC,16));
    idxL = randi(size(FL_bank,1), sum(labels), 1);
    idxR = randi(size(FR_bank,1), sum(~labels),1);
    Ftrue(labels,:)  = FL_bank(idxL,:);
    Ftrue(~labels,:) = FR_bank(idxR,:);

    Y = add_awgn_snr_mat(Ftrue, snrdb);

    [dec, gapLR, rhoI, rhoV, sBest, sLR, fail_reason] = decide_S1_vec_bank_rhoVgate( ...
        Y, TLB, TRB, TLRB, tau_sim, tau_gap, tau_rhoI, tau_LR, rhoV_min, tau_E);

    fail = (dec==0);
    conf = ~fail;

    Fail_C1(s)    = mean(fail);
    CovC1(s)      = mean(conf);
    GapC1_mean(s) = mean(gapLR);

    if any(conf)
        AccC1_cond(s) = mean(dec(conf)==true_dec(conf));
    end

    FailRho(s) = mean(fail_reason==1);
    FailLR(s)  = mean(fail_reason==2);
    FailSim(s) = mean(fail_reason==3);
    FailGap(s) = mean(fail_reason==4);

    if STORE_DIST
        DIST.C1.dec{s}         = dec;
        DIST.C1.true_dec{s}    = true_dec;
        DIST.C1.gapLR{s}       = gapLR;
        DIST.C1.rhoI{s}        = rhoI;
        DIST.C1.rhoV{s}        = rhoV;
        DIST.C1.sBest{s}       = sBest;
        DIST.C1.sLR{s}         = sLR;
        DIST.C1.fail_reason{s} = fail_reason;
    end

    % ---- Others from pool ----
    idxO = randi(size(other_pool,1), MC, 1);
    Fother = other_pool(idxO,:);
    Y2 = add_awgn_snr_mat(Fother, snrdb);

    [dec2, gap2, rhoI2, rhoV2, sBest2, sLR2, reason2] = decide_S1_vec_bank_rhoVgate( ...
        Y2, TLB, TRB, TLRB, tau_sim, tau_gap, tau_rhoI, tau_LR, rhoV_min, tau_E);

    Fail_Others(s) = mean(dec2==0); % Others should be rejected -> higher is better

    if STORE_DIST
        DIST.Others.dec{s}         = dec2;
        DIST.Others.gapLR{s}       = gap2;
        DIST.Others.rhoI{s}        = rhoI2;
        DIST.Others.rhoV{s}        = rhoV2;
        DIST.Others.sBest{s}       = sBest2;
        DIST.Others.sLR{s}         = sLR2;
        DIST.Others.fail_reason{s} = reason2;
    end

    if Fail_Others(s) < worstFailOthers
        worstFailOthers = Fail_Others(s);
        worstSNR = snrdb;
    end
end

fprintf('Worst Others reject=%.3f at SNR=%.1f dB\n', worstFailOthers, worstSNR);
fprintf('Worst Others false-confirm rate = %.3f\n', 1-worstFailOthers);

fprintf('Fail_C1 range: [%.3f, %.3f]\n', min(Fail_C1), max(Fail_C1));
fprintf('Fail_Others range: [%.3f, %.3f]\n', min(Fail_Others), max(Fail_Others));
fprintf('CovC1 range: [%.3f, %.3f]\n', min(CovC1), max(CovC1));

%% ====== SAVE MAT ======
outMat = fullfile(savefold,'S1_bank_results_BANKSIM_SNR.mat');
if STORE_DIST
    save(outMat, ...
        'bankRoot','savefold','SNRdB_list','MC', ...
        'tau_sim','tau_gap','tau_rhoI','tau_LR','rhoV_min','tau_E','mode', ...
        'Fail_C1','Fail_Others','AccC1_cond','CovC1','GapC1_mean', ...
        'FailRho','FailLR','FailSim','FailGap', ...
        'DIST','STORE_DIST');
else
    save(outMat, ...
        'bankRoot','savefold','SNRdB_list','MC', ...
        'tau_sim','tau_gap','tau_rhoI','tau_LR','rhoV_min','tau_E','mode', ...
        'Fail_C1','Fail_Others','AccC1_cond','CovC1','GapC1_mean', ...
        'FailRho','FailLR','FailSim','FailGap', ...
        'STORE_DIST');
end
fprintf('\nSaved: %s\n', outMat);

%% =========================================================
%  FIG) Curves vs SNR
%% =========================================================
fig1 = figure('Color','w','Position',[80 80 1400 560]);

subplot(2,3,1);
plot(SNRdB_list, Fail_C1, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('Rate'); title('FAIL rate: C1'); ylim([0 1]);

subplot(2,3,2);
plot(SNRdB_list, Fail_Others, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('Rate'); title('REJECT rate: Others'); ylim([0 1]);

subplot(2,3,3);
plot(SNRdB_list, AccC1_cond, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('Acc'); title('Direction acc (C1|confirmed)'); ylim([0 1]);

subplot(2,3,4);
plot(SNRdB_list, CovC1, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('Rate'); title('Coverage (C1 confirm rate)'); ylim([0 1]);

subplot(2,3,5);
plot(SNRdB_list, FailLR, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('Rate'); title('C1 fail due to LR gate'); ylim([0 1]);

subplot(2,3,6);
plot(SNRdB_list, FailRho, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('Rate'); title('C1 fail due to rhoI gate'); ylim([0 1]);

sgtitle(sprintf('S1 BANK-SIM (rhoV gate) | mode=%s | tauSim=%.4f tauRhoI=%.3f tauLR=%.3f rhoVmin=%.2f | MC=%d', ...
    mode, tau_sim, tau_rhoI, tau_LR, rhoV_min, MC), 'FontWeight','bold');

set(findall(fig1,'-property','FontName'),'FontName','Times New Roman');
set(findall(fig1,'-property','FontSize'),'FontSize',9);

outbase1 = fullfile(savefold,'Fig_S1_bank_curves_BANKSIM_SNR');
print(fig1,[outbase1 '.png'],'-dpng','-r600');
set(fig1,'PaperPositionMode','auto');
print(fig1,[outbase1 '.pdf'],'-dpdf','-bestfit');
print(fig1,[outbase1 '.emf'],'-dmeta','-r600');
fprintf('Saved: %s.{png,pdf,emf}\n', outbase1);

fprintf('\nDONE.\n');

%% =========================================================
%  Local functions
%% =========================================================
function Y = add_awgn_snr_mat(F, SNRdB)
% Add complex AWGN to achieve per-row SNR (dB).
% Per-row definition: Ps = mean(|F|^2), Pn = Ps / 10^(SNR/10)
    if isinf(SNRdB)
        Y = F; return;
    end
    MC = size(F,1); D = size(F,2);

    Ps = mean(abs(F).^2, 2);      % MC x 1
    snrLin = 10.^(SNRdB/10);
    Pn = Ps ./ snrLin;            % MC x 1 (complex noise power)
    sigma = sqrt(Pn/2);           % per real/imag std

    sigmaMat = sigma(:,ones(1,D));  %#ok<NBRAK>  % MC x D for old MATLAB
    N = sigmaMat .* (randn(MC,D) + 1j*randn(MC,D));
    Y = F + N;
end

function TB = template_pack_bank(Fbank, mode)
    Ft = select_feat_mat(Fbank, mode);     % (N x D) complex
    X  = [real(Ft), imag(Ft)].';           % (2D x N)
    nrm = sqrt(sum(X.^2,1)) + eps;
    TB.M = X ./ nrm;                       % normalized columns
    TB.mode = mode;
end

function [dec, gapLR, rhoI, rhoV, sBest, sLR, fail_reason] = decide_S1_vec_bank_rhoVgate( ...
    Y, TLB, TRB, TLRB, tau_sim, tau_gap, tau_rhoI, tau_LR, rhoV_min, tau_E)

    MC = size(Y,1);
    dec = zeros(MC,1);
    gapLR = zeros(MC,1);
    rhoI = ones(MC,1);
    rhoV = ones(MC,1);
    sBest = -inf(MC,1);
    sLR   = -inf(MC,1);
    fail_reason = zeros(MC,1); % 0 pass; 1 rhoI; 2 LR; 3 sim; 4 gap

    % ---- rhoI gate (CURRENT balance) ----
    IL = Y(:,[1 5 9 13]);
    IR = Y(:,[2 6 10 14]);
    EL = sum(abs(IL).^2,2);
    ER = sum(abs(IR).^2,2);
    EmaxI = max(EL,ER);
    EminI = min(EL,ER);
    rhoI = EminI ./ max(EmaxI,eps);

    badE = (EmaxI < tau_E);
    badRhoI = (rhoI > tau_rhoI);
    kill = badE | badRhoI;
    fail_reason(kill) = 1;

    % ---- rhoV (VOLTAGE balance) for LR gate ----
    VL = Y(:,[3 7 11 15]);
    VR = Y(:,[4 8 12 16]);
    EVL = sum(abs(VL).^2,2);
    EVR = sum(abs(VR).^2,2);
    EmaxV = max(EVL,EVR);
    EminV = min(EVL,EVR);
    rhoV = EminV ./ max(EmaxV,eps);

    % ---- feature vector ----
    Yt  = select_feat_mat(Y, TLB.mode);
    Yri = [real(Yt), imag(Yt)];
    Ynrm = sqrt(sum(Yri.^2,2)) + eps;
    Yn = Yri ./ Ynrm;

    % ---- best similarity to L-bank and R-bank ----
    sL = max(Yn * TLB.M, [], 2);
    sR = max(Yn * TRB.M, [], 2);

    useL = (sL >= sR);
    sBest = sR; s2 = sL; dec0 = -ones(MC,1);
    sBest(useL) = sL(useL);
    s2(useL)    = sR(useL);
    dec0(useL)  = +1;

    gapLR = sBest - s2;

    % ---- best similarity to LR-bank ----
    sLR = max(Yn * TLRB.M, [], 2);

    % ---- LR reject: only when VOLTAGE is balanced enough ----
    rejLR = (rhoV >= rhoV_min) & (sLR >= tau_LR) & (~kill);
    fail_reason(rejLR) = 2;

    % ---- sim / gap gates ----
    badSim = (sBest < tau_sim) & (~kill) & (~rejLR);
    fail_reason(badSim) = 3;

    badGap = (gapLR < tau_gap) & (~kill) & (~rejLR) & (~badSim);
    fail_reason(badGap) = 4;

    pass = (~kill) & (~rejLR) & (~badSim) & (~badGap);
    dec(pass) = dec0(pass);
end

function Ft = select_feat_mat(F, mode)
    switch upper(mode)
        case 'V'
            Ft = F(:,[3 4 7 8 11 12 15 16]);
        case 'I'
            Ft = F(:,[1 2 5 6 9 10 13 14]);
        otherwise
            Ft = F; % 'IV' or others
    end
end