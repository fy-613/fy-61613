%% =========================================================
%  FIG_ABC_ACC_BOX_RHO_KDE_50_25.m
%  (a) Boxplot: success rate (overall accuracy) across SNR (from ACC_mc)
%  (b) KDE: criterion rho @ SNR=50 (C1/C2/C3)
%  (c) KDE: criterion rho @ SNR=25 (C1/C2/C3)
%  Subfigure labels (a)(b)(c) are placed under xlabel (two-line xlabel).
%% =========================================================
clc; clear; close all;

%% ====== USER ======
dataDir = 'E:\桌面\乔方昱课题论文\研究内容1\实验3\数据\数据';
outDir = 'E:\桌面\乔方昱课题论文\研究内容1\实验3\图片';

SNR_show  = [50 25];   % KDE panels (b)(c)
MCplot    = 50;        % KDE用：每档重复加噪次数（20~100）
rng(1);

% KDE smoothing
BW_FACTOR  = 6;        % 越大越圆润
BW_FLOOR   = 0.02;     % rho域[0,1]的最小带宽
NGRID      = 300;
NORM_SHAPE = true;     % true: 每条KDE归一化(max=1)，更好看

%% ====== Figure style (IEEE single column) ======
STYLE.fontName = 'Times New Roman';
STYLE.fontSize = 8;
STYLE.dpi      = 600;

% single-column size
FIGW = 3.5;
FIGH = 2.4;

%% ====== Find latest result mat ======
pat = 'ACC_PROJENERGY_SAVEACC_MC_SNR_*.mat';
L = dir(fullfile(dataDir, pat));
assert(~isempty(L), 'No result mat found: %s', fullfile(dataDir, pat));
[~, idx] = max([L.datenum]);
matPath = fullfile(L(idx).folder, L(idx).name);
R = load(matPath);
fprintf('Loaded: %s\n', matPath);

SNRdB_list = R.SNRdB_list(:);
ACC_mc_all = R.ACC_mc;      % cell, each 800x1

tL  = R.tL;  tH  = R.tH;  sIV = R.sIV;
dictBndFile = R.dictBndFile;
dictIntFile = R.dictIntFile;

%% =========================================================
%  PART (a): boxplot of success rate distributions across SNR
% sort SNR descending (50 -> 25) for nicer order
[SNRs, ord] = sort(SNRdB_list, 'descend');
ACC_mc = ACC_mc_all(ord);
nS = numel(SNRs);

% pack to long vector + group index
xAcc = []; gAcc = [];
for k = 1:nS
    v = ACC_mc{k}(:);
    xAcc = [xAcc; v];
    gAcc = [gAcc; k*ones(numel(v),1)];
end

%% =========================================================
%  PART (b)(c): build rho calculator (dict + projectors + cases)
%% =========================================================
% Load dictionaries
DB = load_dict_matrix(dictBndFile, [2 16], 'DB');  DB = DB.';  % 16x2
DI = load_dict_matrix(dictIntFile, [8 16], 'DI');  DI = DI.';  % 16x8

% FP format
Hblk=4; blkLen=4;
idxV = [];
for b=1:Hblk
    base=(b-1)*blkLen;
    idxV=[idxV, base+3, base+4]; %#ok<AGROW>
end
EPSN=1e-12;

% whiten dicts
Swh = ones(16,1); Swh(idxV)=sIV;
DBw = diag(Swh)*DB;
DIw = diag(Swh)*DI;

% projectors per harmonic block
rowsH = cell(Hblk,1); PBh=cell(Hblk,1); PIh=cell(Hblk,1);
for h=1:Hblk
    rows=(h-1)*blkLen+(1:blkLen);
    rowsH{h}=rows;
    [QB,~]=qr(DBw(rows,:),0);
    [QI,~]=qr(DIw(rows,:),0);
    PBh{h}=QB*QB';
    PIh{h}=QI*QI';
end

% Reload cases: fp16 + cls
Lc = dir(fullfile(dataDir,'*.mat'));
assert(~isempty(Lc),'No case mat found in dataDir.');

Ymat=[]; cls=[];
for i=1:numel(Lc)
    S = load(fullfile(dataDir,Lc(i).name));
    c = pick_field_first(S, {'cls','class','label','ycls','Class'});
    y = pick_field_first(S, {'fp16','FP16','y','y_fp','F','Fin','fin','fingerprint','fp'});
    if isempty(c) || isempty(y), continue; end
    c=double(c); y=y(:);
    if ~isfinite(c) || ~ismember(c,[1 2 3]), continue; end
    if numel(y)~=16 || ~isnumeric(y), continue; end
    cls(end+1,1)=c; %#ok<AGROW>
    Ymat(:,end+1)=complex(y); %#ok<AGROW>
end
N=size(Ymat,2);
assert(N>0,'No valid cases loaded. Check fp16/cls fields.');
fprintf('Cases: N=%d | C1=%d C2=%d C3=%d\n', N, sum(cls==1), sum(cls==2), sum(cls==3));

% Collect rho stacks for each SNR_show
xg = linspace(0,1,NGRID);
rho_c1 = cell(numel(SNR_show),1);
rho_c2 = cell(numel(SNR_show),1);
rho_c3 = cell(numel(SNR_show),1);

n1 = sum(cls==1); n2=sum(cls==2); n3=sum(cls==3);

for s = 1:numel(SNR_show)
    snrdb = SNR_show(s);

    r1 = zeros(n1*MCplot,1);
    r2 = zeros(n2*MCplot,1);
    r3 = zeros(n3*MCplot,1);

    p1=1; p2=1; p3=1;

    for m=1:MCplot
        Ynoisy = add_awgn_snr_colwise(Ymat, snrdb);

        rho_vec = zeros(N,1);
        for i=1:N
            yy = Ynoisy(:,i);
            yy(idxV) = yy(idxV)*sIV;                   % whitening
            yy = block_norm_4x4(yy, blkLen, EPSN);     % block norm
            rho_vec(i) = rho_median_proj_energy(yy, PBh, PIh, rowsH, EPSN);
        end

        v1 = rho_vec(cls==1);  nn1=numel(v1);
        v2 = rho_vec(cls==2);  nn2=numel(v2);
        v3 = rho_vec(cls==3);  nn3=numel(v3);

        r1(p1:p1+nn1-1)=v1; p1=p1+nn1;
        r2(p2:p2+nn2-1)=v2; p2=p2+nn2;
        r3(p3:p3+nn3-1)=v3; p3=p3+nn3;
    end

    rho_c1{s}=r1; rho_c2{s}=r2; rho_c3{s}=r3;
end

%% =========================================================
%  FIG: single-column, 3 panels (a)(b)(c)

fig = figure('Units','inches','Position',[1 1 FIGW FIGH],'Color','w');

% 只调三个子图上下间距，其他不动
axA = axes('Position',[0.1 0.8 0.85 0.18]);   % (a)
axB = axes('Position',[0.1 0.48 0.85 0.18]);   % (b)
axC = axes('Position',[0.1 0.18 0.85 0.18]);   % (c)

% ---------- (a) boxplot: success rate ----------
boxplot(axA, xAcc, gAcc, 'Symbol','');  % no outlier symbol (cleaner)
set(axA,'XTick',1:nS,'XTickLabel',cellstr(string(SNRs)));
ylim(axA,[0.75 1]);
ylabel(axA,'Accuracy');

% ===== xlabel 单独放 =====
hxA = xlabel(axA, 'SNR (dB)');
set(hxA,'Units','normalized');
set(hxA,'Position',[0.5, -0.25, 0]);

% ===== (a) 单独成 text =====
text(axA, 0.5, -0.44, '(a)', ...
    'Units','normalized', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName', STYLE.fontName, ...
    'FontSize', STYLE.fontSize);

grid(axA,'off'); box(axA,'on');
title(axA,''); % don't use title

% ---------- (b)(c) KDE of rho ----------
for s = 1:numel(SNR_show)
    snrdb = SNR_show(s);

    if s==1
        ax = axB;
    else
        ax = axC;
    end
    hold(ax,'on');

    f1 = ksdensity(rho_c1{s}, xg, 'Bandwidth', bw_choose(rho_c1{s},BW_FACTOR,BW_FLOOR), 'Support',[0 1]);
    f2 = ksdensity(rho_c2{s}, xg, 'Bandwidth', bw_choose(rho_c2{s},BW_FACTOR,BW_FLOOR), 'Support',[0 1]);
    f3 = ksdensity(rho_c3{s}, xg, 'Bandwidth', bw_choose(rho_c3{s},BW_FACTOR,BW_FLOOR), 'Support',[0 1]);

    if NORM_SHAPE
        f1=f1/max(f1+eps); f2=f2/max(f2+eps); f3=f3/max(f3+eps);
        ylabel(ax,'Norm. density');
    else
        ylabel(ax,'pdf');
    end

    h1 = plot(ax,xg,f1,'LineWidth',1.2);
    h2 = plot(ax,xg,f2,'LineWidth',1.2);
    h3 = plot(ax,xg,f3,'LineWidth',1.2);

    % 只在第一张KDE（也就是(b)，s==1）记录句柄给全局legend用
    if s==1
        hC1 = h1; hC2 = h2; hC3 = h3;
        % xlabel(ax, {'\eta', '(b)'});   % <- subfigure label under xlabel
        hxB = xlabel(ax, {'(b)'});
        set(hxB,'Units','normalized');
        set(hxB,'Position',[0.5, -0.25, 0]);
    else
    % ===== xlabel 单独放 =====
    hxC = xlabel(ax, 'Projection energy ratio \eta');
    set(hxC,'Units','normalized');
    set(hxC,'Position',[0.5, -0.2, 0]);

    % ===== (c) 单独成 text =====
    text(ax, 0.5, -0.4, '(c)', ...
        'Units','normalized', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','top', ...
        'FontName', STYLE.fontName, ...
        'FontSize', STYLE.fontSize);
    end
    % thresholds (light gray)
    % ===== tL / tH threshold lines (thicker, black) =====
    lw = 1.2;  % 线宽，想更粗就 2.0
    xline(ax, tL, '--', 'Color','k', 'LineWidth', lw);
    xline(ax, tH, '--', 'Color','k', 'LineWidth', lw);

    % ===== rotated labels (clockwise 90° => Rotation = -90) =====
    yl = ylim(ax);
    ypos = yl(2) + 0.4*(yl(2)-yl(1));  % 离顶部一点点，避免贴边

    text(ax, tL, ypos, '\tau_L', ...
        'Color','k', 'Rotation',0, ...
        'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'FontName', STYLE.fontName, 'FontSize', STYLE.fontSize);

    text(ax, tH, ypos, '\tau_H', ...
        'Color','k', 'Rotation',0, ...
        'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'FontName', STYLE.fontName, 'FontSize', STYLE.fontSize);

    xlim(ax,[0.2 0.6]);
    grid(ax,'off'); box(ax,'on');

    hold(ax,'off');
end
lg = legend(ax, [hC1 hC2 hC3], {'C1','C2','C3'}, ...
    'Orientation','horizontal', 'Location','southoutside');
lg.Box = 'off';

% 如果你发现它和xlabel打架，就手动挪一下（推荐）
lg.Units = 'normalized';
lg.Position = [0.25, 0.01, 0.50, 0.04];  % [x y w h] 你可微调 y: 0~0.03

% ====== Force all fonts to Times New Roman, 8 pt ======
set(findall(fig,'-property','FontName'),'FontName','Times New Roman');
set(findall(fig,'-property','FontSize'),'FontSize',8);

% ====== Export ======
outbase = fullfile(outDir, 'Fig_ABC_ACC_BOX_RHO_KDE_singlecol');
try
    exportgraphics(fig,[outbase '.pdf'],'ContentType','vector','Resolution',STYLE.dpi,'Bounds','tight');
    exportgraphics(fig,[outbase '.emf'],'ContentType','vector','Resolution',STYLE.dpi,'Bounds','tight');
    exportgraphics(fig,[outbase '.png'],'Resolution',STYLE.dpi,'Bounds','tight');
catch
    exportgraphics(fig,[outbase '.pdf'],'ContentType','vector','Resolution',STYLE.dpi);
    exportgraphics(fig,[outbase '.emf'],'ContentType','vector','Resolution',STYLE.dpi);
    exportgraphics(fig,[outbase '.png'],'Resolution',STYLE.dpi);
end
fprintf('Saved: %s.[pdf/emf/png]\n', outbase);

%% ===================== Local functions =====================
function bw = bw_choose(v, fac, floorBW)
    v=v(:); v=v(isfinite(v));
    n=numel(v);
    if n<2, bw=floorBW; return; end
    s=std(v);
    bw0 = 1.06*s*n^(-1/5);
    bw = max(floorBW, fac*bw0);
    if ~isfinite(bw) || bw<=0, bw=floorBW; end
end

function Ynoisy = add_awgn_snr_colwise(Y, SNRdB)
    [d,N] = size(Y);
    Ps = mean(abs(Y).^2, 1);
    snrLin = 10.^(SNRdB/10);
    Pn = Ps ./ snrLin;
    sigma = sqrt(Pn/2);
    sigmaMat = repmat(sigma, d, 1);
    Nmat = sigmaMat .* (randn(d,N) + 1j*randn(d,N));
    Ynoisy = Y + Nmat;
end

function yN = block_norm_4x4(y, blkLen, EPSN)
    yN=y;
    nBlk=numel(y)/blkLen;
    for b=1:nBlk
        rows=(b-1)*blkLen+(1:blkLen);
        nb=norm(yN(rows));
        yN(rows)=yN(rows)/max(nb,EPSN);
    end
end

function rho = rho_median_proj_energy(y, PBh, PIh, rowsH, EPSN)
    Hblk=numel(rowsH);
    rho_h=zeros(Hblk,1);
    for h=1:Hblk
        yy=y(rowsH{h});
        yB=PBh{h}*yy; yI=PIh{h}*yy;
        EB=real(yB'*yB); EI=real(yI'*yI);
        rho_h(h)=EB/max(EB+EI,EPSN);
    end
    rho=median(rho_h);
end

function val = pick_field_first(S, names)
    val=[];
    for k=1:numel(names)
        if isfield(S,names{k}), val=S.(names{k}); return; end
    end
end

function A = load_dict_matrix(matFile, targetSize, tag)
    S=load(matFile); fn=fieldnames(S);
    for i=1:numel(fn)
        v=S.(fn{i});
        if isnumeric(v)&&ismatrix(v)&&all(size(v)==targetSize)
            A=complex(v); return;
        end
    end
    error('Cannot find %s matrix of size %dx%d inside %s', tag, targetSize(1), targetSize(2), matFile);
end