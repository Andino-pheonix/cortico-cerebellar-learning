"""Data-derived static figures; no fitted or illustrative result curves."""
import csv,json,math
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import argparse

INK='#17334b';MUTED='#647c8f'
COLORS={'feedback':'#647c8f','original_joint':'#c98964','tuned_cortex':'#147f9a','tuned_cerebellum':'#8458b2','tuned_joint':'#e16a25'}
LABELS={'feedback':'仅反馈','original_joint':'原联合','tuned_cortex':'调优皮层','tuned_cerebellum':'调优小脑','tuned_joint':'调优联合'}

class SVG:
    def __init__(self,w,h):
        self.items=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:"Microsoft YaHei",Arial,sans-serif}</style>']
    def text(self,x,y,t,size=22,color=INK,anchor='start',bold=False):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'+(' font-weight="600"' if bold else '')+'>'+escape(str(t))+'</text>')
    def rect(self,x,y,w,h,fill,stroke=None,rx=0):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="3"' if stroke else '')+'/>')
    def line(self,p,color,width=3,dash=False):
        self.items.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+f'" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
    def dot(self,x,y,col,r=5):self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}"/>')
    def save(self,p):p.write_text('\n'.join(self.items+['</svg>']),encoding='utf-8')

def panel(s,x,y,w,h,xlim,ylim,title,xlab,ylab,xticks,yticks,shade=False):
    def xy(a,b):return x+(a-xlim[0])/(xlim[1]-xlim[0])*w,y+h-(b-ylim[0])/(ylim[1]-ylim[0])*h
    s.text(x,y-42,title,25,bold=True);s.text(x,y-13,ylab,19,MUTED)
    if shade:
        l,_=xy(20.5,0);r,_=xy(100.5,0);s.rect(l,y,r-l,h,'#f5f0f8')
    for t in xticks:
        xx,_=xy(t,ylim[0]);s.line([(xx,y),(xx,y+h)],'#e1e9ee',1);s.text(xx,y+h+25,t,18,anchor='middle')
    for t in yticks:
        _,yy=xy(xlim[0],t);s.line([(x,yy),(x+w,yy)],'#e1e9ee',1);s.text(x-10,yy+6,t,18,anchor='end')
    s.line([(x,y),(x,y+h),(x+w,y+h)],MUTED,1.5);s.text(x+w/2,y+h+61,xlab,21,anchor='middle')
    return xy

def band(s,xy,t,m,sd,color):
    p=[xy(a,b) for a,b in zip(t,m-sd)]+[xy(a,b) for a,b in zip(t[::-1],(m+sd)[::-1])]
    s.items.append('<polygon points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in p)+f'" fill="{color}" opacity=".13"/>')

def read_csv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def heatmap(data,out):
    sel=json.loads((data/'frozen_selection.json').read_text(encoding='utf-8'))
    rows=read_csv(data/'coarse_aggregate.csv');rates=sel['coarse_rates_final'];n=len(rates)
    s=SVG(1770,1270);s.text(60,60,'训练集搜索：分别调节皮层与小脑的更新系数',34,bold=True)
    s.text(60,102,f'5次训练初始化 · {sel["counts"]["total"]}个候选组合 · 颜色表示综合目标，越低越好',24,MUTED)
    x0,y0,cell=135,230,54
    s.text(x0+cell*n/2,168,'小脑系数 ηCb（等间距离散网格，非线性数轴）',23,anchor='middle')
    s.text(40,210,'皮层 ηC',22,bold=True)
    for j,v in enumerate(rates):s.text(x0+(j+.5)*cell,y0-20,f'{v:g}',16,anchor='middle')
    for i,v in enumerate(rates):s.text(x0-18,y0+(i+.5)*cell+6,f'{v:g}',18,anchor='end')
    bypair={(float(r['eta_c']),float(r['eta_cb'])):r for r in rows}
    for i,c in enumerate(rates):
        for j,b in enumerate(rates):
            r=bypair[(c,b)];x=x0+j*cell;y=y0+i*cell;good=r['stable']=='True'
            z=min(1.,max(0.,math.log1p(float(r['objective']))/math.log1p(6.)))
            lo=np.array([20,102,135]);hi=np.array([231,242,246]);rgb=(lo+(hi-lo)*z).astype(int)
            color='#'+''.join(f'{v:02x}' for v in rgb) if good else '#f8eded'
            s.rect(x+1,y+1,cell-2,cell-2,color)
            if not good:s.text(x+cell/2,y+cell/2+7,'×',24,'#b67777',anchor='middle')
    # The selected candidate can lie between coarse cells; label exact rates separately.
    old=(rates.index(.4),rates.index(.4));i,j=old;s.rect(x0+j*cell+3,y0+i*cell+3,cell-6,cell-6,'none','#d9844f')
    s.text(1140,225,'训练集固定选择',27,bold=True)
    yy=280
    for key,name in [('joint','联合'),('cortex','皮层单独'),('cerebellum','小脑单独')]:
        r=sel[key];s.text(1140,yy,name,24,bold=True);s.text(1140,yy+35,f'ηC={r["eta_c"]:.5g}　ηCb={r["eta_cb"]:.5g}',21)
        s.text(1140,yy+69,f'目标分数 {r["objective"]:.4f}',21,MUTED);yy+=145
    s.text(1140,740,'筛查标准',25,bold=True)
    for k,t in enumerate(['全部试次、状态与权重有限','全部过程没有小脑输出限幅','适应期峰值起步误差 ≤60°','适应期峰值终点误差 ≤50 mm']):s.text(1140,781+k*35,t,20)
    s.text(1140,970,'×：至少一个训练种子未通过',20,'#a26666')
    s.text(1140,1007,'橙框：原联合系数0.4／0.4',20,'#c98964')
    s.text(1140,1065,'目标颜色（log(1+分数)映射）',20,MUTED)
    for k,z in enumerate(np.linspace(0,1,100)):
        rgb=(np.array([20,102,135])+(np.array([231,242,246])-np.array([20,102,135]))*z).astype(int)
        s.rect(1140+k*4,1083,4.1,20,'#'+''.join(f'{v:02x}' for v in rgb))
    s.text(1140,1131,'0：较优',18);s.text(1540,1131,'6及以上：较差',18,anchor='end')
    s.text(60,1190,'粗网格含单模块边界；细搜使用训练集，图中网格仅显示粗搜／扩域。',23)
    s.text(60,1231,'初始上限3.2命中边界后，按规则扩域；所有参数在验证集运行前冻结。筛查通过不等于动力学稳定证明。',21,MUTED)
    s.save(out/'tuning_search.svg')

def heldout(data,out):
    summary=json.loads((data/'test_summary.json').read_text(encoding='utf-8'));rows=read_csv(data/'test_trials.csv')
    seeds=sorted({int(r['seed']) for r in rows});stats={}
    for label in COLORS:
        stats[label]={m:np.array([[float(r[m]) for r in rows if int(r['seed'])==seed and r['condition']==label] for seed in seeds]) for m in ('early_error_deg','endpoint_error_mm')}
    s=SVG(1640,1080);s.text(60,56,'独立初始化验证：分别调参后，联合学习的控制表现',33,bold=True)
    s.text(60,98,'全新5个网络 · 参数由训练集固定 · 阴影为30°旋转期 · 曲线为均值±样本标准差',23,MUTED)
    lo=min(np.min(v['early_error_deg'].mean(0)-v['early_error_deg'].std(0,ddof=1)) for v in stats.values());hi=max(np.max(v['early_error_deg'].mean(0)+v['early_error_deg'].std(0,ddof=1)) for v in stats.values())
    a=panel(s,110,205,610,300,(1,120),(math.floor(lo/15)*15,math.ceil(hi/15)*15),'A  起步方向与撤扰后效','试次','有符号误差（°）；100 ms',[1,20,60,100,120],list(np.arange(math.ceil(lo/15)*15,hi+1,15)),True)
    top=max(np.max(v['endpoint_error_mm'].mean(0)+v['endpoint_error_mm'].std(0,ddof=1)) for v in stats.values())
    b=panel(s,920,205,610,300,(1,120),(0,math.ceil(top/5)*5),'B  终点误差','试次','距离误差（mm）；800 ms',[1,20,60,100,120],list(np.arange(0,math.ceil(top/5)*5+1,5)),True)
    for label,color in COLORS.items():
        for mapper,metric in [(a,'early_error_deg'),(b,'endpoint_error_mm')]:
            data=stats[label][metric];mean=data.mean(0);sd=data.std(0,ddof=1);t=np.arange(1,121)
            band(s,mapper,t,mean,sd,color);s.line([mapper(x,y) for x,y in zip(t,mean)],color,3,label in ('feedback','original_joint'))
    for x,label in zip([100,385,670,990,1290],COLORS):
        s.line([(x,600),(x+40,600)],COLORS[label],3,label in ('feedback','original_joint'));s.text(x+52,608,LABELS[label],22)
    s.text(95,693,'末期10次验证均值±SD',26,bold=True)
    headers=[(95,'条件'),(420,'ηC／ηCb'),(735,'起步绝对误差（°）'),(1100,'终点误差（mm）'),(1400,'目标分数')]
    for x,t in headers:s.text(x,744,t,21,bold=True)
    for i,label in enumerate(COLORS):
        r=summary[label];yy=800+i*48;s.text(95,yy,LABELS[label],23,COLORS[label],bold=True)
        s.text(420,yy,f'{r["eta_c"]:.5g}／{r["eta_cb"]:.5g}',21)
        for x,key in [(735,'late_abs_early_deg'),(1100,'late_endpoint_mm'),(1400,'objective')]:
            v=r[key];s.text(x,yy,f'{v["mean"]:.2f} ± {v["sd"]:.2f}',21)
    s.text(95,1060,'独立验证仅更换网络初始化，仍为同一目标、旋转幅度与延迟；不能据此宣称跨任务最优。',21,MUTED)
    s.save(out/'tuning_heldout.svg')

def contributions(data,out):
    probes=read_csv(data/'test_probes.csv');seedrows=read_csv(data/'test_per_seed.csv')
    tests=['intact','cb_block','cortex_reset','both_reset'];names=['完整','阻断小脑','撤除皮层增量','两者撤除'];cols=['#e16a25','#147f9a','#8458b2','#647c8f']
    s=SVG(1640,830);s.text(60,57,'联合学习的贡献检验：撤除两处学习作用后发生什么？',32,bold=True)
    s.text(60,99,'新5次初始化 · 第100次更新后的固定检查点 · 探测期间冻结学习 · 误差取逐种子绝对值',22,MUTED)
    for x,field,title,unit in [(100,'early_error_deg','A  起步方向误差','绝对角误差（°）'),(660,'endpoint_error_mm','B  终点距离误差','距离误差（mm）')]:
        vals=[np.array([float(r[field]) for r in probes if r['condition']=='tuned_joint' and r['test']==t]) for t in tests]
        if field=='early_error_deg':vals=[np.abs(v) for v in vals]
        high=max(v.mean()+v.std(ddof=1) for v in vals)*1.15;high=math.ceil(high/5)*5
        xy=panel(s,x,235,450,300,(-.5,3.5),(0,high),title,'冻结探测',unit,[],list(np.linspace(0,high,5)))
        for i,(v,col,name) in enumerate(zip(vals,cols,names)):
            m=v.mean();sd=v.std(ddof=1);xx,yy=xy(i,m);_,base=xy(i,0)
            s.rect(xx-27,yy,54,base-yy,col);s.line([(xx,xy(i,max(0,m-sd))[1]),(xx,xy(i,m+sd)[1])],INK,2)
            s.text(xx,xy(i,m+sd)[1]-12,f'{m:.2f}',20,anchor='middle');s.text(xx,base+28,name,18,anchor='middle')
    s.text(1200,198,'C  配对验证目标分数',24,bold=True);s.text(1200,231,'同种子比较；越低越好',20,MUTED)
    seeds=sorted({int(r['seed']) for r in seedrows});values={label:np.array([float(next(r for r in seedrows if r['condition']==label and int(r['seed'])==seed)['objective']) for seed in seeds]) for label in ('tuned_cerebellum','tuned_joint')}
    high=max(v.max() for v in values.values())*1.25;xy=panel(s,1210,305,300,230,(-.5,1.5),(0,high),'','条件','综合目标',[],[round(v,2) for v in np.linspace(0,high,4)])
    for i,seed in enumerate(seeds):
        aa=values['tuned_cerebellum'][i];bb=values['tuned_joint'][i];s.line([xy(0,aa),xy(1,bb)],'#b4c2cc',2)
        s.dot(*xy(0,aa),'#8458b2');s.dot(*xy(1,bb),'#e16a25')
    s.text(xy(0,0)[0],564,'调优小脑',20,anchor='middle');s.text(xy(1,0)[0],564,'调优联合',20,anchor='middle')
    s.text(1210,645,'五条连线对应五次初始化',18,MUTED)
    s.text(60,713,'撤除皮层增量保留基本读出与循环网络；阻断小脑只撤除回传信号，权重仍保留。',22)
    s.text(60,756,'干预作用是当前检查点的指标依赖结果；贡献不按系数比或权重范数相加，也不等同真实脑区能力。',21,MUTED)
    s.save(out/'tuning_contributions.svg')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'results');ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'figures');args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    heatmap(args.data,args.out);heldout(args.data,args.out);contributions(args.data,args.out)

if __name__=='__main__':main()
