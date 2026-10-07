"""Figures for four-arm distributed learning, drawn from recorded data."""
import csv
import json
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'results';OUT=ROOT/'figures'
COLORS=dict(feedback='#607889',cortex='#147f9a',cerebellum='#8458b2',joint='#d77734')
LABELS=dict(feedback='仅反馈',cortex='仅皮层学习',cerebellum='仅小脑学习',joint='联合学习')
INK='#17334b';MUTED='#607889'


class SVG:
    def __init__(self,w,h):
        self.items=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:"Microsoft YaHei",Arial,sans-serif}</style>']
    def text(self,x,y,t,size=22,color=INK,anchor='start',bold=False):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'+(' font-weight="600"' if bold else '')+'>'+escape(str(t))+'</text>')
    def rect(self,x,y,w,h,fill,stroke=None,rx=8):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="2"' if stroke else '')+'/>')
    def line(self,p,color,width=3,dash=False,arrow=False):
        self.items.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+f'" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
        if arrow:
            a,b=np.array(p[-2]),np.array(p[-1]);d=(b-a)/np.linalg.norm(b-a);v=np.array([-d[1],d[0]])
            self.items.append('<polygon points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in [b,b-14*d+6*v,b-14*d-6*v])+f'" fill="{color}"/>')
    def save(self,name):
        (OUT/name).write_text('\n'.join(self.items+['</svg>']),encoding='utf-8')


def panel(s,x,y,w,h,xlim,ylim,title,xlabel,ylabel,xticks,yticks,shade=False):
    def xy(a,b):return x+(a-xlim[0])/(xlim[1]-xlim[0])*w,y+h-(b-ylim[0])/(ylim[1]-ylim[0])*h
    s.text(x,y-48,title,25,bold=True);s.text(x,y-18,ylabel,19,MUTED)
    if shade:
        l,_=xy(20.5,0);r,_=xy(100.5,0);s.rect(l,y,r-l,h,'#f2eef8',rx=0)
    for t in xticks:
        p,_=xy(t,0);s.line([(p,y),(p,y+h)],'#e2e9ee',1);s.text(p,y+h+29,t,18,anchor='middle')
    for t in yticks:
        _,p=xy(0,t);s.line([(x,p),(x+w,p)],'#e2e9ee',1);s.text(x-12,p+6,t,18,anchor='end')
    s.line([(x,y),(x,y+h),(x+w,y+h)],MUTED,1.5);s.text(x+w/2,y+h+65,xlabel,21,anchor='middle')
    return xy


def band(s,xy,x,m,sd,color):
    p=[xy(a,b) for a,b in zip(x,m-sd)]+[xy(a,b) for a,b in zip(x[::-1],(m+sd)[::-1])]
    s.items.append('<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in p)+f'" fill="{color}" opacity="0.13"/>')


def architecture():
    s=SVG(1600,920);blue=COLORS['cortex'];purple=COLORS['cerebellum'];orange=COLORS['joint']
    s.text(60,55,'分布式学习模型：同一回路，分别开放皮层与小脑的输出可塑性',33,bold=True)
    s.text(60,96,'四组共享预训练、感觉误差和试次安排；联合组平分总学习系数。',22,MUTED)
    s.rect(480,135,580,195,'#f4eff9',purple)
    s.text(510,177,'小脑样模块：48维固定特征展开',27,purple,bold=True)
    s.text(510,221,'可塑部位：输出权重 B',24)
    s.text(510,264,'读取目标、时序、计划速度与皮层状态',22)
    s.text(510,306,'学到的调整信号回传皮层输入',23)
    s.rect(480,410,580,190,'#eef6f8',blue)
    s.text(510,453,'M1样模块：64单元储备池式RNN',27,blue,bold=True)
    s.text(510,496,'固定：循环连接、输入连接、基本读出 L₀',22)
    s.text(510,536,'可塑部位：读出层增量 ΔL',24)
    s.text(510,578,'实际读出 L₀＋ΔL → 二维运动速度',23)
    s.rect(55,420,300,165,'#f0f5f8');s.text(80,462,'任务与感觉输入',27,bold=True)
    s.text(80,505,'目标、Go、计划速度',22);s.text(80,550,'延迟的位置误差',22)
    s.rect(1200,410,340,190,'#f0f5f8');s.text(1225,453,'二维“手”与光标',26,bold=True)
    s.text(1225,496,'皮层速度积分 → 位置',22);s.text(1225,537,'光标旋转30°',23);s.text(1225,578,'视觉延迟120 ms',22)
    s.line([(355,500),(480,500)],blue,3,arrow=True);s.line([(1060,500),(1200,500)],blue,3,arrow=True)
    s.text(1130,477,'运动指令',20,anchor='middle')
    s.line([(565,410),(565,330)],blue,3,arrow=True);s.text(547,381,'状态副本',20,blue,anchor='end')
    s.line([(960,330),(960,410)],purple,5,arrow=True);s.text(980,381,'回传信号',21,purple)
    s.rect(480,695,580,100,'#fff4e9',orange)
    s.text(510,735,'共享教学误差：期望速度 − 延迟观测速度',23,orange,bold=True)
    s.text(510,774,'与过去特征配对；试次结束后更新允许的权重',21)
    s.line([(1370,600),(1370,745),(1060,745)],orange,3,arrow=True)
    s.line([(690,695),(690,600)],orange,3,arrow=True)
    s.line([(480,745),(430,745),(430,232),(480,232)],orange,3,arrow=True)
    s.text(60,850,'仅反馈：0／0　　仅皮层：0.8／0　　仅小脑：0／0.8　　联合：0.4／0.4',24,bold=True)
    s.text(60,893,'系数顺序为皮层／小脑。可塑参数量分别为258、96、354；这是功能抽象与局部更新规则。',22,MUTED)
    s.save('distributed_architecture.svg')


def results():
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    with (DATA/'trial_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    seeds=sorted(set(int(r['seed']) for r in rows))
    s=SVG(1640,1080);s.text(60,55,'四组实际结果：皮层能独立学习，联合学习未自动获得优势',32,bold=True)
    s.text(60,95,'总学习系数0.8 · 5次初始化均值±样本标准差 · 阴影为旋转期 · 负值表示反方向过度补偿',22,MUTED)
    a=panel(s,100,205,610,275,(1,120),(-75,60),'A  起步方向误差','试次','有符号误差（°）；100 ms',[1,20,60,100,120],[-60,-30,0,30,60],True)
    b=panel(s,920,205,610,275,(1,120),(0,18),'B  终点距离误差','试次','距离误差（mm）；800 ms',[1,20,60,100,120],[0,5,10,15],True)
    for arm,color in COLORS.items():
        for mapper,metric in [(a,'early_error_deg'),(b,'endpoint_error_mm')]:
            data=np.array([[float(r[metric]) for r in rows if int(r['seed'])==seed and r['arm']==arm] for seed in seeds])
            mean,sd=data.mean(axis=0),data.std(axis=0,ddof=1);t=np.arange(1,121)
            band(s,mapper,t,mean,sd,color);s.line([mapper(x,y) for x,y in zip(t,mean)],color,3,arm=='feedback')
    for x,arm in zip([100,425,810,1195],COLORS):
        s.line([(x,583),(x+42,583)],COLORS[arm],3,arm=='feedback');s.text(x+55,591,LABELS[arm],22)
    c=panel(s,100,715,610,235,(-.6,3.6),(0,36),'C  适应末10次：起步误差大小','学习条件','平均绝对误差（°）',[],[0,10,20,30])
    for i,arm in enumerate(COLORS):
        value=summary[arm]['late_abs_early_deg'];m,sd=value['mean'],value['sd'];x,y=c(i,m);_,base=c(i,0)
        s.rect(x-40,y,80,base-y,COLORS[arm]);s.line([(x,c(i,m-sd)[1]),(x,c(i,m+sd)[1])],INK,2)
        s.text(x,c(i,m+sd)[1]-14,f'{m:.2f}',21,anchor='middle');s.text(x,base+29,LABELS[arm].replace('学习',''),20,anchor='middle')
    d=panel(s,920,715,610,235,(-.6,3.6),(0,19),'D  联合组：学习后干预','仅撤除相应学习作用；反馈保留','终点距离误差（mm）',[],[0,5,10,15])
    for i,(test,label,col) in enumerate([('intact','完整',COLORS['joint']),('cb_block','阻断小脑',COLORS['cortex']),('cortex_reset','重置皮层',COLORS['cerebellum']),('both_reset','两者重置',COLORS['feedback'])]):
        val=summary['joint']['probes'][test]['endpoint_error_mm'];m,sd=val['mean'],val['sd'];x,y=d(i,m);_,base=d(i,0)
        s.rect(x-40,y,80,base-y,col);s.line([(x,d(i,m-sd)[1]),(x,d(i,m+sd)[1])],INK,2)
        s.text(x,d(i,m+sd)[1]-14,f'{m:.2f}',21,anchor='middle');s.text(x,base+29,label,20,anchor='middle')
    s.text(60,1060,'共享总学习系数仍不等于相同有效更新；结果用于检验当前模型分工，不能作为脑区能力排名。',23,MUTED)
    s.save('distributed_results.svg')


def activity():
    params=json.loads((DATA/'parameters.json').read_text(encoding='utf-8'))
    hidden_diff=dict(cb_block=[],cortex_reset=[]);signals=dict(cortex=[],cerebellum=[])
    commands=dict(intact=[],cb_block=[],cortex_reset=[],both_reset=[])
    for seed in params['seeds']:
        z=np.load(DATA/f'activity_joint_seed{seed}.npz');cp=np.load(DATA/f'checkpoint_joint_seed{seed}.npz')
        hh=np.vstack([z['intact__preparation'],z['intact__hidden']])
        for test in hidden_diff:
            other=np.vstack([z[test+'__preparation'],z[test+'__hidden']]);hidden_diff[test].append(np.sqrt(np.mean((hh-other)**2,axis=1)))
        h=z['intact__hidden'];previous=np.vstack([z['intact__preparation'][-1],h[:-1]])
        q=np.column_stack([h,previous,np.ones(len(h))])
        signals['cortex'].append((q@cp['cortical_delta'])[:,1]*125)
        signals['cerebellum'].append(z['intact__cb'][:,1])
        for test in commands:commands[test].append(z[test+'__commands'][:,1])
    s=SVG(1640,650);s.text(60,55,'联合学习后的联系：回传改变皮层状态，皮层学习改变读出映射',30,bold=True)
    s.text(60,95,'同一已学习检查点 · 所有干预均冻结更新 · 五次初始化均值±标准差',22,MUTED)
    a=panel(s,100,205,410,285,(-100,800),(0,.12),'A  相对完整连接的皮层状态差异','相对Go时间（ms）','64单元RMS（任意单位）',[-100,0,400,800],[0,.04,.08,.12])
    b=panel(s,650,205,370,285,(0,800),(-170,110),'B  两处学习的调整信号','时间（ms）','y方向调整（mm/s）',[0,400,800],[-150,-100,-50,0,50,100])
    c=panel(s,1160,205,370,285,(0,800),(-220,100),'C  皮层最终输出','时间（ms）','y方向速度（mm/s）',[0,400,800],[-200,-100,0,100])
    for key,data in hidden_diff.items():
        arr=np.array(data);mean,sd=arr.mean(axis=0),arr.std(axis=0,ddof=1);t=np.arange(-10,80)*10
        col=COLORS['cortex'] if key=='cb_block' else COLORS['cerebellum'];band(s,a,t,mean,sd,col);s.line([a(x,y) for x,y in zip(t,mean)],col,3)
    for key,data in signals.items():
        arr=np.array(data);mean,sd=arr.mean(axis=0),arr.std(axis=0,ddof=1);t=np.arange(80)*10
        col=COLORS[key];band(s,b,t,mean,sd,col);s.line([b(x,y) for x,y in zip(t,mean)],col,3)
    for key,data in commands.items():
        arr=np.array(data);mean,sd=arr.mean(axis=0),arr.std(axis=0,ddof=1);t=np.arange(80)*10
        col={'intact':COLORS['joint'],'cb_block':COLORS['cortex'],'cortex_reset':COLORS['cerebellum'],'both_reset':COLORS['feedback']}[key]
        band(s,c,t,mean,sd,col);s.line([c(x,y) for x,y in zip(t,mean)],col,3,key=='both_reset')
    for mapper,lim in [(a,(0,.12)),(b,(-170,110)),(c,(-220,100))]:s.line([mapper(120,lim[0]),mapper(120,lim[1])],MUTED,1.5,True)
    s.text(60,588,'A：蓝色阻断小脑，紫色重置皮层。B：紫色回传输入，蓝色皮层读出增量，两者不能直接相加归因。',21)
    s.text(60,625,'C：橙色完整，蓝色阻断小脑，紫色重置皮层，灰色两者重置。虚线时标：120 ms反馈延迟。',21)
    s.save('distributed_activity.svg')


def sensitivity():
    with (DATA/'rate_sensitivity.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    s=SVG(1450,620);s.text(65,55,'学习率敏感性：三个系数、全部五次初始化，保留所有结果',31,bold=True)
    s.text(65,96,'单模块使用总系数；联合组各用一半。无事后挑选最佳学习率。',22,MUTED)
    # Derive padded limits from actual means and SDs rather than hide bad runs.
    stats={}
    for arm in ('cortex','cerebellum','joint'):
        stats[arm]={}
        for metric in ('late_abs_early_deg','late_endpoint_mm'):
            values=[]
            for rate in (.4,.8,1.6):
                vv=np.array([float(r[metric]) for r in rows if r['arm']==arm and float(r['rate'])==rate]);values.append((vv.mean(),vv.std(ddof=1)))
            stats[arm][metric]=np.array(values)
    for x,metric,title,unit in [(110,'late_abs_early_deg','A  适应末起步误差','绝对误差（°）'),(865,'late_endpoint_mm','B  适应末终点误差','距离误差（mm）')]:
        maximum=max(float((stats[a][metric][:,0]+stats[a][metric][:,1]).max()) for a in stats)*1.15
        ticks=np.linspace(0,maximum,4);tick_values=[round(float(v),1) for v in ticks]
        mapper=panel(s,x,205,460,275,(0,2),(0,maximum),title,'总学习系数（联合组平分）',unit,[],tick_values)
        for j,rate in enumerate((.4,.8,1.6)):
            px,py=mapper(j,0);s.text(px,py+29,rate,22,anchor='middle')
        for arm,col in [(a,COLORS[a]) for a in stats]:
            means,sds=stats[arm][metric].T
            s.line([mapper(j,m) for j,m in enumerate(means)],col,3)
            for j,(m,sd) in enumerate(zip(means,sds)):
                px,py=mapper(j,m);s.line([(px,mapper(j,m-sd)[1]),(px,mapper(j,m+sd)[1])],col,2);s.rect(px-4,py-4,8,8,col,rx=0)
    for x,arm in zip((110,550,980),stats):
        s.line([(x,588),(x+45,588)],COLORS[arm],3);s.text(x+58,596,LABELS[arm],22)
    s.save('distributed_sensitivity.svg')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();DATA=args.data;OUT=args.out
    OUT.mkdir(exist_ok=True,parents=True)
    architecture();results();activity();sensitivity()
