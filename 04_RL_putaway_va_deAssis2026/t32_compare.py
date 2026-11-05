import json, glob, numpy as np, pandas as pd
from scipy import stats
U='results/'; T='results/'
C=json.load(open('t0.json')); MI0,T0=C['MI0'],C['T0']
J=lambda mi,t,l: l*mi/MI0+(1-l)*t/T0
rl=pd.concat([pd.read_csv(f) for f in glob.glob(U+'rl_main2*.csv')]); bl=pd.read_csv(U+'baselines.csv')
t32=pd.concat([pd.read_csv(f) for f in glob.glob(T+'t32/row_*.csv')])
def holm(ps):
    ps=np.asarray(ps);o=np.argsort(ps);m=len(ps);a=np.empty(m);r=0
    for i,j in enumerate(o): r=max(r,min(1,ps[j]*(m-i)));a[j]=r
    return a
atag=lambda s:'main2' if s==0 else f'main2_s{s}'
rows=[];tests=[]
for l in [0.3,0.4,0.5]:
    g=bl[(bl.policy=='GreedyJ')&np.isclose(bl.lam,l)]; jg=J(g.MI_avg,g.T_avg,l).mean()
    hg=np.load(U+f'picks_GreedyJ_{l}.npz')['hist']; cg=J(hg[:,0],hg[:,2],l)
    for s in range(5):
        a=rl[(rl.tag==atag(s))&np.isclose(rl.lam,l)].iloc[0]; b=t32[(t32.tag==f'T32_s{s}')&np.isclose(t32.lam,l)].iloc[0]
        ja=J(a.MI_avg,a.T_avg,l); jb=J(b.MI_avg,b.T_avg,l)
        ha=np.load(U+f'picks_RL_{atag(s)}_{l}.npz')['hist']; hb=np.load(T+f'picks_RL_T32_s{s}_{l}.npz')['hist']
        ca=J(ha[:,0],ha[:,2],l); cb=J(hb[:,0],hb[:,2],l)
        tests.append((l,s,'A_vs_T32',np.sign(jb-ja),stats.wilcoxon(ca,cb).pvalue))
        tests.append((l,s,'T32_vs_G',np.sign(jg-jb),stats.wilcoxon(cb,cg).pvalue))
        rows.append(dict(lam=l,seed=s,J_G=jg,J_T32=jb,J_A=ja,MI_T32=b.MI_avg,T_T32=b.T_avg,MI_A=a.MI_avg,T_A=a.T_avg,MI_G=g.MI_avg.mean(),T_G=g.T_avg.mean()))
df=pd.DataFrame(rows); adj=holm([t[4] for t in tests])
df['A_tot_hon_T32']=[None]*len(df); df['T32_tot_hon_G']=None
for (l,s,k,sg,p),a in zip(tests,adj):
    i=df.index[(df.lam==l)&(df.seed==s)][0]; v='có' if (a<0.05 and sg>0) else ('NGƯỢC' if a<0.05 else 'ko ý nghĩa')
    df.loc[i,'A_tot_hon_T32' if k=='A_vs_T32' else 'T32_tot_hon_G']=f'{v} (p={a:.1e})'
pd.set_option('display.width',250); print(df.round(4).to_string())
print(df.groupby('lam')[['J_G','J_T32','J_A']].agg(['mean','std']).round(4))
df.to_csv('results/t32/so_sanh_T32.csv',index=False)
