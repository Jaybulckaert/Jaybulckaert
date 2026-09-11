import cv2, numpy as np, time, json
t0=time.time()
SRC='/root/.claude/uploads/f2cfcbbc-5a6e-5c13-9381-f02af2b44473/6d424e4c-image.jpg'
img=cv2.imread(SRC); H,W=img.shape[:2]
def poly(pts):
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(pts,np.int32)],255); return m
def line(pts,th):
    m=np.zeros((H,W),np.uint8); cv2.polylines(m,[np.array(pts,np.int32)],False,255,th); return m
def K(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def grabcut_local(prior, x0,y0,x1,y1, erode, dilate, iters=4):
    sub=img[y0:y1,x0:x1]; f=prior[y0:y1,x0:x1]
    inner=cv2.erode(f,K(erode)); outer=cv2.dilate(f,K(dilate))
    gc=np.full(f.shape,cv2.GC_BGD,np.uint8)
    gc[outer>0]=cv2.GC_PR_BGD; gc[f>0]=cv2.GC_PR_FGD; gc[inner>0]=cv2.GC_FGD
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    cv2.grabCut(sub,gc,None,bg,fg,iters,cv2.GC_INIT_WITH_MASK)
    out=prior.copy()
    out[y0:y1,x0:x1]=np.where((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    return out

# ---------------- mountain (ridge vs sky) ----------------
BASE=1460; PIVOT=(1245,BASE)
# invented flank behind the spruce: straight line + smooth wander + conifer spikes, so the edge reads as a treed ridge
_rng=np.random.default_rng(3)
_fx=np.arange(1560,W+8,4).astype(np.float32)
_fy=np.interp(_fx,[1560,1620,1700,1800,1900,2000,2100,2200,W],[1165,1195,1235,1290,1340,1385,1420,1440,1450])
_w=cv2.GaussianBlur(_rng.normal(0,1,len(_fx)).astype(np.float32).reshape(1,-1),(0,0),6).ravel(); _w=_w/(_w.std()+1e-6)*5
_sp=np.where(_rng.random(len(_fx))<0.35,-_rng.uniform(3,9,len(_fx)),0).astype(np.float32)
FLANK=[(float(x),float(y+w+s)) for x,y,w,s in zip(_fx,_fy,_w,_sp)]
ridge=[(0,1190),(200,1150),(400,1110),(548,1095),(600,1092),(650,1087),(700,1080),(750,1074),(800,1082),(827,1087),(850,1088),(880,1080),(905,1078),(930,1060),(960,1052),(1000,1044),(1030,1027),(1050,1030),(1080,1017),(1105,1007),(1130,1005),(1160,1005),(1180,1000),(1210,985),
 (1230,975),(1245,965),(1260,972),(1290,980),(1315,1005),(1340,1020),(1370,1050),(1390,1055),(1410,1065),(1430,1080),(1440,1105),
 (1460,1110),(1480,1117),(1500,1127),(1520,1140),(1540,1155),(1560,1165)]+FLANK
mtn_poly=poly(ridge+[(W,BASE),(0,BASE)])
mt=grabcut_local(mtn_poly,0,900,W,1470,12,12)
mt=cv2.bitwise_or(mt,cv2.erode(mtn_poly,K(30)))
mt=cv2.bitwise_and(mt,cv2.dilate(mtn_poly,K(4))); mt[BASE:]=0
n,lab,stats,_=cv2.connectedComponentsWithStats(mt,8)
mt=np.where(lab==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]),255,0).astype(np.uint8)
mt=cv2.morphologyEx(mt,cv2.MORPH_CLOSE,K(3))
print('mountain mask',time.time()-t0,flush=True)

# ---------------- foreground ----------------
town_top=[(0,1020),(85,975),(275,1060),(548,1062),(560,1140),(575,1195),(755,1197),(820,1275),(840,1275),(865,1265),(900,1300),
 (930,1330),(960,1320),(1000,1312),(1040,1305),(1055,1300),(1080,1330),(1100,1325),(1150,1340),(1190,1365),(1230,1360),(1260,1340),
 (1290,1315),(1330,1310),(1350,1295),(1370,1250),(1385,1220),(1405,1190),(1430,1170),(1480,1160)]
town_right=[(2053,1300),(2185,1300),(2270,1300),(2285,1210),(2310,1150),(2340,1120),(2400,1125),(2430,1180),(2445,1240),(2560,1108),(W,1095)]
town=poly(town_top+[(1480,1300),(2053,1300)]+town_right+[(W,H),(0,H)])
town=grabcut_local(town,0,940,W,1430,5,8)
town[1400:]=255
spruce_pts=[(1747,395),(1800,580),(1880,713),(1920,820),(1987,860),(1947,980),(2013,1047),(1987,1180),(2053,1300),
        (1480,1300),(1520,1073),(1540,913),(1507,847),(1560,780),(1600,647),(1640,540),(1693,460)]
spruce=grabcut_local(poly(spruce_pts),1440,370,2090,1330,18,26,iters=5)
n,lab,stats,_=cv2.connectedComponentsWithStats(spruce,8)
spruce=np.where(lab==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]),255,0).astype(np.uint8)
lamp=line([(1905,955),(1950,948),(1985,975),(2000,1030),(2005,1150),(2008,1300)],18)
lamp=grabcut_local(lamp,1880,930,2030,1310,2,6)
conifer=grabcut_local(poly([(1398,1045),(1365,1195),(1430,1195)]),1350,1030,1450,1210,4,8)
pole=line([(2185,1225),(2185,1330)],14)
pole=grabcut_local(pole,2150,1215,2220,1335,2,5)
fg=town|spruce|lamp|conifer|pole
fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,K(1))
print('foreground mask',time.time()-t0,flush=True)

# soft alphas
guide=img.astype(np.float32)/255
fg_a=cv2.ximgproc.guidedFilter(guide,fg.astype(np.float32)/255,4,1e-3)
fg_a=np.clip(fg_a,0,1); fg_a[1420:]=1
mt_a=cv2.GaussianBlur(mt.astype(np.float32)/255,(0,0),0.9); mt_a=np.clip(mt_a,0,1)

# overlay for inspection
ov=img.copy()
ov=(ov*(1-0.45*mt_a[...,None])+0.45*mt_a[...,None]*np.array([40,120,255])).astype(np.uint8)
ov=(ov*(1-0.45*fg_a[...,None])+0.45*fg_a[...,None]*np.array([80,220,80])).astype(np.uint8)
cv2.imwrite('overlay3.jpg',cv2.resize(ov,(1288,966),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,88])
np.save('fg_a.npy',fg_a); np.save('mt_a.npy',mt_a)

# ---------------- mountain layer: fill texture behind the occluders ----------------
y0=int(np.argmax(mt.max(axis=1)>0))-4; y0=max(0,y0)
region=img[y0:BASE].copy()
valid=((mt[y0:BASE]>0)&(fg_a[y0:BASE]<0.02)).astype(np.uint8)*255
hole=(mt[y0:BASE]>0)&(valid==0)
print('fill px',int(hole.sum()),flush=True)
# --- step 1: Telea diffusion everywhere (structure + the deep strip that becomes forest) ---
small=cv2.resize(region,(region.shape[1]//2,region.shape[0]//2),interpolation=cv2.INTER_AREA)
smask=cv2.dilate(cv2.resize(hole.astype(np.uint8)*255,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST),K(2))
up=cv2.resize(cv2.inpaint(small,smask,9,cv2.INPAINT_TELEA),(region.shape[1],region.shape[0]),interpolation=cv2.INTER_CUBIC)
rng=np.random.default_rng(7)
hp=region.astype(np.float32)-cv2.GaussianBlur(region,(0,0),3).astype(np.float32)
ys,xs=np.where(valid>0); idx=rng.integers(0,len(ys),size=int(hole.sum()))
filled=up.astype(np.float32); filled[hole]+=hp[ys[idx],xs[idx]]*0.6
filled=np.clip(filled,0,255).astype(np.uint8); filled[valid>0]=region[valid>0]
# --- step 2: ridge-relative texture cloning for the flank hidden by the spruce (x >= 1300) ---
rx=[p[0] for p in ridge]; ry=[p[1] for p in ridge]
r=np.interp(np.arange(W),rx,ry)-y0                      # ridge row per column, region coords
RH=region.shape[0]
hy,hx=np.where(hole); sel=hx>=1300; hy,hx=hy[sel],hx[sel]
d=hy-r[hx]                                              # depth below the ridge
X0,X1=1000,1300; span=X1-X0
jx=cv2.GaussianBlur(rng.normal(0,1,(RH,W)).astype(np.float32),(0,0),25); jx=jx/(jx.std()+1e-6)*18
jd=cv2.GaussianBlur(rng.normal(0,1,(RH,W)).astype(np.float32),(0,0),25); jd=jd/(jd.std()+1e-6)*4
u=(hx-X1)+jx[hy,hx]                                     # distance right of the source span, jittered
u=np.mod(u,2*span); u=np.where(u>span,2*span-u,u)       # ping-pong
sx=np.clip(np.round(X1-u).astype(int),0,W-1)
# deepest valid row per source column; deeper requests ping-pong back up the column
dmax=np.zeros(W,np.float32)
for cx in range(W):
    rows=np.where(valid[:,cx]>0)[0]
    dmax[cx]=(rows.max()-r[cx]) if len(rows) else 0
dm=np.maximum(dmax[sx],8)
dd=np.mod(d+jd[hy,hx],2*dm); dd=np.where(dd>dm,2*dm-dd,dd)
sy=np.clip(np.round(r[sx]+dd).astype(int),0,RH-1)
ok=valid[sy,sx]>0
# second chance: shift the span if the first sample lands on an occluder
sx2=np.clip(sx-97,0,W-1); dm2=np.maximum(dmax[sx2],8); dd2=np.mod(d,2*dm2); dd2=np.where(dd2>dm2,2*dm2-dd2,dd2)
sy2=np.clip(np.round(r[sx2]+dd2).astype(int),0,RH-1)
ok2=(~ok)&(valid[sy2,sx2]>0); sx=np.where(ok2,sx2,sx); sy=np.where(ok2,sy2,sy); ok=ok|ok2
clone=filled.copy()
clone[hy[ok],hx[ok]]=region[sy[ok],sx[ok]]
cm=np.zeros((RH,W),np.uint8); cm[hy[ok],hx[ok]]=255
cm=cv2.morphologyEx(cm,cv2.MORPH_OPEN,K(2)); cm=cv2.morphologyEx(cm,cv2.MORPH_CLOSE,K(3))
# texture transfer: cloned high frequencies on top of the diffusion fill's tonal ramp
lo_t=cv2.GaussianBlur(filled.astype(np.float32),(0,0),16)
lo_c=cv2.GaussianBlur(clone.astype(np.float32),(0,0),16)
xfer=np.clip(0.5*lo_t+0.5*lo_c+(clone.astype(np.float32)-lo_c)*1.0,0,255).astype(np.uint8)
clone=np.where(cm[...,None]>0,xfer,clone)
print('cloned px',int((cm>0).sum()),flush=True)
# --- step 3: gradient-domain blend so the cloned patch matches the surrounding tone ---
pad=24
src=cv2.copyMakeBorder(clone,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
dst=cv2.copyMakeBorder(filled,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
msk=cv2.copyMakeBorder(cm,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
yy,xx=np.where(msk>0)
if len(yy):
    x0b,x1b,y0b,y1b=xx.min(),xx.max(),yy.min(),yy.max()
    center=((x0b+x1b)//2,(y0b+y1b)//2)
    try:
        out=cv2.seamlessClone(src,dst,msk,center,cv2.NORMAL_CLONE)
        filled=out[pad:-pad,pad:-pad]
    except Exception as e:
        print('seamlessClone failed',e); a=cv2.GaussianBlur(cm.astype(np.float32)/255,(0,0),4)[...,None]
        filled=np.clip(clone*a+filled*(1-a),0,255).astype(np.uint8)
filled[valid>0]=region[valid>0]
print('fill done',time.time()-t0,flush=True)

mtn_rgba=np.dstack([filled,(mt_a[y0:BASE]*255).astype(np.uint8)])
fy0=int(np.argmax(fg_a.max(axis=1)>0.01))-2; fy0=max(0,fy0)
fg_rgba=np.dstack([img[fy0:],(fg_a[fy0:]*255).astype(np.uint8)])
ok1=cv2.imwrite('mountain.webp',mtn_rgba,[cv2.IMWRITE_WEBP_QUALITY,88])
ok2=cv2.imwrite('foreground.webp',fg_rgba,[cv2.IMWRITE_WEBP_QUALITY,88])
if not (ok1 and ok2):
    cv2.imwrite('mountain.png',mtn_rgba); cv2.imwrite('foreground.png',fg_rgba)
json.dump({'W':W,'H':H,'mtn_y0':y0,'fg_y0':fy0,'pivot':PIVOT,'base':BASE,'peak':[1245,int(np.argmax(mt.max(axis=1)>0))]},open('layers.json','w'))
print('layers written',time.time()-t0, ok1, ok2)
