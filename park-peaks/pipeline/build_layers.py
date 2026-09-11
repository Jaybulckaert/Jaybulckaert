import cv2, numpy as np, time, json
t0=time.time()
full=cv2.imread('park.jpg'); img=cv2.resize(full,(full.shape[1]//2,full.shape[0]//2),interpolation=cv2.INTER_AREA)
cv2.imwrite('park_half.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,90])
H,W=img.shape[:2]; print('working size',W,H)
h=lambda pts:[(p[0]/2,p[1]/2) for p in pts]          # native -> half
def poly(pts):
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(pts,np.int32)],255); return m
def K(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def grabcut_local(prior, x0,y0,x1,y1, erode, dilate, iters=4, hard_bg=None, lock=None):
    sub=img[y0:y1,x0:x1]; f=prior[y0:y1,x0:x1]
    inner=cv2.erode(f,K(erode)); outer=cv2.dilate(f,K(dilate))
    gc=np.full(f.shape,cv2.GC_BGD,np.uint8)
    gc[outer>0]=cv2.GC_PR_BGD; gc[f>0]=cv2.GC_PR_FGD; gc[inner>0]=cv2.GC_FGD
    if lock is not None: gc[(inner>0)&(lock[y0:y1,x0:x1]==0)]=cv2.GC_PR_FGD
    if hard_bg is not None: gc[hard_bg[y0:y1,x0:x1]>0]=cv2.GC_BGD
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    cv2.grabCut(sub,gc,None,bg,fg,iters,cv2.GC_INIT_WITH_MASK)
    out=prior.copy(); out[y0:y1,x0:x1]=np.where((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    return out

BASE=2540; PIV_Y=2150
# ---------- ridges (native coords) ----------
ridgeL=[(0,1606),(100,1598),(200,1612),(270,1624),(330,1606),(400,1596),(470,1586),(545,1574),(600,1580),(650,1592),(700,1600),(760,1612),
 (820,1630),(860,1643),(900,1651),(950,1676),(1000,1693),(1050,1726),(1100,1760),(1150,1785),(1200,1814),(1250,1843),(1300,1876),(1340,1901),
 (1450,1960),(1550,2000),(1650,2035),(1750,2068),(1900,2093),(2000,2105),(2150,2130),(2300,2160),(2450,2220),(2600,2300)]
ridgeR=[(2400,2250),(2500,2150),(2650,2050),(2800,1950),(2950,1880),(3060,1830),(3150,1800),(3247,1755),(3266,1749),(3310,1743),(3341,1718),
 (3385,1690),(3435,1696),(3485,1677),(3535,1637),(3578,1623),(3622,1637),(3660,1655),(3685,1674),(3747,1724),(3810,1780),(3872,1830),
 (3950,1890),(4100,1990),(4300,2100),(4500,2200),(4700,2300),(4896,2380)]
polyL=poly(h(ridgeL+[(2600,BASE),(0,BASE)])); polyR=poly(h(ridgeR+[(4896,BASE),(2400,BASE)]))
sky=np.zeros((H,W),np.uint8); sky[:int(1500/2)]=255
sky=cv2.bitwise_and(sky,cv2.bitwise_not(cv2.dilate(polyL|polyR,K(6))))

# ---------- foreground prior: vegetation hue + everything low + the person ----------
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); Hh,S,V=hsv[...,0].astype(int),hsv[...,1].astype(int),hsv[...,2].astype(int)
veg=((Hh>=18)&(Hh<=95)&(S>=45)&(V>=25)).astype(np.uint8)*255          # green / yellow foliage
dark=((V<70)&~((Hh>=95)&(Hh<=130)&(S>=40))).astype(np.uint8)*255      # deep shadow that is not blue
prior=cv2.morphologyEx(veg|dark,cv2.MORPH_OPEN,K(2))
prior[:int(1350/2)]=0
prior[int(2230/2):]=255
person=poly(h([(1870,1770),(2000,1745),(2200,1745),(2420,1770),(2560,1850),(2640,2000),(2600,2200),(2560,2400),(2500,2600),(3000,3200),
               (3100,3672),(1050,3672),(1300,3200),(1780,2700),(1820,2400),(1840,2100),(1830,1950)]))
prior=cv2.morphologyEx(prior,cv2.MORPH_CLOSE,K(3))
prior_def=prior.copy()
prior=prior|person            # the person is probable-foreground only; colour decides the exact edge
print('prior built',time.time()-t0,flush=True)

# GrabCut in four vertical strips so the colour models stay local
fg=prior.copy()
y0,y1=int(1350/2),int(2260/2)
for i,(a,b) in enumerate([(0,700),(650,1300),(1250,1900),(1850,W)]):
    part=grabcut_local(prior,a,y0,b,y1,4,16,iters=5,hard_bg=sky,lock=prior_def)
    seam=np.zeros((H,W),np.uint8); seam[:,a:b]=255
    if i: seam[:,a:a+50]=0
    fg=np.where(seam>0,part,fg)
fg[int(2230/2):]=255
# keep components that touch the prior
n,lab,stats,_=cv2.connectedComponentsWithStats(fg,8); keep=np.zeros_like(fg)
for i in range(1,n):
    if stats[i,cv2.CC_STAT_AREA]>=40 and (prior[lab==i]>0).any(): keep[lab==i]=255
fg=cv2.morphologyEx(keep,cv2.MORPH_OPEN,K(1))
print('fg done',time.time()-t0,flush=True)

# ---------- mountain masks: refine ridge against sky ----------
def refine(polym,x0,x1):
    m=grabcut_local(polym,x0,int(1500/2),x1,int(2300/2),6,6,iters=4)
    m=cv2.bitwise_or(m,cv2.erode(polym,K(24))); m=cv2.bitwise_and(m,polym); m[int(BASE/2):]=0
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    m=np.where(lab==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]),255,0).astype(np.uint8)
    return cv2.morphologyEx(m,cv2.MORPH_CLOSE,K(3))
mL=refine(polyL,0,int(2700/2)); mR=refine(polyR,int(2300/2),W)
print('mountain masks',time.time()-t0,flush=True)

guide=img.astype(np.float32)/255
fg_a=np.clip(cv2.ximgproc.guidedFilter(guide,fg.astype(np.float32)/255,4,1e-3),0,1); fg_a[int(2260/2):]=1
np.save('park_fg_a.npy',fg_a); np.save('park_mL.npy',mL); np.save('park_mR.npy',mR)
ov=img.copy().astype(np.float32)
for m,col in ((mL,(40,120,255)),(mR,(60,200,255))):
    a=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),0.8)[...,None]; ov=ov*(1-0.45*a)+0.45*a*np.array(col)
a=fg_a[...,None]; ov=ov*(1-0.45*a)+0.45*a*np.array([80,220,80])
cv2.imwrite('park_overlay.jpg',cv2.resize(ov.astype(np.uint8),(1288,966),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,88])
# zoomed overlay of the tree line for inspection
cv2.imwrite('park_overlay_zoom.jpg',cv2.resize(ov[int(1400/2):int(2300/2)].astype(np.uint8),(2448//2*1,450),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,88])
print('overlay',time.time()-t0)

# ================= plates =================
def build_plate(m, ridge_native, summit_x_native, X0, X1, name):
    """m: plate mask (half res). Fill the parts hidden by the foreground, export RGBA webp."""
    r=np.interp(np.arange(W),[p[0]/2 for p in ridge_native],[p[1]/2 for p in ridge_native])
    y0=max(0,int(np.argmax(m.max(axis=1)>0))-4); y1=int(BASE/2)
    region=img[y0:y1].copy(); mm=m[y0:y1]
    valid=((mm>0)&(fg_a[y0:y1]<0.02)).astype(np.uint8)*255
    hole=(mm>0)&(valid==0)
    rr=r-y0; RH=region.shape[0]
    small=cv2.resize(region,(region.shape[1]//2,RH//2),interpolation=cv2.INTER_AREA)
    smask=cv2.dilate(cv2.resize(hole.astype(np.uint8)*255,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST),K(2))
    up=cv2.resize(cv2.inpaint(small,smask,9,cv2.INPAINT_TELEA),(region.shape[1],RH),interpolation=cv2.INTER_CUBIC)
    rng=np.random.default_rng(5)
    hp=region.astype(np.float32)-cv2.GaussianBlur(region,(0,0),3).astype(np.float32)
    ys,xs=np.where(valid>0); idx=rng.integers(0,len(ys),size=int(hole.sum()))
    filled=up.astype(np.float32); filled[hole]+=hp[ys[idx],xs[idx]]*0.6
    filled=np.clip(filled,0,255).astype(np.uint8); filled[valid>0]=region[valid>0]
    # ridge-relative cloning from the visible rock band [X0,X1] (half-res x)
    hy,hx=np.where(hole); d=hy-rr[hx]
    dmax=np.zeros(W,np.float32)
    for cx in range(W):
        rows=np.where(valid[:,cx]>0)[0]; dmax[cx]=(rows.max()-rr[cx]) if len(rows) else 0
    span=X1-X0
    jx=cv2.GaussianBlur(rng.normal(0,1,(RH,W)).astype(np.float32),(0,0),25); jx=jx/(jx.std()+1e-6)*14
    jd=cv2.GaussianBlur(rng.normal(0,1,(RH,W)).astype(np.float32),(0,0),25); jd=jd/(jd.std()+1e-6)*3
    right=hx>=X1; left=hx<X0
    u=np.where(right,hx-X1,np.where(left,X0-hx,0)).astype(np.float32)+jx[hy,hx]
    u=np.mod(np.abs(u),2*span); u=np.where(u>span,2*span-u,u)
    sx=np.where(left,X0+u,X1-u); sx=np.clip(np.round(sx).astype(int),0,W-1)
    def sample(sx):
        dd=d+jd[hy,hx]; within=dd<=dmax[sx]-2
        sy=np.clip(np.round(rr[sx]+dd).astype(int),0,RH-1); return sy, (valid[sy,sx]>0)&within
    sy,ok=sample(sx)
    sx2=np.clip(sx+np.where(left,61,-61),0,W-1); sy2,ok2=sample(sx2); ok2=(~ok)&ok2
    sx=np.where(ok2,sx2,sx); sy=np.where(ok2,sy2,sy); ok=ok|ok2
    clone=filled.copy(); clone[hy[ok],hx[ok]]=region[sy[ok],sx[ok]]
    cm=np.zeros((RH,W),np.uint8); cm[hy[ok],hx[ok]]=255
    cm=cv2.morphologyEx(cm,cv2.MORPH_OPEN,K(2)); cm=cv2.morphologyEx(cm,cv2.MORPH_CLOSE,K(3))
    lo_t=cv2.GaussianBlur(filled.astype(np.float32),(0,0),16); lo_c=cv2.GaussianBlur(clone.astype(np.float32),(0,0),16)
    xfer=np.clip(0.5*lo_t+0.5*lo_c+(clone.astype(np.float32)-lo_c),0,255).astype(np.uint8)
    clone=np.where(cm[...,None]>0,xfer,clone)
    pad=24
    src=cv2.copyMakeBorder(clone,pad,pad,pad,pad,cv2.BORDER_REPLICATE); dst=cv2.copyMakeBorder(filled,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    msk=cv2.copyMakeBorder(cm,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
    yy,xx=np.where(msk>0)
    if len(yy):
        c=((xx.min()+xx.max())//2,(yy.min()+yy.max())//2)
        try: filled=cv2.seamlessClone(src,dst,msk,c,cv2.NORMAL_CLONE)[pad:-pad,pad:-pad]
        except Exception as e:
            print('seamlessClone failed',e); a=cv2.GaussianBlur(cm.astype(np.float32)/255,(0,0),4)[...,None]
            filled=np.clip(clone*a+filled*(1-a),0,255).astype(np.uint8)
    filled[valid>0]=region[valid>0]
    a=np.clip(cv2.GaussianBlur(mm.astype(np.float32)/255,(0,0),0.9),0,1)
    rgba=np.dstack([filled,(a*255).astype(np.uint8)])
    cv2.imwrite(name,rgba,[cv2.IMWRITE_WEBP_QUALITY,88])
    print(name,'y0',y0,'hole px',int(hole.sum()),'cloned',int((cm>0).sum()),time.time()-t0,flush=True)
    return y0

yL=build_plate(mL,ridgeL,545,50,650,'park_mtn_left.webp')
yR=build_plate(mR,ridgeR,3578,1580,1840,'park_mtn_right.webp')
fy0=max(0,int(np.argmax(fg_a.max(axis=1)>0.01))-2)
cv2.imwrite('park_fg.webp',np.dstack([img[fy0:],(fg_a[fy0:]*255).astype(np.uint8)]),[cv2.IMWRITE_WEBP_QUALITY,88])
cv2.imwrite('park_base.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,86])
json.dump({'W':W,'H':H,'fg_y0':fy0,'plates':[
  {'name':'left','file':'park_mtn_left.webp','y0':yL,'pivot':[545/2,PIV_Y/2],'peak':[545/2,int(np.argmax(mL.max(axis=1)>0))],
   'ridge':[[p[0]/2,p[1]/2] for p in ridgeL if p[0]<=1340]},
  {'name':'right','file':'park_mtn_right.webp','y0':yR,'pivot':[3578/2,PIV_Y/2],'peak':[3578/2,int(np.argmax(mR.max(axis=1)>0))],
   'ridge':[[p[0]/2,p[1]/2] for p in ridgeR if 3060<=p[0]<=3685]}]},open('park_layers.json','w'))
print('plates written',time.time()-t0)
