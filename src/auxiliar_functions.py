"""Funciones reutilizables para leer totalizadores y controlar su calidad.
No ejecuta análisis al importar, no modifica los datos ni crea archivos de auditoría.
"""
import hashlib, zipfile, math, re, statistics as st
from datetime import date, timedelta
from collections import defaultdict, Counter
from pathlib import Path
from lxml import etree as ET
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
SEMILLA = 42
FIN_TRAIN = date(2022,11,12)
CORTE = pd.Timestamp(FIN_TRAIN)
CAL = pd.Timestamp('2022-03-01')
BASE = date(1899,12,30)
H23, CIERRE = 23*3600, 86340
ESPERADAS = set(range(0,86400,3600)) | {CIERRE}
HORAS = sorted(ESPERADAS)
DUR = [(b-a)/3600 for a,b in zip(HORAS,HORAS[1:])]
HOJAS = {'Consolidado EE':'EE','Consolidado Produccion':'Produccion','Totalizadores Energia':'Mycom','Consolidado Agua':'Agua','Consolidado GasVapor':'GasVapor','Consolidado Aire':'Aire','Consolidado KPI':'KPI'}
KPI={'EE Planta / Hl','Agua Planta / Hl','ET Planta / Hl','Aire Planta / Hl','Meta EE Planta','Meta Agua Planta','Meta ET Planta','Meta Aire Planta'}

def huella(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def hora(v):
    try:
        if ':' in v:
            h,m,s=map(float,v.split(':')); n=round(h*3600+m*60+s)
        else: n=round(float(v)*86400)
        if not 0<=n<86400:return None
        return n-1 if n-1 in ESPERADAS else n
    except (ValueError,TypeError):return None

def fecha(v):
    try:
        d=float(v)
        return BASE+timedelta(days=int(d)) if d==int(d) and 40000<=d<=60000 else None
    except (ValueError,TypeError):
        try:return date.fromisoformat(v[:10])
        except (ValueError,TypeError):return None

def leer_temporal(z,ruta,strings):
    def val(c):
        v=c.find('{*}v');t=v.text if v is not None else None
        if c.get('t')=='s' and t is not None:return strings[int(t)]
        if c.get('t')=='inlineStr':return ''.join(c.xpath('.//*[local-name()="t"]/text()'))
        return t
    with z.open(ruta) as f:
        for _,r in ET.iterparse(f,events=('end',),tag='{*}row'):
            i=int(r.get('r'));ab={}
            for c in r:
                ref=c.get('r','')
                if ref in (f'A{i}',f'B{i}'):ab[ref[0]]=val(c)
                else:break
            if i==1:
                if ab.get('A')!='DIA' or ab.get('B')!='HORA':raise ValueError(f'Encabezados inesperados {ruta}: {ab}')
            else:yield i,fecha(ab.get('A')),hora(ab.get('B'))
            r.clear()
            while r.getprevious() is not None:del r.getparent()[0]

def segmentar(rows):
    grupos=[]; b=[]; retro=[]; invalidas=0
    for i,d,h in rows:
        if d is None or h is None:invalidas+=1;continue
        if b:
            p=b[-1]
            if d!=p[1] or p[2]-h>=43200:grupos.append(b);b=[]
            elif h<p[2]:retro.append((i,d,p[2],h))
        b.append((i,d,h))
    if b:grupos.append(b)
    return grupos,retro,invalidas

def cierres(z,ruta,strings,req):
    out={};col=None;ultimo=max(req,default=1)
    with z.open(ruta) as f:
        for _,r in ET.iterparse(f,events=('end',),tag='{*}row'):
            i=int(r.get('r'))
            if i>ultimo:break
            if i==1:
                for c in r:
                    v=c.findtext('{*}v')
                    if c.get('t')=='s' and v is not None and strings[int(v)]=='Frio (Kw)':col=c.get('r')[:-1];break
                if col is None:raise ValueError('Falta Frio (Kw)')
            elif i in req:
                for c in r:
                    if c.get('r')==f'{col}{i}':
                        try:
                            v=float(c.findtext('{*}v'))
                            if math.isfinite(v):out[i]=v
                        except (TypeError,ValueError):pass
                        break
            r.clear()
            while r.getprevious() is not None:del r.getparent()[0]
    return out

def leer_archivo(p):
    with zipfile.ZipFile(p) as z:
        strings=[''.join(x.xpath('.//*[local-name()="t"]/text()')) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        rutas={s.get('name'):rels[s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')] for s in ET.fromstring(z.read('xl/workbook.xml')).findall('.//{*}sheet')}
        t=rutas['Consolidado EE']; ruta=t.lstrip('/') if t.startswith('/') else 'xl/'+t
        grupos,_,_=segmentar(leer_temporal(z,ruta,strings)); inicio=grupos[0][0][1]
        elegidos=[(j+1,b,b[0][1],inicio+timedelta(days=j)) for j,b in enumerate(grupos) if max(b[0][1],inicio+timedelta(days=j))<=FIN_TRAIN]
        req={i:None for _,b,_,_ in elegidos for i,_,h in b}
        vals=cierres(z,ruta,strings,req)
        for j,b,orig,cand in elegidos:
            yield p.name,j,orig,cand,[(i,h,vals.get(i)) for i,_,h in b]

def clasificar(rows):
    horas=[h for _,h,_ in rows]; vs=[v for _,_,v in rows]; d=dict(zip(horas,vs)); dif=[b-a for a,b in zip(vs,vs[1:]) if a is not None and b is not None]
    completo=len(horas)==25 and set(horas)==ESPERADAS
    prefijo=len(horas)==24 and set(horas)==ESPERADAS-{CIERRE}
    sin_nulos=all(v is not None for v in vs)
    fisico=sin_nulos and d.get(0)==0 and min(vs)>=0 and all(x>=0 for x in dif) and vs[-1]>0
    return d,dict(completo=completo,prefijo_sin_cierre=prefijo,nulos=sum(v is None for v in vs),inicio=d.get(0),cierre=d.get(CIERRE),valor_23=d.get(H23),descensos=sum(x<0 for x in dif),max_incremento=max(dif,default=None),min_incremento=min(dif,default=None),valido_completo=completo and fisico,prefijo_valido=prefijo and fisico)

def incluir(h,c):
    if not c or c in {'DIA','HORA','Id','Hl de Mosto Copia'} or c.startswith('Unnamed'):return False
    if h=='EE':return c.endswith('(Kw)') and c!='Frio (Kw)'
    if h=='Energia':return c.lower().startswith('kw')
    if h=='Mycom':return c in {f'KW Mycom {i}' for i in range(1,8)}
    if h=='KPI':return c in KPI
    return True

def leer(z,ruta,strings,filas,pref):
    cols={}; valores={}; ultimo=max(filas,default=1)
    def txt(c):
        v=c.findtext('{*}v');t=c.get('t')
        if t=='s' and v is not None:return strings[int(v)]
        if t=='inlineStr':return ''.join(c.xpath('.//*[local-name()="t"]/text()'))
        return v
    with z.open(ruta) as f:
        for _,r in ET.iterparse(f,events=('end',),tag='{*}row'):
            i=int(r.get('r'))
            if i>ultimo:break
            if i==1:
                for c in r:
                    n=(txt(c) or '').strip()
                    if incluir(pref,n):cols[re.sub(r'\d+$','',c.get('r'))]=f'{pref} | {n}'
                assert len(set(cols.values()))==len(cols)
            elif i in filas:
                vv={};h=None
                for c in r:
                    k=re.sub(r'\d+$','',c.get('r'))
                    if k=='B':h=hora(txt(c))
                    if k in cols:
                        raw=txt(c);status='vacio' if raw is None or not str(raw).strip() else 'no_numerico'
                        try:
                            v=float(raw)
                            if np.isfinite(v):vv[cols[k]]=(v,'observado');continue
                        except (TypeError,ValueError):pass
                        vv[cols[k]]=(np.nan,status)
                if h!=86340:raise ValueError(f'{ruta} fila {i}: no es cierre ({h})')
                valores[filas[i]]={c:vv.get(c,(np.nan,'vacio')) for c in cols.values()}
            r.clear()
            while r.getprevious() is not None:del r.getparent()[0]
    assert len(valores)==len(filas)
    return set(cols.values()),valores

def extraer(data,mapa,refs,q):
    q=q.merge(refs[['fecha_candidata','real']],on='fecha_candidata',validate='many_to_one');q=q[q.valido_completo & q.cierre.eq(q.real)].copy()
    q['prioridad']=q.archivo.str.extract(r'(20\d\d)_20\d\d')[0].astype(int)
    fuentes=q.sort_values(['fecha_candidata','prioridad','archivo']).drop_duplicates('fecha_candidata',keep='last').set_index('fecha_candidata').archivo
    assert len(fuentes)==len(refs)
    idx=set(zip(q.archivo,q.fecha_candidata));mapa=mapa[mapa.apply(lambda r:(r.archivo,r.fecha_candidata) in idx,axis=1)]
    assert (mapa.fecha_original<=CORTE).all() and (mapa.fecha_candidata<=CORTE).all()
    schema={};registros={};completos={}
    for p in sorted(data.glob('Totalizadores Planta*.xlsx')):
        with zipfile.ZipFile(p) as z:
            strings=[''.join(x.xpath('.//*[local-name()="t"]/text()')) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
            rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
            rutas={s.get('name'):rels[s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')] for s in ET.fromstring(z.read('xl/workbook.xml')).findall('.//{*}sheet')}
            for hoja,pref in HOJAS.items():
                sub=mapa[(mapa.archivo==p.name)&(mapa.hoja==hoja)];assert sub.fecha_candidata.is_unique and sub.secuencia_igual_EE.all()
                filas={int(r.fila_fin):r.fecha_candidata for r in sub.itertuples() if r.completo}
                t=rutas[hoja];ruta=t.lstrip('/') if t.startswith('/') else 'xl/'+t
                cols,vals=leer(z,ruta,strings,filas,pref);schema[p.name,pref]=cols
                for r in sub.itertuples():completos[p.name,r.fecha_candidata,pref]=bool(r.completo)
                for d,v in vals.items():registros.setdefault((p.name,d),{}).update(v)
            print('Leídos cierres de referencia:',p.name,flush=True)
    columnas=sorted(set().union(*schema.values()));estados=[];datos=[];real=refs.set_index('fecha_candidata').real
    for d,a in fuentes.items():
        row={'fecha_candidata':d,'archivo':a,'frio_observado':real[d]}
        for c in columnas:
            pref=c.split(' | ')[0];existe=c in schema[a,pref];completo=completos.get((a,d,pref),False)
            v,s=registros.get((a,d),{}).get(c,(np.nan,'vacio')) if existe and completo else (np.nan,'sin_columna' if not existe else 'bloque_incompleto')
            row[c]=v;estados.append((d,a,c,s))
        datos.append(row)
    viejo='GasVapor | Vapor _Vapor_L5 (KG)';nuevo='GasVapor | Vapor_L5 (KG)';alias=[]
    for d in sorted(fuentes.index):
        aa=[(a,v[viejo][0]) for (a,dd),v in registros.items() if dd==d and viejo in v and np.isfinite(v[viejo][0])]
        bb=[(a,v[nuevo][0]) for (a,dd),v in registros.items() if dd==d and nuevo in v and np.isfinite(v[nuevo][0])]
        for a,x in aa:
            for b,y in bb:alias.append((d,a,b,x,y,x-y))
    return pd.DataFrame(datos).set_index('fecha_candidata').sort_index(),pd.DataFrame(estados,columns=['fecha_candidata','archivo','variable','estado']),pd.DataFrame(alias,columns=['fecha_candidata','archivo_antiguo','archivo_nuevo','valor_antiguo','valor_nuevo','diferencia'])

def simular(X):
    raw=X.drop(columns=['archivo','frio_observado']);cal=raw.loc[:CAL];chk=raw.loc[raw.index>CAL]
    cols=cal.columns[(cal.notna().mean()>=.5)&(cal.nunique()>1)];cal=cal[cols];chk=chk[cols]
    lim=10*cal.quantile(.99);lim=lim.where(lim>0,np.inf);cal=cal.mask(cal.gt(lim));chk=chk.mask(chk.gt(lim));med=cal.median();escala=cal.quantile(.75)-cal.quantile(.25);escala=escala.where(escala>0,cal.abs().median()).where(lambda s:s>0,1.)
    cols=cal.columns[med.notna()];cal=cal[cols];chk=chk[cols];med=med[cols];escala=escala[cols]
    knn=KNNImputer(n_neighbors=5).fit((cal-med)/escala);rng=np.random.default_rng(42);results=[]
    for escenario in ['aislado','bloques_7d']:
        for rep in range(5):
            mask=pd.DataFrame(False,index=chk.index,columns=cols)
            for c in cols:
                obs=chk.index[chk[c].notna()]
                if escenario=='aislado':
                    if len(obs):mask.loc[rng.choice(obs,size=max(1,int(.2*len(obs))),replace=False),c]=True
                else:
                    for d in rng.choice(chk.index,size=7,replace=False):mask.loc[(chk.index>=d)&(chk.index<pd.Timestamp(d)+pd.Timedelta(days=7)),c]=True
            mask &= chk.notna();visible=chk.mask(mask);hist=pd.concat([cal,visible]).reindex(pd.date_range(cal.index.min(),chk.index.max()))
            preds={'mediana_calibracion':pd.DataFrame(np.broadcast_to(med.to_numpy(),visible.shape),index=visible.index,columns=cols),'ultimo_7d':hist.shift(1).ffill(limit=6).reindex(chk.index).fillna(med),'mediana_previa_7d':hist.shift(1).rolling(7,min_periods=1).median().reindex(chk.index).fillna(med),'mediana_previa_30d':hist.shift(1).rolling(30,min_periods=1).median().reindex(chk.index).fillna(med),'knn_5':pd.DataFrame(knn.transform((visible-med)/escala),index=chk.index,columns=cols)*escala+med}
            for metodo,pred in preds.items():
                err=(pred-chk).where(mask)
                for c in cols:
                    e=err[c].dropna();n=int(mask[c].sum());assert len(e)==n
                    if n:results.append(dict(escenario=escenario,repeticion=rep,variable=c,metodo=metodo,n=n,suma_abs=e.abs().sum(),suma_error=e.sum(),escala_calibracion=escala[c]))
    det=pd.DataFrame(results);agg=det.groupby(['escenario','variable','metodo'],as_index=False).agg(n=('n','sum'),suma_abs=('suma_abs','sum'),suma_error=('suma_error','sum'),escala_calibracion=('escala_calibracion','first'));agg['MAE']=agg.suma_abs/agg.n;agg['sesgo']=agg.suma_error/agg.n;agg['MAE_normalizado']=agg.MAE/agg.escala_calibracion
    base=agg[agg.metodo.eq('mediana_calibracion')][['escenario','variable','MAE']].rename(columns={'MAE':'MAE_base'});agg=agg.merge(base,on=['escenario','variable']);agg['ratio_vs_mediana']=agg.MAE/agg.MAE_base.replace(0,np.nan)
    res=agg.groupby(['escenario','metodo'],as_index=False).agg(variables=('variable','nunique'),mediana_ratio_vs_mediana=('ratio_vs_mediana','median'),mediana_MAE_normalizado=('MAE_normalizado','median'))
    return det,agg,res,len(cal),len(chk),len(cols)

def percentil(v,p):
    v=sorted(v); i=(len(v)-1)*p; j=int(i)
    return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(i-j)

def tasas(v):
    return [(b-a)/h for a,b,h in zip(v,v[1:],DUR)]

def detectar(v, piso):
    r=tasas(v); limite=max(piso, 6*st.median(abs(x) for x in r))
    return [i+1 for i,x in enumerate(r) if abs(x)>limite], limite

def reparar(v, j, metodo):
    r=tasas(v); i=j-1
    otros=[x for k,x in enumerate(r) if k!=i and x>=0]
    vecinos=[x for k,x in enumerate(r) if k!=i and abs(k-i)<=3 and x>=0]
    ritmo=st.median(vecinos or otros) if metodo=='mediana_local' else st.median(otros)
    offset=(v[j]-v[j-1])-ritmo*DUR[i]
    return [x-offset if k>=j else x for k,x in enumerate(v)]

def metricas(rows,keys):
    out=[]
    for grupo in sorted({tuple(x[k] for k in keys) for x in rows}):
        rr=[x for x in rows if tuple(x[k] for k in keys)==grupo]
        d=dict(zip(keys,grupo)); d['n']=len(rr)
        d['deteccion_exacta']=sum(x['deteccion_exacta'] for x in rr)
        d['tasa_deteccion_exacta']=d['deteccion_exacta']/len(rr)
        for k in ['sin_corregir','mediana_local','mediana_diaria']:
            e=[x[k]-x['real'] for x in rr];d['MAE_'+k]=st.mean(abs(x) for x in e);d['sesgo_'+k]=st.mean(e)
        out.append(d)
    return out

def correlaciones(X,y,etapa):
    out=[]
    for c in X:
        t=pd.concat([X[c].rename('x'),y.rename('y')],axis=1).dropna()
        if len(t)>=30 and t.x.nunique()>1 and t.y.nunique()>1:out.append(dict(analisis=etapa,variable=c,n=len(t),pearson=t.x.corr(t.y),spearman=t.x.corr(t.y,method='spearman')))
    return pd.DataFrame(out).sort_values('pearson',key=abs,ascending=False)

def leer_ee_original(a):
    with zipfile.ZipFile(a) as z:
        ss=[''.join(x.xpath('.//*[local-name()="t"]/text()')) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rel={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        hoja=next(s for s in ET.fromstring(z.read('xl/workbook.xml')).findall('.//{*}sheet') if s.get('name')=='Consolidado EE')
        t=rel[hoja.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]; ruta=t.lstrip('/') if t.startswith('/') else 'xl/'+t
        cols={}; filas={}
        def valor(c):
            v=c.findtext('{*}v'); tipo=c.get('t')
            if tipo=='s' and v is not None:return ss[int(v)]
            if tipo=='inlineStr':return ''.join(c.xpath('.//*[local-name()="t"]/text()'))
            if v is None:return None
            if tipo in ('str','e','b'):return v
            try:return float(v)
            except ValueError:return v
        with z.open(ruta) as f:
            for _,r in ET.iterparse(f,events=('end',),tag='{*}row'):
                i=int(r.get('r')); d={}
                for c in r:
                    col=re.sub(r'\d+$','',c.get('r','')); v=valor(c)
                    if i==1 and v is not None:cols[col]=str(v).strip()
                    elif col in cols and v is not None:d[cols[col]]=v
                if i>1 and d:filas[i]=d
                r.clear()
                while r.getprevious() is not None:del r.getparent()[0]
        out=pd.DataFrame.from_dict(filas,orient='index').reindex(index=range(2,max(filas)+1),columns=list(cols.values()))
        out.insert(0,'archivo_origen',a.name);out.insert(1,'fila_excel',out.index)
        return out.reset_index(drop=True)

def energia(p,sub):
    with zipfile.ZipFile(p) as z:
        ss=[''.join(x.xpath('.//*[local-name()="t"]/text()')) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets={s.get('name'):rels[s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')] for s in ET.fromstring(z.read('xl/workbook.xml')).findall('.//{*}sheet')}
        t=sheets['Totalizadores Energia'];ruta=t.lstrip('/') if t.startswith('/') else 'xl/'+t
        req={int(r.fila_fin):r.fecha_candidata for r in sub.itertuples() if r.completo}
        _,vs=leer(z,ruta,ss,req,'Energia')
        return {d:{c:v[0] for c,v in row.items()} for d,row in vs.items()}
