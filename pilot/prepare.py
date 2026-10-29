from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .common import DATA,RESULTS,write_json

COUNTS=['voters','received','early','inside','outside','cancelled','mobile','stationary','invalid','valid','lost','unaccounted']
PARTIES=['КПРФ','Зелёные','ЛДПР','Новые люди','Единая Россия','СР–За правду','Яблоко','Партия роста','РПСС','Коммунисты России','Гражданская платформа','Зелёная альтернатива','Родина','Партия пенсионеров']
CANDIDATES=['Даванков','Путин','Слуцкий','Харитонов']
REGIONS={'Алтайский край':'altai','Краснодарский край':'krasnodar','Москва':'moscow','Московская область':'moscow_oblast','Орловская область':'oryol','Ростовская область':'rostov'}

def main():
    DATA.mkdir(exist_ok=True);audit=[];excluded=[]
    for year,pattern in [(2021,'*_дума.csv'),(2024,'*_президент.csv')]:
        paths=sorted(Path('/workspace/attachments').rglob(pattern))
        if len(paths)!=6:raise ValueError(f'Expected six {year} files, found {len(paths)}')
        for p in paths:
            d=pd.read_csv(p,sep=';',encoding='cp1251',decimal=',');features=PARTIES if year==2021 else CANDIDATES
            if year==2021:d.columns=['region','oik','tik','uik']+COUNTS+features
            else:
                d.columns=['tik','uik']+COUNTS+features+[c+'_pct' for c in features]
                d['region']=p.name.removesuffix('_президент.csv');d['oik']=''
            d['region']=d.region.replace({'город Москва':'Москва','Красндарский край':'Краснодарский край'})
            region=d.region.iloc[0]
            if d.region.nunique()!=1:raise ValueError('Mixed regions')
            key=f'{year}_{REGIONS[region]}'
            d['source_record']=np.arange(1,len(d)+1);d['source_file']=p.name
            d['uik']=d.uik.astype('string').str.replace(r'^УИК\s*№\s*','',regex=True).str.strip()
            if d[COUNTS+features].isna().any().any() or (d[COUNTS+features]<0).any().any():raise ValueError('Bad counts')
            if not d[features].sum(axis=1).eq(d.valid).all():raise ValueError('Party totals inconsistent')
            if d[d.uik.notna()].duplicated(['region','uik']).any():raise ValueError('Duplicate precinct IDs')
            deg=d.tik.str.contains('Дистанционное электронное голосование',case=False,na=False)
            if year==2021 and region=='Москва':deg=deg|d.uik.isin([str(x) for x in range(5001,5016)])
            zero=d.valid.eq(0);drop=deg|zero
            e=d.loc[drop,['region','oik','tik','uik','source_file','source_record']].copy()
            e['year']=year;e['reason']=np.where(deg[drop],'confirmed_separate_DEG','zero_valid_votes');excluded.append(e)
            clean=d.loc[~drop].copy().sort_values(['tik','uik'],kind='stable').reset_index(drop=True)
            X=clean[features].to_numpy(dtype=float)/clean.valid.to_numpy()[:,None]
            if not np.allclose(X.sum(axis=1),1):raise ValueError('Invalid proportions')
            clean['issued']=clean[['early','inside','outside']].sum(axis=1)
            clean['turnout_pct']=100*clean.issued/clean.voters.replace(0,np.nan)
            clean['invalid_pct']=100*clean.invalid/(clean.invalid+clean.valid)
            clean['outside_issued_pct']=100*clean.outside/clean.issued.replace(0,np.nan)
            meta=clean[['region','oik','tik','uik','source_file','source_record','voters','valid','issued','invalid','turnout_pct','invalid_pct','outside_issued_pct']]
            meta.to_csv(DATA/f'{key}.csv',index=False,encoding='utf-8-sig')
            np.savez_compressed(DATA/f'{key}.npz',X=X,features=np.array(features),region=np.array(region),year=year)
            audit.append(dict(dataset=key,region=region,year=year,source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              raw_rows=len(d),included_rows=len(clean),excluded_deg=int(deg.sum()),excluded_zero_valid=int(zero.sum()),
                              features=features,party_sum_checks_passed=True,boxes_checks_passed=bool((d.mobile+d.stationary).eq(d.valid+d.invalid).all())))
    pd.concat(excluded).to_csv(DATA/'excluded_records.csv',index=False,encoding='utf-8-sig')
    write_json(DATA/'manifest.json',audit)
    print('Prepared',len(audit),'datasets;',sum(a['included_rows'] for a in audit),'included rows')

if __name__=='__main__':main()
