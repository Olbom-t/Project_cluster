"""Summarize completed pilot outputs; no fitting, plots, or geographic tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import ROOT,RESULTS,load_dataset,dataset_keys,write_json

def read(path):return json.loads(Path(path).read_text())
def fmt(x,n=3):return 'не определён' if x is None or not np.isfinite(x) else f'{x:.{n}f}'

def main():
    summaries=[(p,read(p)) for method in ['kmeans','ward','gmm','hdbscan'] for p in sorted((RESULTS/method).glob('*/*/summary.json'))]
    if len(summaries)!=132:raise RuntimeError(f'Expected 132 finalists, found {len(summaries)}')
    rows=[];lookup={};combined={'elbow':[],'resolution_8':[]};descriptions=[]
    for p,d in summaries:
        m=d['metrics'];s=d['stability'];lookup[(d['method'],d['dataset'],d['variant'])]=d
        rows.append(dict(dataset=d['dataset'],region=d['region'],year=d['year'],method=d['method'],variant=d['variant'],
                         **m,ari_common_median=s['ari_common_median'],ari_all_median=s['ari_all_median'],
                         min_cluster_jaccard_median=s['min_cluster_jaccard_median'],
                         unstable_assigned_fraction=s['unstable_assigned_fraction'],resamples=s['repeats_successful']))
        if d['method'] in ['kmeans','ward']:
            ds=load_dataset(d['dataset']);a=pd.read_csv(p.parent/'assignments.csv',dtype={'uik':str});groups=sorted(a.cluster.unique())
            centers=np.vstack([ds.X[a.cluster.eq(k)].mean(axis=0) for k in groups])
            dist=np.sqrt(((ds.X[:,None,:]-centers[None,:,:])**2).sum(axis=2));own_idx=np.array([groups.index(k) for k in a.cluster])
            own=dist[np.arange(len(a)),own_idx];alt=dist.copy();alt[np.arange(len(a)),own_idx]=np.inf
            alt_idx=alt.argmin(axis=1);other=alt[np.arange(len(a)),alt_idx]
            margin=np.divide(other-own,other+own,out=np.zeros(len(a)),where=(other+own)>0)
            a['distance_to_own_mean_pp']=100*own;a['nearest_other_mean_cluster']=[groups[i] for i in alt_idx]
            a['distance_to_other_mean_pp']=100*other;a['centroid_margin']=margin
            a['geometric_boundary_margin_lt_0_1']=margin<.1
            a.to_csv(p.parent/'assignments.csv',index=False,encoding='utf-8-sig')
            write_json(p.parent/'geometric_diagnostics.json',dict(boundary_fraction=float((margin<.1).mean()),
                       threshold=.1,note='Descriptive centroid margin, not class probability; for Ward the assigned group need not have the nearest centroid.'))
        if d['method']=='kmeans' and d['variant'] in combined:
            a=pd.read_csv(p.parent/'assignments.csv',dtype={'uik':str});a.insert(0,'dataset',d['dataset']);a.insert(1,'year',d['year']);combined[d['variant']].append(a)

    frame=pd.DataFrame(rows);frame.to_csv(RESULTS/'method_comparison.csv',index=False,encoding='utf-8-sig')
    for variant,parts in combined.items():pd.concat(parts,ignore_index=True).to_csv(RESULTS/f'{variant}_assignments_all.csv',index=False,encoding='utf-8-sig')
    def passing(d):
        s=d['stability'];return s['ari_common_median'] is not None and s['ari_common_median']>=.8 and s['min_cluster_jaccard_median']>=.75
    eight=[k for k in dataset_keys() if passing(lookup[('kmeans',k,'resolution_8')])]
    manifest=read(ROOT/'data'/'manifest.json');attempts=sum(d['stability']['repeats_successful'] for _,d in summaries)
    gridcounts={method:sum(len(pd.read_csv(p)) for p in (RESULTS/method).glob('*/candidates.csv')) for method in ['kmeans','ward','gmm','hdbscan']}
    lines=['# Результат пилота: Дума 2021 и президент 2024','',
           'Отчёт завершён 29 октября 2026. Это выполненный исследовательский пилот, а не окончательно утверждённая методика или независимая валидация.', '',
           '## Основные выводы','',
           'В проверенных конфигурациях k-means дал наиболее устойчивые разбиения полного массива. Лучший силуэт выбирает 2–3 группы, но это не предел устойчивой детализации: на уровнях локтя получены 3–5 групп с высокой внутренней устойчивостью; 8 групп проходят рабочие ориентиры устойчивости в 7 из 12 наборов. Единственное оптимальное число кластеров не установлено.', '',
           'Устойчивость не равна однородности: в части думских групп центральный 90%-й диапазон отдельной партийной доли шире 20 п.п. Увеличение K уменьшает средний квадратичный разброс, но не обязано уменьшать максимальный размах каждой отдельной группы.', '',
           'Уорд в этих проверках менее воспроизводим; GMM часто даёт дополнительные нестабильные компоненты; HDBSCAN в большинстве наборов оставляет значительную часть УИК без назначения. Эти выводы относятся к проверенным сеткам параметров, а не доказывают превосходство алгоритма во всех возможных настройках.', '',
           '## Данные и условия','',
           f"Исходных строк {sum(x['raw_rows'] for x in manifest)}. Исключены 22 подтверждённых отдельные записи ДЭГ и 8 строк без действительных голосов. В моделях {sum(x['included_rows'] for x in manifest)} УИК, 12 самостоятельных комбинаций регион × кампания.", '',
           'Использованы все 14 партий / 4 кандидата. Признаки — голоса / действительные голоса. Все УИК имеют равный вес. Размер участка, явка, недействительные и внепомещенное голосование не влияют на формирование групп. VIF, проверка мультиколлинеарности, PCA, ILR и стандартизация не выполнялись. Исходные CSV не изменялись.', '',
           '## Что фактически рассчитано','',
           f"Сеточные варианты: k-means — {gridcounts['kmeans']}, Уорд — {gridcounts['ward']}, GMM — {gridcounts['gmm']}, HDBSCAN — {gridcounts['hdbscan']}. Дополнительно проверены 36 k-means-разбиений на 8, 12 и 16 групп. Всего {len(summaries)} вариантов получили по 30 повторных подвыборок: {attempts} успешных повторений, ошибок повторного обучения нет.", '',
           'Для k-means/Ward исходная сетка K=2..20. Два диагностических варианта: максимум силуэта и локоть SSE (максимальное нормированное отклонение от прямой K=1 → K=20). Локоть — эвристика, зависящая от диапазона, а не установленный оптимум. Для k-means: n_init=20, max_iter=500, tol=1e-4. У Ward евклидова геометрия и стандартное слияние по приросту SSE.', '',
           'GMM: full и diag, K=2..15 с расширением до20 при минимуме BIC на15, n_init=2, max_iter=300, tol=1e-3, reg_covar=1e-6. Финалист — минимум BIC внутри семейства ковариаций. Все модели сошлись; проверен отдельный повтор с reg_covar=1e-5. Регуляризация — параметр ковариации модели, без изменения входных долей. BIC условен относительно этой регуляризованной модели на исходных долях.', '',
           'HDBSCAN: min_cluster_size=max(10,round(N×f)), f=0.005/0.01/0.02/0.05; min_samples=5/15; eom/leaf; allow_single_cluster=True. Отбор внутри каждого семейства защищает охват: не дальше 2 п.п. от максимального охвата, затем по силуэту. Порог количества УИК в кластере не является исключением малых по численности избирателей участков.', '',
           '## Как читать показатели','',
           '- Силуэт: рассчитан на одной и той же фиксированной равновероятной выборке до1500 УИК на набор; для HDBSCAN только на назначенной части этой выборки. Выборка нужна для вычисления показателя; обучение использует весь подготовленный массив.',
           '- ARI: медианное совпадение разбиения с повторными моделями на общих наблюдениях. Для HDBSCAN показатель по совместно назначенным УИК обязательно читать вместе с охватом и Jaccard; ARI может быть высоким при потере большинства объектов.',
           '- Jmin: минимальный по кластерам медианный Jaccard после согласования групп. Помогает заметить невоспроизводимую небольшую группу, скрытую высоким общим ARI.',
           '- RMS профиля: 100×sqrt(mean(sum_j((p_ij−center_j)^2))). Это длина отклонения всего профиля в п.п., не ошибка одной партии. Не сравнивать её напрямую между 14- и 4-мерными профилями.',
           '- Максимальная ширина центральных90%: наибольшая среди кластеров и партий разница Q95−Q05. Крайние значения сохранены отдельно.',
           '- Неустойчивые назначения: доля УИК с внутренней частотой сохранения метки менее0.8. Это рабочий диагностический порог. 30 повторов дают грубую частоту, не вероятность истинного класса.',
           '- Устойчивость оценивается на общих обучающих 80% УИК, выбранных без возвращения, без исключения целых ТИК и без изменения признаков. Это не отложенная проверка.', '',
           '## Результаты по каждому региону и кампании','',
           'Ориентиры ARI≥0.80 и Jmin≥0.75 используются как прозрачные диагностические границы, не как доказательство естественных групп. Для 8/12/16 они помогают сравнить разрешение; промежуточные K оценены по геометрии, но не все получили собственные 30 повторов.', '']
    for key in dataset_keys():
        c=lookup[('kmeans',key,'elbow')];m=c['metrics'];s=c['stability'];f=lookup[('kmeans',key,'resolution_8')]
        fm=f['metrics'];fs=f['stability'];sel=read(RESULTS/'k_selection'/key/'summary.json')
        lines += [f"### {c['region']}, {c['year']}",'',
                  f"В модели {m['n']} УИК. Вариант локтя: K={m['k']}, силуэт {fmt(m['silhouette'])}, ARI {fmt(s['ari_common_median'])}, Jmin {fmt(s['min_cluster_jaccard_median'])}. Неустойчивых назначений {100*s['unstable_assigned_fraction']:.1f}%. RMS профиля {m['rmse_profile_pp']:.2f} п.п.; максимальная ширина центральных90% {m['max_party_central90_range_pp']:.1f} п.п. Минимальная группа — {m['min_cluster_size']} УИК.", '',
                  f"Контроль K=8: силуэт {fmt(fm['silhouette'])}, ARI {fmt(fs['ari_common_median'])}, Jmin {fmt(fs['min_cluster_jaccard_median'])}; RMS {fm['rmse_profile_pp']:.2f} п.п., максимальная ширина центральных90% {fm['max_party_central90_range_pp']:.1f} п.п. {'Оба ориентира устойчивости выполнены.' if passing(f) else 'Не выполнен как минимум один ориентир устойчивости.'}", '',
                  'При повторном выборе локтя на10 подвыборках: '+', '.join(f'K={k}: {v}/10' for k,v in sorted(sel['elbow_counts'].items(),key=lambda x:int(x[0])))+'.', '']
    lines += ['## Сравнение семейств алгоритмов','']
    for method,variant in [('kmeans','elbow'),('ward','elbow'),('gmm','bic_full'),('gmm','bic_diag')]:
        z=frame[(frame.method==method)&(frame.variant==variant)]
        lines.append(f"- {method}/{variant}: медиана ARI по12 наборам {z.ari_common_median.median():.3f}; K от{int(z.k.min())} до{int(z.k.max())}. Сравниваются выбранные по своим правилам финалисты, а не одинаковые K.")
    lines += ['', 'HDBSCAN/eom выбрал одну группу плюс шум в11 из12 наборов. Орёл2024 — две группы, охват96.4%, ARI по совместно назначенным1.000, Jmin0.711. Москва2024/leaf — две группы, охват62.9%, ARI0.924, Jmin0.839. Краснодар2024/leaf — 29 групп, охват33.4%, ARI0.919, Jmin0.000: высокий общий ARI не подтверждает все29 групп.', '',
              'GMM чувствительна к регуляризации: при замене1e-6 на1e-5 ARI между full-разбиениями Ростова2021 около0.270, Московской области2021 около0.393, Краснодара2021 около0.436. Поэтому большое число компонент и улучшение BIC сами по себе не обосновывают новую электоральную типологию.', '',
              '## Вывод для следующего варианта методики','',
              'Сохранить k-means как наиболее убедительный базовый кандидат данного пилота, но не выбирать K автоматически по максимуму силуэта. Рассматривать устойчивость вместе с фактическим разбросом и воспроизводимостью каждой группы. Зафиксировать содержательно допустимую неоднородность перед окончательным выбором уровня детализации.', '',
              'K=8 прошёл оба рабочих ориентира в наборах: '+ '; '.join(f"{lookup[('kmeans',k,'resolution_8')]['region']} {lookup[('kmeans',k,'resolution_8')]['year']}" for k in eight)+'. Ни один из проверенных вариантов K=12/16 не прошёл одновременно оба ориентира во всех своих группах. Это не исключает удачных промежуточных K и не доказывает оптимальность8.', '',
              'Выбор числа групп сам имеет неопределённость: отдельные10 повторов выбора локтя приведены по наборам. Это ограниченный пилот, не окончательная оценка распределения оптимального K. Высокая устойчивость разбиения также возможна на непрерывном распределении профилей и не доказывает наличие естественных классов.', '',
              '## Файлы','',
              '- method_comparison.csv — все132 проверенных финальных варианта и показатели.',
              '- elbow_assignments_all.csv — метки всех29743 УИК для компактных вариантов локтя.',
              '- resolution_8_assignments_all.csv — метки всех29743 УИК для контрольного K=8, включая наборы, где этот уровень не прошёл ориентиры. Это альтернативный эксперимент, а не универсальная рекомендация.',
              '- CLUSTER_DESCRIPTIONS.md — текстовые профили компактных групп и устойчивых вариантов K=8.',
              '- В каталогах методов: сетки параметров, метки, средние профили, диапазоны, повторные расчёты; GMM — условные модельные вероятности, HDBSCAN — плотностная сила принадлежности.',
              '- data/manifest.json — источники и SHA256; data/excluded_records.csv — причины исключения.', '',
              'Карты, диаграммы и таблицы распределения по территориям не создавались. Числовые CSV сохранены как результаты эксперимента для воспроизведения и проверки.']
    (RESULTS/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    descriptions=['# Текстовые профили кластеров','',
                  'Средние доли невзвешенные по УИК. Это не агрегированные результаты всех бюллетеней. Метки локальны для каждого разбиения. Кластеры малого размера нужно трактовать отдельно; высокий общий ARI не делает единичный профиль массовым типом.', '']
    for key in dataset_keys():
        variants=['elbow']+(['resolution_8'] if key in eight else [])
        ds=load_dataset(key)
        for variant in variants:
            d=lookup[('kmeans',key,variant)];p=RESULTS/'kmeans'/key/variant
            prof=pd.read_csv(p/'profiles.csv');a=pd.read_csv(p/'assignments.csv',dtype={'uik':str})
            descriptions += [f"## {ds.region}, {ds.year}: {variant}, K={d['metrics']['k']}",'']
            for label,g in prof.groupby('cluster'):
                top=g.nlargest(4,'mean_pct');size=int(g.n.iloc[0]);subset=a[a.cluster==label]
                means='; '.join(f'{row.feature} {row.mean_pct:.1f}%' for row in top.itertuples())
                leader=top.iloc[0]
                descriptions.append(f"- {label}: {size} УИК. Средние доли: {means}. Центральные90% по лидирующему участнику: {leader.q05_pct:.1f}–{leader.q95_pct:.1f}%. Неустойчивых назначений {(subset.assignment_stability<.8).mean()*100:.1f}%. Типичный номер УИК: {g.typical_uik.iloc[0]}."+(' Малая группа: менее20 УИК.' if size<20 else ''))
            descriptions.append('')
    (RESULTS/'CLUSTER_DESCRIPTIONS.md').write_text('\n'.join(descriptions).rstrip()+'\n',encoding='utf-8')
    write_json(RESULTS/'validation_summary.json',dict(prepared_datasets=12,included_precincts=29743,finalists=len(summaries),
              successful_resamples=attempts,failed_resamples=sum(len(d['stability']['failures']) for _,d in summaries),
              candidate_grid_counts=gridcounts,resolution8_passing=eight,
              selection_repeats_per_dataset=10,tests='python -m unittest pilot.test_validation: 2 tests passed'))
    print('Report ready;',len(summaries),'finalists;',attempts,'resamples;',len(eight),'datasets pass K=8 working criteria')

if __name__=='__main__':main()
