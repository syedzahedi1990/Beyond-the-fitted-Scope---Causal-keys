import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCATIONS = ('box', 'basket', 'shelf', 'drawer', 'cabinet', 'closet')
PERM = dict(zip(LOCATIONS, ('basket', 'box', 'drawer', 'shelf', 'closet', 'cabinet')))
SEEDS = (101, 102, 103)
FAMILIES = ('independent_template', 'choice_order', 'lexical_support', 'distractor_structure', 'joint_shift')
VIEWS = ('direct', 'world', 'other_agent', 'search', 'observed_overwrite', 'unobserved_overwrite', 'irrelevant_object', 'report')
ACCOUNTS = ('belief_edit', 'event_rewrite', 'broad_state_write', 'persistent_direct_override', 'cleared_direct_override', 'global_output_permutation')
PRIMARY_METHODS = ('clean', 'source') + tuple(f'{o}_ts{s}' for o in ('f_star', 'm3', 'pca') for s in SEEDS)
CONSEQUENCE_METHODS = ('clean', 'source', 'natural_d') + PRIMARY_METHODS[2:]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_evidence(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'SOURCE_PROVENANCE.json').read_text())
    names = ('main.json.gz', 'supporting.json.gz', 'mediation.json.gz')
    require(set(manifest['files']) == set(names), 'Complete three-file evidence inventory required')
    result = {}
    for name in names:
        raw = (directory / name).read_bytes()
        pin = manifest['files'][name]
        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'Compressed evidence differs: ' + name)
        plain = gzip.decompress(raw)
        require(len(plain) == pin['uncompressed_bytes'] and digest(plain) == pin['uncompressed_sha256'], 'Uncompressed evidence differs: ' + name)
        result[name.split('.')[0]] = json.loads(plain, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    validate_main(result['main'])
    return result, manifest


def valid_score(score):
    require(isinstance(score, list) and len(score) == 3, 'Saved score must retain candidate/global/mass')
    require(score[0] in LOCATIONS and (score[1] is None or score[1] in LOCATIONS), 'Invalid normalized score label')
    require(type(score[2]) in (int, float) and math.isfinite(score[2]) and 0 <= score[2] <= 1.000001, 'Invalid candidate mass')


def validate_main(data):
    require(data['schema_version'] == 1, 'Unsupported main schema')
    require(data['score_fields'] == ['candidate_prediction', 'global_prediction', 'candidate_mass'], 'Score field order differs')
    require(data['primary_methods'] == list(PRIMARY_METHODS) and data['consequence_methods'] == list(CONSEQUENCE_METHODS), 'Fixed methods differ')
    require(set(data['models']) == {'qwen', 'mistral'}, 'Both models required')
    records = data['primary_records']
    require(len(records) == 1500 and len({r['pair_uid'] for r in records}) == 1500, 'Primary pair coverage differs')
    require(collections.Counter(r['family'] for r in records) == {x: 300 for x in FAMILIES}, 'Five-family balance differs')
    for r in records:
        require(r['base_answer'] in LOCATIONS and r['source_answer'] in LOCATIONS, 'Unknown primary target')
    require(set(data['cores']) == {'discovery', 'checking'}, 'Both consequence panels required')
    all_ids = []
    for split, cores in data['cores'].items():
        require(len(cores) == 120 and len({c['uid'] for c in cores}) == 120, 'Consequence core coverage differs')
        all_ids.extend(c['uid'] for c in cores)
        for core in cores:
            require(all(core[k] in LOCATIONS for k in ('base', 'source', 'initial', 'new', 'distractor_location')), 'Unknown core location')
    require(len(set(all_ids)) == 240, 'Discovery/checking overlap')
    for model, values in data['models'].items():
        require(set(values['primary']) == {'original', 'answer_prefill'}, 'Both interfaces required')
        for rows in values['primary'].values():
            require(len(rows) == 1500, 'Primary row count differs')
            for row in rows:
                require(len(row) == len(PRIMARY_METHODS), 'Primary method count differs')
                for score in row:
                    valid_score(score)
        require(set(values['consequences']) == {'discovery', 'checking'}, 'Missing consequence split')
        for split, panel in values['consequences'].items():
            views = VIEWS[:-1] if split == 'discovery' else VIEWS
            require(panel['views'] == list(views) and len(panel['scores']) == 120, 'Consequence grid differs')
            for row in panel['scores']:
                require(len(row) == len(views), 'Consequence views differ')
                for cell in row:
                    require(len(cell) == len(CONSEQUENCE_METHODS), 'Consequence methods differ')
                    for score in cell:
                        valid_score(score)
    require('reconstructed' in data['models']['qwen']['pca_identity'], 'Qwen reconstruction provenance missing')
    require('exact' in data['models']['mistral']['pca_identity'], 'Mistral exact-initializer provenance missing')


def clean_answer(core, view, location):
    if view == 'other_agent':
        return core['initial']
    if view == 'observed_overwrite':
        return core['new']
    if view == 'irrelevant_object':
        return core['distractor_location']
    return location


def forecast(core, view, objective, account='event_rewrite'):
    require(view in VIEWS and objective in ('f_star', 'm3') and account in ACCOUNTS, 'Unknown symbolic condition')
    target = core['source'] if objective == 'f_star' else PERM[core['source']]
    if account == 'global_output_permutation':
        answer = clean_answer(core, view, core['source'])
        return answer if objective == 'f_star' else PERM[answer]
    if view == 'irrelevant_object':
        return core['distractor_location']
    if view == 'observed_overwrite':
        return target if account == 'persistent_direct_override' else core['new']
    if view == 'unobserved_overwrite':
        return core['base'] if account == 'cleared_direct_override' else target
    if view == 'world':
        return target if account in ('event_rewrite', 'broad_state_write') else core['base']
    if view == 'other_agent':
        return target if account == 'broad_state_write' else core['initial']
    if view in ('search', 'report'):
        return core['base'] if account in ('persistent_direct_override', 'cleared_direct_override') else target
    return target


def paired(left, right, strata=None):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(left.shape == right.shape and left.ndim == 2 and left.shape[1] == 3, 'Whole-case, three-fixed-fit arrays required')
    require(np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite case metric')
    groups = [np.arange(len(left))] if strata is None else [np.flatnonzero(np.asarray(strata) == f) for f in FAMILIES]
    if any(not len(g) for g in groups):
        return {'n_cases':len(left),'status':'NOT_ESTIMABLE_EMPTY_DECLARED_STRATUM','difference':None,'ci95':None}
    delta = left - right
    l = np.mean([left[g].mean(axis=0) for g in groups], axis=0)
    r = np.mean([right[g].mean(axis=0) for g in groups], axis=0)
    d = np.mean([delta[g].mean(axis=0) for g in groups], axis=0)
    rng = np.random.default_rng(2026091301)
    draws = np.zeros(2000)
    for g in groups:
        ids = rng.choice(g, (2000, len(g)), replace=True)
        draws += delta.mean(axis=1)[ids].mean(axis=1) / len(groups)
    return {'n_cases':len(left), 'left_per_fit':l.tolist(), 'right_per_fit':r.tolist(), 'difference_per_fit':d.tolist(), 'left_mean':float(l.mean()), 'right_mean':float(r.mean()), 'difference':float(d.mean()), 'ci95':np.quantile(draws, [.025, .975]).tolist()}


def counts(matrix):
    a = np.asarray(matrix, dtype=bool)
    require(a.ndim == 3 and a.shape[1] == 3, 'Expected case by fit by view')
    return {'n_cases':a.shape[0], 'n_views':a.shape[2], 'per_fit':[{'seed':seed,'correct_answers':int(a[:,k].sum()),'total_answers':int(a.shape[0]*a.shape[2]),'joint_correct':int(a[:,k].all(axis=1).sum()),'view_correct':a[:,k].sum(axis=0).astype(int).tolist()} for k,seed in enumerate(SEEDS)]}


def primary_results(data):
    out = {}
    records = data['primary_records']
    strata = [r['family'] for r in records]
    targets = {'f_star':[r['source_answer'] for r in records], 'm3':[PERM[r['source_answer']] for r in records]}
    for model, values in data['models'].items():
        out[model] = {}
        for fmt, rows in values['primary'].items():
            arrays, methods = {}, {}
            for objective in ('f_star', 'm3'):
                for method in ('learned', 'pca', 'recoded_pca'):
                    candidate, global_, mass = [], [], []
                    for i,row in enumerate(rows):
                        c,g,m = [],[],[]
                        for seed in SEEDS:
                            name = f'{objective if method == "learned" else "pca"}_ts{seed}'
                            score = row[PRIMARY_METHODS.index(name)]
                            cp,gp = score[:2]
                            if method == 'recoded_pca' and objective == 'm3':
                                cp,gp = PERM[cp],PERM.get(gp)
                            c.append(cp == targets[objective][i]);g.append(gp == targets[objective][i]);m.append(score[2])
                        candidate.append(c);global_.append(g);mass.append(m)
                    key = objective + '/' + method
                    arrays[key] = np.asarray(global_,dtype=float)
                    methods[key] = {'candidate_correct_per_fit':np.sum(candidate,axis=0).astype(int).tolist(),'global_correct_per_fit':np.sum(global_,axis=0).astype(int).tolist(),'candidate_mean':float(np.mean(candidate)),'global_mean':float(np.mean(global_)),'candidate_mass_mean':float(np.mean(mass)),'candidate_mass_per_fit':np.mean(mass,axis=0).tolist(),'n_pairs':1500}
            out[model][fmt] = {'methods':methods,'intended_minus_alternative_global':paired(arrays['f_star/learned'],arrays['m3/learned'],strata),'learned_minus_pca':{o:paired(arrays[o+'/learned'],arrays[o+'/pca'],strata) for o in ('f_star','m3')},'learned_minus_recoded_pca':paired(arrays['m3/learned'],arrays['m3/recoded_pca'],strata)}
    return out



def opportunity_results(data):
    records=data['primary_records']
    nonidentity=[r['base_answer'] not in (r['source_answer'],PERM[r['source_answer']]) for r in records]
    out={}
    for fmt in ('original','answer_prefill'):
        for scorer,index in [('candidate',0),('global',1)]:
            masks={}
            for model,v in data['models'].items():
                masks[model]=[bool(nonidentity[i] and row[0][index]==records[i]['base_answer'] and row[1][index]==records[i]['source_answer']) for i,row in enumerate(v['primary'][fmt])]
            masks['shared']=[q and m for q,m in zip(masks['qwen'],masks['mistral'])]
            entries={}
            for population,mask in masks.items():
                ids=[i for i,k in enumerate(mask) if k]
                entry={'n_pairs':len(ids),'selected_pair_uids':[records[i]['pair_uid'] for i in ids],'per_family_n':{f:sum(records[i]['family']==f for i in ids) for f in FAMILIES},'models':{}}
                for model in (('qwen','mistral') if population=='shared' else (population,)):
                    rows=data['models'][model]['primary'][fmt]
                    matrix={o:[[rows[i][PRIMARY_METHODS.index(f'{o}_ts{seed}')][index]==(records[i]['source_answer'] if o=='f_star' else PERM[records[i]['source_answer']]) for seed in SEEDS] for i in ids] for o in ('f_star','m3')}
                    if not ids:
                        metric={'n_cases':0,'status':'NOT_ESTIMABLE_EMPTY_DECLARED_STRATUM','difference':None,'ci95':None}
                    else:metric=paired(matrix['f_star'],matrix['m3'],[records[i]['family'] for i in ids])
                    entry['models'][model]=metric
                entries[population]=entry
            out[fmt+'/'+scorer]=entries
    return {'rule':'Correct clean base and source under the same interface/scorer; both targets unequal base; shared is exact model intersection. No patched outputs used for eligibility.','panels':out}

def consequence_results(data):
    out = {}
    for model, values in data['models'].items():
        out[model] = {}
        for split, panel in values['consequences'].items():
            cores,views = data['cores'][split], panel['views']
            split_result = {}
            for objective in ('f_star','m3'):
                methods,matrices,predictions = {},{},{}
                for method in ('learned','pca','recoded_pca'):
                    preds = []
                    for row in panel['scores']:
                        fits=[]
                        for seed in SEEDS:
                            name=f'{objective if method=="learned" else "pca"}_ts{seed}'
                            p=[cell[CONSEQUENCE_METHODS.index(name)][1] for cell in row]
                            if method=='recoded_pca' and objective=='m3':
                                p=[PERM.get(x) for x in p]
                            fits.append(p)
                        preds.append(fits)
                    truth=[[forecast(c,v,objective) for v in views] for c in cores]
                    matrix=np.asarray([[[p==t for p,t in zip(fit,truth[i])] for fit in row] for i,row in enumerate(preds)])
                    predictions[method]=preds;matrices[method]=matrix
                    entry=counts(matrix);entry['views']=views;entry['consistency']=[]
                    for k,seed in enumerate(SEEDS):
                        counter=collections.Counter({key:0 for key in ('invalid_direct','direct_correct','direct_wrong','all_other_consistent','correct_coherent','correct_inconsistent','wrong_coherent','wrong_inconsistent')})
                        for i,c in enumerate(cores):
                            p=preds[i][k];anchor=p[views.index('direct')]
                            if anchor not in LOCATIONS:
                                counter['invalid_direct']+=1;continue
                            correct=anchor==forecast(c,'direct',objective)
                            coherent=all(p[j]==clean_answer(c,v,anchor) for j,v in enumerate(views) if v!='direct')
                            counter['direct_correct' if correct else 'direct_wrong']+=1
                            counter['all_other_consistent']+=int(coherent)
                            counter[('correct' if correct else 'wrong')+('_coherent' if coherent else '_inconsistent')]+=1
                        entry['consistency'].append({'seed':seed,**dict(counter)})
                    methods[method]=entry
                accounts={}
                for account in ACCOUNTS:
                    target=[[forecast(c,v,objective,account) for v in views] for c in cores]
                    event=[[forecast(c,v,objective) for v in views] for c in cores]
                    mask=np.asarray(target)!=np.asarray(event)
                    perfit=[]
                    for k,seed in enumerate(SEEDS):
                        p=np.asarray([r[k] for r in predictions['learned']],dtype=object)
                        hits=p==np.asarray(target)
                        eventhits=p==np.asarray(event)
                        perfit.append({'seed':seed,'account_hits':int(hits.sum()),'event_hits_on_disagreement':int(eventhits[mask].sum()),'account_hits_on_disagreement':int(hits[mask].sum()),'neither_on_disagreement':int((~eventhits & ~hits)[mask].sum()),'net_event_hits':int(eventhits[mask].sum()-hits[mask].sum())})
                    accounts[account]={'n_disagreement_views':int(mask.sum()),'n_disagreement_stories':int(mask.any(axis=1).sum()),'all_views_denominator':len(cores)*len(views),'per_fit':perfit}
                equivalent=[a for a in ACCOUNTS if all(forecast(c,v,objective)==forecast(c,v,objective,a) for c in cores for v in views)]
                alternatives=[a for a in ACCOUNTS if a not in equivalent]
                opportunity=[all(any(forecast(c,v,objective)!=forecast(c,v,objective,a) for v in views) for a in alternatives) for c in cores]
                comparisons={o:paired(matrices['learned'].mean(axis=2),matrices[o].mean(axis=2)) for o in ('pca','recoded_pca')}
                split_result[objective]={'methods':methods,'account_discrimination':accounts,'common_opportunity_cases':int(sum(opportunity)),'accounts_equivalent_to_event_on_panel':equivalent,'learned_minus_comparators_all_views':comparisons}
            natural={}
            for role in ('clean','source','natural_d'):
                cells={}
                for j,view in enumerate(views):
                    scores=[r[j][CONSEQUENCE_METHODS.index(role)] for r in panel['scores']]
                    targets=[clean_answer(c,view,c['base'] if role=='clean' else c['source'] if role=='source' else PERM[c['source']]) for c in cores]
                    cells[view]=cell_result(scores,targets)
                natural[role]=cells
            split_result['natural_controls']=natural
            out[model][split]=split_result
    return out


def cell_result(scores, targets):
    require(len(scores)==len(targets)>0, 'Complete nonempty cell required')
    for score in scores:
        valid_score(score)
    n=len(scores);masses=np.asarray([s[2] for s in scores]);g=sum(s[1]==t for s,t in zip(scores,targets));c=sum(s[0]==t for s,t in zip(scores,targets))
    masspass=bool(np.median(masses)>=.9 and np.mean(masses>=.5)>=.95)
    return {'n':n,'global_correct':g,'candidate_correct':c,'invalid_global':sum(s[1] is None for s in scores),'mean_candidate_mass':float(masses.mean()),'median_candidate_mass':float(np.median(masses)),'mass_at_least_half':int(np.sum(masses>=.5)),'mass_pass':masspass,'historical_cell_pass':g/n>=.95 and masspass}


def support_results(data):
    out={'breadth':{},'rank':{},'belief':{},'route':{}}
    require(len(data['breadth'])==12, 'All twelve breadth fits required')
    for name,fit in data['breadth'].items():
        require(fit['train_pairs']==1000 and fit['steps']==300, 'Breadth recipe differs')
        cells={}
        for split,n in [('iid',500),('shifted',300)]:
            rows=fit[split];require(len(rows)==n and len({r['pair_uid'] for r in rows})==n, 'Breadth rows differ')
            target=[r['source_answer'] if name.startswith('f_star_') else r.get('trained_wrong_target',r['source_answer']) for r in rows]
            clean=np.asarray([r['base_prediction']==r['base_answer'] and r['source_prediction']==r['source_answer'] for r in rows]); hit=np.asarray([r['patched_prediction']==t for r,t in zip(rows,target)]);inform=np.asarray([r['base_answer']!=t for r,t in zip(rows,target)])
            cells[split]={'n':n,'own_target_correct':int(hit.sum()),'source_target_correct':sum(r['patched_prediction']==r['source_answer'] for r in rows),'own_target_accuracy':float(hit.mean()),'nontrivial_n':int(inform.sum()),'nontrivial_correct':int(hit[inform].sum()),'clean_endpoints_n':int(clean.sum()),'clean_endpoints_correct':int(hit[clean].sum())}
        out['breadth'][name]={'prediction_scope':'Archived candidate-label decisions; not a fresh global-next-token audit.',**cells}
    require(len(data['rank'])==5, 'All five historical rank panels required')
    for name,panel in data['rank'].items():
        cells={}
        for row in panel['rows']:
            key='/'.join(str(row.get(k,'all')) for k in ('rank','basis_group','basis_relation'))
            cell=cells.setdefault(key,{'n':0,'source_transfer_correct':0,'clean_endpoints_n':0,'source_transfer_given_clean':0})
            hit=row['patched_prediction']==row['source_answer'];clean=row['base_prediction']==row['base_answer'] and row['source_prediction']==row['source_answer']
            cell['n']+=1;cell['source_transfer_correct']+=int(hit);cell['clean_endpoints_n']+=int(clean);cell['source_transfer_given_clean']+=int(hit and clean)
        out['rank'][name]={'train_pairs':panel['train_pairs'],'eval_pairs':panel['eval_pairs'],'cells':cells}
    for name,panel in data['belief'].items():
        rows=panel['rows'];require(len(rows)==60 and len({r['core']['uid'] for r in rows})==60,'Belief core coverage differs')
        for row in rows:
            require(set(row['views'])==set(VIEWS),'Selective-belief eight-view grid differs')
            if panel['family_stratified']:
                require(row['core']['history']=='shared_initial','Unexecuted belief history in development evidence')
            for view,cells in row['views'].items():
                require(set(cells)=={'clean','source','belief_counterfactual','full_event'},'Selective-belief four-control grid differs')
                for method,cell in cells.items():
                    core=row['core'];loc=core['base'] if method=='clean' else core['source']
                    target=core['base'] if view=='world' and method=='belief_counterfactual' else clean_answer(core,view,loc)
                    require(cell['target']==target,'Selective-belief symbolic target differs')
        families=sorted({r['core']['history'] for r in rows}) if panel['family_stratified'] else ['all']
        cells,joints={},{}
        for family in families:
            selected=[r for r in rows if family=='all' or r['core']['history']==family]
            methods=tuple(selected[0]['views'][panel['views'][0]])
            for method in methods:
                for view in panel['views']:
                    cells[f'{family}/{method}/{view}']=cell_result([r['views'][view][method]['score'] for r in selected],[r['views'][view][method]['target'] for r in selected])
                h=sum(all(r['views'][v][method]['score'][1]==r['views'][v][method]['target'] for v in panel['views']) for r in selected)
                joints[f'{family}/{method}']={'correct':h,'n':len(selected),'historical_joint_pass':h/len(selected)>=.8}
        out['belief'][name]={'cells':cells,'joint':joints,'historical_natural_gate_pass':all(x['historical_cell_pass'] for x in cells.values()) and all(x['historical_joint_pass'] for x in joints.values()),'learned_stage_run':False}
    for name,panel in data['route'].items():
        cores={r['id']:r for r in panel['data']['stories']};rows=panel['rows'];require(len(rows)==(1080 if name=='pilot' else 252),'Route grid count differs');require(len({(r['story_id'],r['role'],r['view'],r.get('variant')) for r in rows})==len(rows),'Duplicate route row')
        grouped={}
        for row in rows:
            c=cores[row['story_id']];v=row['view'];loc=c[row['role']]
            if v.startswith('route_'):
                target=LOCATIONS[(LOCATIONS.index(loc)+(1 if v=='route_forward' else -1))%6]
            elif v=='other_agent':target=c['initial']
            elif v=='observed_overwrite':target=c['update']
            elif v=='irrelevant_object':target=c['distractor_location']
            else:target=loc
            key='/'.join([row.get('variant') or 'direct_control',row['role'],v]) if name=='development' else '/'.join([row['role'],v])
            grouped.setdefault(key,[]).append((row['score'],target))
        cells={k:cell_result([s for s,t in x],[t for s,t in x]) for k,x in grouped.items()}
        if name=='pilot':
            required={k:v for k,v in cells.items() if '/route_' in k};passed=all(v['global_correct']>=54 for v in required.values());ranking=[]
        else:
            ranking=[]
            for order,variant in enumerate(panel['data']['variants']):
                rc=[v for k,v in cells.items() if k.startswith(variant+'/') and '/route_' in k]
                require(len(rc)==6 and all(v['n']==12 for v in rc),'Incomplete development route cells')
                ranking.append({'variant':variant,'order':order,'minimum_correct':min(c['global_correct'] for c in rc),'total_correct':sum(c['global_correct'] for c in rc),'eligible':all(c['global_correct']>=11 for c in rc)})
            ranking.sort(key=lambda x:(-x['minimum_correct'],-x['total_correct'],x['order']));passed=any(x['eligible'] for x in ranking)
        out['route'][name]={'rows':len(rows),'cells':cells,'historical_natural_gate_pass':passed,'ranking':ranking,'learned_stage_run':False}
    return out


def mediation_results(data):
    rows=data['rows'];require(len(rows)==9600,'Complete 9600-row mediation screen required')
    require(len({(r['component'],r['story_key'],r['view']) for r in rows})==9600,'Duplicate mediation row')
    by=collections.defaultdict(list)
    for row in rows:
        require(row['seed']==101 and set(row['conditions'])=={'clean','clean_disrupted','clean_identity','patched','patched_identity','restored','transplanted'},'Mediation seed/condition grid differs')
        require(len(row['controls'])==3,'All three matched controls required')
        for score in [*row['conditions'].values(),*[s for c in row['controls'] for s in c['conditions'].values()]]:
            require(len(score)==3 and all(math.isfinite(x) for x in score[:2]),'Nonfinite saved log probability')
        by[row['component']].append(row)
    require(len(by)==20 and all(len(v)==480 for v in by.values()),'All twenty candidates required')
    result={}
    mean=lambda x:sum(x)/len(x)
    margin=lambda x:x[0]-x[1]
    for component,rs in sorted(by.items()):
        aff=[r for r in rs if r['clean']!=r['target']];unaff=[r for r in rs if r['clean']==r['target']]
        require(len(aff)==300 and len(unaff)==180,'Fixed mediation witness support differs')
        t=mean([margin(r['conditions']['patched'])-margin(r['conditions']['clean']) for r in aff]);rest=mean([margin(r['conditions']['patched'])-margin(r['conditions']['restored']) for r in aff]);trans=mean([margin(r['conditions']['transplanted'])-margin(r['conditions']['clean']) for r in aff])
        cr=[mean([margin(r['conditions']['patched'])-margin(r['controls'][k]['conditions']['restored']) for r in aff]) for k in range(3)]
        ct=[mean([margin(r['controls'][k]['conditions']['transplanted'])-margin(r['conditions']['clean']) for r in aff]) for k in range(3)]
        acc=lambda subset,condition:mean([r['conditions'][condition][2]==r['clean'] for r in subset])
        preservation=max(acc(unaff,'patched')-acc(unaff,'restored'),acc(unaff,'clean')-acc(unaff,'transplanted'))
        margins=[rest-.5*t,trans-.5*t,rest-max(cr),trans-max(ct)]
        flags={'total_effect':t>.1,'restoration_half':margins[0]>0,'transplant_half':margins[1]>0,'restoration_beats_controls':margins[2]>0,'transplant_beats_controls':margins[3]>0,'preservation':preservation<=.05}
        result[component]={'affected_rows':len(aff),'unaffected_rows':len(unaff),'total_effect':t,'restoration_effect':rest,'transplant_effect':trans,'restoration_fraction':rest/t if t>.1 else None,'transplant_fraction':trans/t if t>.1 else None,'control_restoration_effects':cr,'control_transplant_effects':ct,'restoration_half_margin':margins[0],'transplant_half_margin':margins[1],'restoration_control_margin':margins[2],'transplant_control_margin':margins[3],'unaffected_accuracy_loss':preservation,'clean_disruption_accuracy_loss':acc(rs,'clean')-acc(rs,'clean_disrupted'),'criterion_flags':flags,'historical_practical_pass':all(flags.values())}
    return {'rows':9600,'candidates':result,'passed_candidates':sum(x['historical_practical_pass'] for x in result.values()),'checking_run':False,'scope':'Saved final-answer-site scores; no claim about other sites or native tensor integrity.'}


def analyze(evidence):
    validate_main(evidence['main'])
    return {'schema_version':1,'scope':'CPU results from saved case-level predictions; no model execution or native/tensor audit.','bootstrap':{'draws':2000,'seed':2026091301,'unit':'pair within each of five equally weighted families, or whole consequence core; three fixed fits stay paired','pointwise_not_multiplicity_adjusted':True},'primary':primary_results(evidence['main']),'clean_opportunity':opportunity_results(evidence['main']),'pca_identity':{k:v['pca_identity'] for k,v in evidence['main']['models'].items()},'consequences':consequence_results(evidence['main']),'supporting':support_results(evidence['supporting']),'final_answer_mediation':mediation_results(evidence['mediation'])}


def compare(actual, expected, path=''):
    require(type(actual) is type(expected), 'Reference type differs at '+path)
    if isinstance(actual,dict):
        require(actual.keys()==expected.keys(),'Reference fields differ at '+path)
        for key in actual:compare(actual[key],expected[key],path+'/'+key)
    elif isinstance(actual,list):
        require(len(actual)==len(expected),'Reference length differs at '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i))
    elif isinstance(actual,float):
        require(math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12),'Reference numeric result differs at '+path)
        if '/ci95/' in path:
            require((actual>0)-(actual<0)==(expected>0)-(expected<0),'Reference interval sign differs at '+path)
    else:require(actual==expected,'Reference result differs at '+path)


def markdown(report):
    lines=['# Recomputed behavioral results','','Candidate ranking and global-next-token correctness are distinct. All intervals below are pointwise paired intervals conditional on the three fixed fits.','','| Model | Interface | Intended C / G (%) | Alternative C / G (%) | Global gap [95% CI], pp |','|---|---|---:|---:|---:|']
    for model,formats in report['primary'].items():
        for fmt,v in formats.items():
            a,b=(v['methods'][o+'/learned'] for o in ('f_star','m3'));d=v['intended_minus_alternative_global'];lo,hi=d['ci95']
            lines.append(f"| {model} | {fmt} | {100*a['candidate_mean']:.2f} / {100*a['global_mean']:.2f} | {100*b['candidate_mean']:.2f} / {100*b['global_mean']:.2f} | {100*d['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['','| Model | Population | Objective | PCA (%) | Learned (%) | Learned − PCA [95% CI], pp |','|---|---|---|---:|---:|---:|']
    for model in ('qwen','mistral'):
        for objective in ('f_star','m3'):
            for population in ('five shifts','eight checking views'):
                v=report['primary'][model]['answer_prefill']['learned_minus_pca'][objective] if population=='five shifts' else report['consequences'][model]['checking'][objective]['learned_minus_comparators_all_views']['pca']
                lo,hi=v['ci95']
                lines.append(f"| {model} | {population} | {objective} | {100*v['right_mean']:.2f} | {100*v['left_mean']:.2f} | {100*v['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['','| Model | Objective | Seed | All eight answers / 960 | Joint / 120 | Report / 120 | Other-view consistency / 120 |','|---|---|---:|---:|---:|---:|---:|']
    for model,panels in report['consequences'].items():
        for objective in ('f_star','m3'):
            v=panels['checking'][objective]
            learned=v['methods']['learned']
            for fit,cons in zip(learned['per_fit'],learned['consistency']):
                lines.append(f"| {model} | {objective} | {fit['seed']} | {fit['correct_answers']} | {fit['joint_correct']} | {fit['view_correct'][-1]} | {cons['all_other_consistent']} |")
    lines+=['','The JSON retains both panels, PCA/output-recoding baselines, every account contrast and opportunity denominator, and every supporting screen cell.','',f"Final-answer mediation: {report['final_answer_mediation']['passed_candidates']}/20 candidates passed the original practical nomination rule; checking was not run."]
    return '\n'.join(lines)+'\n'


def run(output, data_dir=None, expected=None):
    data_dir=Path(data_dir) if data_dir is not None else ROOT/'data/behavior'
    output=Path(output)
    require(not output.exists(),'Output must be a new directory')
    evidence,manifest=load_evidence(data_dir)
    result=analyze(evidence)
    reference=Path(expected) if expected is not None else ROOT/'expected/behavior.json' if data_dir.resolve()==(ROOT/'data/behavior').resolve() else None
    if reference is not None:compare(result,json.loads(reference.read_text()))
    output.mkdir(parents=True)
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    (output/'behavior.json').write_text(payload)
    (output/'behavior_tables.md').write_text(markdown(result))
    receipt={'status':'PASS','new_model_calls':0,'data_manifest_sha256':digest((data_dir/'SOURCE_PROVENANCE.json').read_bytes()),'input_files':manifest['files'],'source_sha256':digest(Path(__file__).read_bytes()),'expected_reference_compared':reference is not None,'expected_reference_sha256':digest(reference.read_bytes()) if reference else None,'reference_float_absolute_tolerance':1e-12,'ci_signs_compared_without_tolerance':True,'outputs':{n:digest((output/n).read_bytes()) for n in ('behavior.json','behavior_tables.md')}}
    (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description='Recompute matched behavioral results and retained negative screens from compact case-level evidence.')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--expected',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.data_dir,args.expected),sort_keys=True))


if __name__=='__main__':
    main()
