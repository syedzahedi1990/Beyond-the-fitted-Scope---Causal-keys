






from __future__ import annotations
import copy
import hashlib
import json



COORDINATE_FIELDS=('schema_version','span','value_position','prefix_ids','source_identity','adapter_provenance')

def require(condition,message):
    if not condition:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _snapshot(engine,reference):
    return {'metadata':copy.deepcopy(reference.metadata),'hashes':copy.deepcopy(reference.hashes),
        **{field:{int(l):engine.tensor_hash(t) for l,t in getattr(reference,field).items()} for field in ('keys','values','output_prefixes')}}

def _coordinates(recipient,source):
    r=recipient.metadata;m=source.metadata;a,z=r['span'];p=r['value_position']
    require(type(p)is int and 0<=a<=p<z and len(r['prefix_ids'])==z,'Invalid recipient event coordinates')
    require(m['schema_version']==r['schema_version']==1 and m['span']==r['span'] and m['value_position']==p and m['source_identity']==r['source_identity'] and m['adapter_provenance']==r['adapter_provenance'],'Component source coordinates/native implementation differ')
    require(len(m['prefix_ids'])==z and all(x==y for i,(x,y) in enumerate(zip(r['prefix_ids'],m['prefix_ids'])) if i!=p),'Component source changes prefix outside critical token')

def _origin(reference,source_id,origin,component,recipient_id):
    require(isinstance(origin,dict) and origin.get('reference_id')==source_id,'Origin must identify exact source reference')
    digest(origin)
    synthetic=reference.metadata.get('synthetic_provenance')
    if synthetic is None:
        require(origin.get('kind')=='captured' and not source_id.startswith('synthetic/'),'Captured origin/source kind differs')
    else:
        require(origin.get('kind')=='signed_component' and origin.get('component')==component and origin.get('recipient_reference_id')==recipient_id and source_id.startswith('synthetic/'),'Signed component provenance or recipient differs')
        require(synthetic.get('kind')=='signed_coordinate_permutation' and synthetic.get('natural_capture') is False and synthetic.get('seed')==2026091610 and synthetic.get('recipient_reference_id')==recipient_id and synthetic.get('coordinate_compatibility_reference_id')==recipient_id,'Existing signed null belongs to another recipient or realization')
                                                                                      
        require(reference.output_prefixes=={} and all(kind in ('k_payload','v_payload') for _,kind in reference.hashes),'Signed reference fabricates natural captures')

def _clone_payload(engine,reference,field,layer):
    name='keys' if field=='k' else 'values';t=getattr(reference,name)[layer]
    require(tuple(t.shape)==(8,128) and t.dtype==engine.torch.bfloat16 and not t.requires_grad,'Expected native BF16 payload shape/dtype without gradients')
    before=engine.tensor_hash(t);require(before==reference.hashes[layer,field+'_payload'],'Frozen component payload hash differs')
    clone=t.detach().cpu().clone()
    require(tuple(clone.shape)==(8,128) and clone.dtype==engine.torch.bfloat16 and not clone.requires_grad and engine.tensor_hash(clone)==before,'Exact native BF16 clone differs')
    require(clone is not t,'Hybrid component must own an independent clone')
    return clone,before

def assemble_hybrid(engine,recipient,key_reference,value_reference,*,recipient_id,key_source_id,value_source_id,key_origin,value_origin):
    require(all(isinstance(x,type(recipient)) for x in (recipient,key_reference,value_reference)),'Native RelayReference inputs required')
    require(all(isinstance(x,str) and x for x in (recipient_id,key_source_id,value_source_id)),'Explicit reference identities required')
    require(set(key_reference.keys)==set(value_reference.values)==set(tuple(range(5,len(engine.layers)+1))),'All native payload layers required')
    for source in (key_reference,value_reference):_coordinates(recipient,source)
    _origin(key_reference,key_source_id,key_origin,'k',recipient_id)
    _origin(value_reference,value_source_id,value_origin,'v',recipient_id)
    originals=(recipient,key_reference,value_reference);before=[_snapshot(engine,r) for r in originals]
    fields={'keys':{},'values':{}};hashes={};components={}
    for f,name,reference,source_id,origin in (('k','keys',key_reference,key_source_id,key_origin),('v','values',value_reference,value_source_id,value_origin)):
        per_layer={}
        for layer in tuple(range(5,len(engine.layers)+1)):
            tensor,h=_clone_payload(engine,reference,f,layer);fields[name][layer]=tensor;hashes[layer,f+'_payload']=h;per_layer[str(layer)]=h
        components[f]={'source_reference_id':source_id,'origin':copy.deepcopy(origin),
            'source_metadata_sha256':digest(reference.metadata),'payload_sha256':per_layer}
    provenance={'kind':'exact_key_value_source_hybrid','natural_capture':False,'recipient_reference_id':recipient_id,
        'components':copy.deepcopy(components),'operation':'independent native BF16 clones; no arithmetic or fitting'}
    metadata={key:copy.deepcopy(recipient.metadata[key]) for key in COORDINATE_FIELDS}
    metadata['synthetic_provenance']=provenance
    hybrid=type(recipient)(metadata=metadata,keys=fields['keys'],values=fields['values'],output_prefixes={},hashes=hashes)
    require(all(_snapshot(engine,r)==snap for r,snap in zip(originals,before)),'Hybrid assembly mutated an input reference')
    require(set(hybrid.metadata)==set(COORDINATE_FIELDS)|{'synthetic_provenance'} and len(hybrid.hashes)==2*(len(engine.layers)-4) and hybrid.output_prefixes=={},'Synthetic donor metadata/capture scope differs')
    notes={'schema_version':1,'kind':'exact_key_value_source_hybrid','recipient_reference_id':recipient_id,
        'components':components,'available_payload_layers':list(tuple(range(5,len(engine.layers)+1))),'payload_shape':[8,128],
        'component_arithmetic':'none','independent_clones':True,'input_references_unchanged':True,
        'only_payload_hashes_supplied':True,'output_prefixes_supplied':False,
        'synthetic_metadata_sha256':digest(metadata)}
    return hybrid,notes
