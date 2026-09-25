







from __future__ import annotations
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys

from .qwen_geometry import (WIDTH, PADDING, N_LAYERS, FIRST_LAYER, Q_HEADS, KV_HEADS,
                      HEAD_DIM, MODEL, REVISION, NATIVE_SOURCE_HASHES)
SOURCE_HASHES = dict(NATIVE_SOURCE_HASHES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime(engine):
    pass                                                                           
    require(engine.environment["model"] == MODEL and engine.environment["revision"] == REVISION,
            "Unexpected model/revision")
    require(len(engine.layers) == N_LAYERS and engine.model.config._attn_implementation == "sdpa" and
            engine.model.config.use_cache is False, "Unexpected native architecture/cache/backend")
    modeling = inspect.getsourcefile(type(engine.layers[0]))
    sdpa_module = sys.modules.get("transformers.integrations.sdpa_attention")
    require(sdpa_module is not None, "Native SDPA module is not loaded")
    paths = {"modeling_qwen2.py": modeling, "sdpa_attention.py": inspect.getsourcefile(sdpa_module)}
    for name, path in paths.items():
        require(path is not None and file_sha(path) == SOURCE_HASHES[name], "Pinned native source differs: " + name)
    for index, block in enumerate(engine.layers):
        attention = block.self_attn
        require(not block.training and not attention.training and block.hidden_size == WIDTH and
                attention.layer_idx == index and attention.head_dim == HEAD_DIM and
                attention.num_key_value_groups == Q_HEADS // KV_HEADS and attention.is_causal and
                attention.attention_dropout == 0 and getattr(attention, "sliding_window", None) is None,
                "Native attention contract differs")
        for name, shape in (("q_proj", (Q_HEADS * HEAD_DIM, WIDTH)),
                            ("k_proj", (KV_HEADS * HEAD_DIM, WIDTH)),
                            ("v_proj", (KV_HEADS * HEAD_DIM, WIDTH)),
                            ("o_proj", (WIDTH, Q_HEADS * HEAD_DIM))):
            module = getattr(attention, name)
            require(tuple(module.weight.shape) == shape and
                    module.weight.dtype == engine.torch.bfloat16 and module.weight.device == engine.device,
                    "Native projection contract differs")
            if name == "o_proj":
                require(module.bias is None, "Native output projection must not have bias")
            else:
                require(module.bias is not None and tuple(module.bias.shape) == (shape[0],) and
                        module.bias.dtype == engine.torch.bfloat16 and module.bias.device == engine.device,
                        "Native Qwen Q/K/V projection bias differs")
    return dict(SOURCE_HASHES)


PARENT_ADAPTER_SHA256 = "0b34c34760dc4888c8ca1822d44f4a2536439bc034bad35d2660a73c0c33da4e"

@dataclass(frozen=True)
class RelayReference:
    metadata: dict
    keys: dict
    values: dict
    output_prefixes: dict
    hashes: dict


class RelayEdgeEngine:
    pass                                                                     

                                                                               
                                                                                  
                                                                                     
                                                                              
                                                                              
       
    def __init__(self, engine):
        self.engine,self.torch=engine,engine.torch
        self.source_identity=validate_runtime(engine)
        self.adapter_provenance={'parent_adapter_sha256':PARENT_ADAPTER_SHA256,'source_sha256':file_sha(__file__),
                                 'key_site':'biased k_proj output before native reshape/RoPE','value_site':'biased v_proj output',
                                 'direct_query_intervention':False}
        self._active=False;self.last_receipt=None;self.modules={}
        for layer in range(FIRST_LAYER,N_LAYERS+1):
            block=engine.layers[layer-1];self.modules[layer,'residual']=block
            for name in ('q','k','v','o'):self.modules[layer,name]=getattr(block.self_attn,name+'_proj')
        require(len({id(m) for m in self.modules.values()})==len(self.modules),'Shared native modules unsupported')
        self._dispatch={key:self._function(module.forward) for key,module in self.modules.items()}

    @staticmethod
    def _function(function):return getattr(function,'__func__',function)

    def _native_guard(self):
        require(file_sha(__file__)==self.adapter_provenance['source_sha256'],'Adapter source changed')
        for key,module in self.modules.items():
            require(self._function(module.forward) is self._dispatch[key] and not module.training,'Native dispatch/training changed')
        return tuple((id(p),int(p._version)) for p in self.engine.model.parameters())

    def _hash(self,tensor):return self.engine.tensor_hash(tensor.detach().contiguous())
    def _clone_cpu(self,tensor):return tensor.detach().cpu().clone()
    def _equal_bytes(self,left,right):
        return self.torch.equal(left.contiguous().view(self.torch.uint16),right.contiguous().view(self.torch.uint16))
    def _tensor(self,tensor,shape):
        require(self.torch.is_tensor(tensor) and tuple(tensor.shape)==tuple(shape) and tensor.dtype==self.torch.bfloat16 and
                not tensor.requires_grad,'Unexpected BF16 native tensor shape/dtype')

    def _identity(self,encoded,layer4_patch):
        ids,(a,z),p=encoded['input_ids'],encoded['span'],encoded['event_value_position']
        require(type(p) is int and 0<=a<=p<z<len(ids)<PADDING,'Invalid event/value coordinates')
        require(digest(ids)==encoded['input_ids_sha256'] and digest(ids[:z])==encoded['prefix_sha256'],'Encoded input hash differs')
        return {'schema_version':1,'prefix_ids':list(ids[:z]),'prefix_sha256':encoded['prefix_sha256'],
                'span':[a,z],'value_position':p,'restore_stops':{'critical_token':p+1,'event_end':z},
                'layer4_patch_sha256':None if layer4_patch is None else self._hash(layer4_patch),
                'source_identity':self.source_identity,'adapter_provenance':self.adapter_provenance}

    def capture(self,encoded,*,layer4_patch=None,expected_layer4=None):
        return self._execute(encoded,layer4_patch=layer4_patch,expected_layer4=expected_layer4,
                             recipient=None,donor=None,groups=None,mode=None,restore_cutoff=None,clone=False)

    def run(self,encoded,*,recipient,donor,groups,mode,restore_cutoff,layer4_patch=None,expected_layer4=None,clone=False):
        return self._execute(encoded,layer4_patch=layer4_patch,expected_layer4=expected_layer4,
                             recipient=recipient,donor=donor,groups=groups,mode=mode,restore_cutoff=restore_cutoff,clone=clone)

    def _execute(self,encoded,*,layer4_patch,expected_layer4,recipient,donor,groups,mode,restore_cutoff,clone):
        require(not self._active,'Overlapping relay execution')
        identity=self._identity(encoded,layer4_patch);p=identity['value_position'];z=identity['span'][1];n=len(encoded['input_ids'])
        capture=recipient is None
        if capture:
            require(donor is None and groups is None and mode is None and restore_cutoff is None,'Capture contains an intervention')
            selected={l:tuple(range(KV_HEADS)) for l in range(FIRST_LAYER,N_LAYERS+1)}
            stop=z;fields=()
        else:
            require(mode in ('k','v','kv') and restore_cutoff in ('critical_token','event_end'),'Explicit K/V operation and restoration cutoff required')
            fields=tuple(mode);stop=identity['restore_stops'][restore_cutoff]
            require(isinstance(recipient,RelayReference) and isinstance(donor,RelayReference),'Frozen relay references required')
            require(recipient.metadata==identity,'Recipient prefix/patch/source binding differs')
            dm=donor.metadata
            require(dm['span']==identity['span'] and dm['value_position']==p and dm['source_identity']==self.source_identity and
                    dm['adapter_provenance']==self.adapter_provenance,'Donor coordinates/native source differ')
            require(len(dm['prefix_ids'])==z and all(a==b for i,(a,b) in enumerate(zip(dm['prefix_ids'],identity['prefix_ids'])) if i!=p),
                    'Donor changes matched prefix outside value token')
            require(isinstance(groups,dict) and groups,'Explicit nonempty KV-group selection required')
            selected={}
            for layer,values in groups.items():
                values=tuple(values)
                require(type(layer) is int and FIRST_LAYER<=layer<=N_LAYERS and values and
                        all(type(g) is int and 0<=g<KV_HEADS for g in values) and len(set(values))==len(values),'Invalid KV-group selection')
                selected[layer]=tuple(sorted(values))
            for layer in selected:
                self._tensor(recipient.output_prefixes[layer],(z,WIDTH))
                require(self._hash(recipient.output_prefixes[layer])==recipient.hashes[layer,'o_prefix_event_end'],'Frozen recipient prefix mutated')
                for field in fields:
                    payload=donor.keys[layer] if field=='k' else donor.values[layer]
                    self._tensor(payload,(KV_HEADS,HEAD_DIM))
                    require(self._hash(payload)==donor.hashes[layer,field+'_payload'],'Frozen donor payload mutated')
        before_parameters=self._native_guard()
        before_hooks={key:tuple(module._forward_hooks) for key,module in self.modules.items()}
        require(not any(before_hooks.values()) and all(not getattr(m,'_forward_pre_hooks',{}) for m in self.modules.values()),'Preexisting hooks conflict with relay ownership')
        self._active=True
        handles,observed,keys,values,outputs,hashes,changes=[],[],{},{},{},{},{}
        expected=[(l,k) for l in range(FIRST_LAYER,N_LAYERS+1) for k in (('q','k','v','o','residual') if l in selected else ('residual',))]
        receipt={'schema_version':1,'status':'STARTED','mode':'capture' if capture else 'outgoing_relay',
                 'operation_mode':mode,'restore_cutoff':restore_cutoff,'restore_stop':None if capture else stop,
                 'identity':identity,'adapter_provenance':self.adapter_provenance,
                 'groups':{str(k):list(v) for k,v in sorted(selected.items())},'expected_native_call_id':self.engine.call_count+1,
                 'q_directly_modified':False,'k_site':'pre_RoPE_k_proj','position_cache_backend_unchanged':True,
                 'allowed_later_prefix_span':[stop,z] if not capture else None,'valid_suffix_span':[z,n]}
        self.last_receipt=receipt

        def capture_prefixes(layer,kind,tensor):
            for name,end in identity['restore_stops'].items():hashes[layer,kind+'_prefix_'+name]=self._hash(tensor[0,:end])

        def hook(layer,kind):
            def apply(_module,_inputs,output):
                self.engine.check();observed.append((layer,kind))
                require(observed==expected[:len(observed)],'Native field order/repetition differs')
                tensor=output[0] if isinstance(output,tuple) else output
                width=WIDTH if kind in ('o','residual') else (Q_HEADS if kind=='q' else KV_HEADS)*HEAD_DIM
                self._tensor(tensor,(1,PADDING,width))
                if capture:
                    capture_prefixes(layer,kind,tensor)
                else:
                    current=self._hash(tensor[0,:stop])
                    if kind!='o':
                        require(current==recipient.hashes[layer,kind+'_prefix_'+restore_cutoff],
                                'Relay changed preserved '+kind+' prefix before declared insertion/restoration')
                        hashes[layer,kind+'_prefix_'+restore_cutoff]=current
                if kind=='residual':
                    hashes[layer,'residual_later_prefix']=self._hash(tensor[0,(p+1 if capture else stop):z])
                    hashes[layer,'residual_valid_suffix']=self._hash(tensor[0,z:n])
                    return None
                if kind=='q':return None
                if kind in ('k','v'):
                    if capture:
                        payload=self._clone_cpu(tensor[0,p].reshape(KV_HEADS,HEAD_DIM))
                        (keys if kind=='k' else values)[layer]=payload
                        hashes[layer,kind+'_payload']=self._hash(payload)
                        return None
                    if kind not in fields:
                        changes.setdefault(layer,{})[kind]={'directly_modified':False,'preserved_prefix_sha256':self._hash(tensor[0,:stop])}
                        return None
                    original=tensor.clone();mixed=tensor.clone()
                    payload=donor.keys[layer] if kind=='k' else donor.values[layer]
                    for group in selected[layer]:
                        a,b=group*HEAD_DIM,(group+1)*HEAD_DIM
                        mixed[0,p,a:b]=payload[group].to(self.engine.device)
                    require(self._equal_bytes(mixed[:,:p],original[:,:p]) and self._equal_bytes(mixed[:,p+1:],original[:,p+1:]),'A noncritical '+kind+' row changed')
                    for group in range(KV_HEADS):
                        a,b=group*HEAD_DIM,(group+1)*HEAD_DIM
                        wanted=payload[group].to(self.engine.device) if group in selected[layer] else original[0,p,a:b]
                        require(self._equal_bytes(mixed[0,p,a:b],wanted),'Selected/unselected '+kind+' group differs')
                    require(self._equal_bytes(tensor,original),'Native '+kind+' storage mutated')
                    changes.setdefault(layer,{})[kind]={'directly_modified':True,'site':'pre_RoPE_k_proj' if kind=='k' else 'v_proj',
                        'donor_payload_sha256':donor.hashes[layer,kind+'_payload'],'applied_row_sha256':self._hash(mixed[0,p]),
                        'other_rows_exact':True,'unselected_groups_exact':True,'native_storage_unchanged':True}
                    return mixed
                if capture:
                    outputs[layer]=self._clone_cpu(tensor[0,:z])
                    return None
                original=tensor.clone();restored=tensor.clone()
                later_hash=self._hash(original[0,stop:z]);suffix_hash=self._hash(original[0,z:n])
                restored[0,:stop]=recipient.output_prefixes[layer][:stop].to(self.engine.device)
                require(self._hash(restored[0,:stop])==recipient.hashes[layer,'o_prefix_'+restore_cutoff],'Recipient o prefix restoration differs')
                require(self._equal_bytes(restored[:,stop:],original[:,stop:]),'Retained later-prefix or padded suffix changed during restoration')
                require(self._equal_bytes(tensor,original),'Native o storage mutated')
                changes[layer]['o']={'restored_prefix_sha256':recipient.hashes[layer,'o_prefix_'+restore_cutoff],
                    'mixed_later_prefix_sha256':later_hash,'mixed_valid_suffix_sha256':suffix_hash,
                    'later_prefix_span':[stop,z],'valid_suffix_span':[z,n],'all_retained_rows_exact':True,
                    'native_storage_unchanged':True}
                return restored
            return apply
        try:
            for layer in range(FIRST_LAYER,N_LAYERS+1):
                for kind in (('q','k','v','o','residual') if layer in selected else ('residual',)):
                    handles.append(self.modules[layer,kind].register_forward_hook(hook(layer,kind)))
            result,engine_captures=self.engine.run(encoded,layer4_patch=layer4_patch,capture_layers=(4,),
                expected_prefix=identity['prefix_ids'],expected_layer4=expected_layer4,clone=clone)
            require(observed==expected,'Native hook sequence incomplete')
            require(result['audit']['call_id']==receipt['expected_native_call_id'] and result['audit']['prefix_sha256']==identity['prefix_sha256'] and
                    result['audit']['layer4_patch_sha256']==identity['layer4_patch_sha256'] and result['audit']['replacement_sha256']=={},'Engine call/intervention binding differs')
            require(self._native_guard()==before_parameters,'Model parameters changed')
            if not capture:
                for layer in selected:
                    require(self._hash(recipient.output_prefixes[layer])==recipient.hashes[layer,'o_prefix_event_end'],'Frozen recipient prefix changed')
                    for field in fields:
                        payload=donor.keys[layer] if field=='k' else donor.values[layer]
                        require(self._hash(payload)==donor.hashes[layer,field+'_payload'],'Frozen donor payload changed')
            receipt.update({'status':'COMPLETE','native_call_id':result['audit']['call_id'],'guarded_layers':list(range(FIRST_LAYER,N_LAYERS+1)),
                'all_preserved_residual_prefixes_exact':not capture,'changes':{str(k):v for k,v in changes.items()},
                'residual_preserved_prefix_sha256':{str(l):hashes[l,'residual_prefix_'+('event_end' if capture else restore_cutoff)] for l in range(FIRST_LAYER,N_LAYERS+1)},
                'residual_later_prefix_sha256':{str(l):hashes[l,'residual_later_prefix'] for l in range(FIRST_LAYER,N_LAYERS+1)},
                'residual_valid_suffix_sha256':{str(l):hashes[l,'residual_valid_suffix'] for l in range(FIRST_LAYER,N_LAYERS+1)},'hook_calls':len(observed)})
            if capture:return result,engine_captures,RelayReference(identity,keys,values,outputs,hashes)
            return result,engine_captures,receipt
        except BaseException as error:
            receipt.update({'status':'FAILED','error_type':type(error).__name__,'error':str(error),'native_call_frontier':self.engine.call_count,'observed_hook_calls':len(observed)})
            raise
        finally:
            for handle in reversed(handles):handle.remove()
            self._active=False
            require(all(tuple(module._forward_hooks)==before_hooks[key] for key,module in self.modules.items()),'Relay hook cleanup failed')
