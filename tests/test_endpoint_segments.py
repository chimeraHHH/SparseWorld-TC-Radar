import hashlib
import importlib.util
import json
import sys
import types
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

# Geometry tests do not mutate the process-wide MMDetection registry.
_registry=types.ModuleType('mmdet.datasets.builder')
_registry.PIPELINES=types.SimpleNamespace(register_module=lambda:lambda cls:cls)
with patch.dict(sys.modules, {'mmdet.datasets.builder':_registry}):
    endpoint=load_file('endpoint_segments',ROOT/'loaders/pipelines/endpoint_segments.py')
builder=load_file('build_endpoint_cache',ROOT/'tools/build_endpoint_cache.py')


def frame(tokens=('instance-a',),labels=(4,),centers=None):
    n=len(tokens)
    return dict(centers=np.array(centers if centers is not None else [[.5,.5,.5]]*n,dtype=np.float32),
                rotations=np.repeat(np.eye(3)[None],n,axis=0).astype(np.float32),
                sizes=np.ones((n,3),dtype=np.float32)*1.1,labels=np.array(labels,dtype=np.int64),
                instance_tokens=np.array(tokens),timestamp_us=np.int64(0),
                ego2global_rotation=np.eye(3),ego2global_translation=np.zeros(3))


def fixture_cache(tmp_path,frames):
    directory=tmp_path/'train'/'frames';directory.mkdir(parents=True)
    tokens=['sample-'+str(i) for i in range(len(frames))]
    for t,f in zip(tokens,frames):np.savez(directory/(t+'.npz'),**f)
    anchors={'sample-0':tokens};path=directory.parent/'anchors.json';path.write_text(json.dumps(anchors))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema_version=endpoint.SCHEMA_VERSION,
        occupancy_labels_cached=False,horizons=[0,2,4,6],splits={'train':dict(anchor_tokens=['sample-0'],
        anchors_sha256=hashlib.sha256(path.read_bytes()).hexdigest())})))
    return endpoint.LoadEndpointSegments(tmp_path,pc_range=(0,0,0,4,4,2),voxel_size=(1,1,1))


def results_for(frames):
    sem=[];masks=[];poses=[]
    R0=frames[0]['ego2global_rotation'];t0=frames[0]['ego2global_translation']
    for f in frames:
        x=np.full((4,4,2),17,dtype=np.uint8);x[0,0,0]=4;sem.append(x)
        masks.append(np.ones_like(x,dtype=bool));T=np.eye(4);T[:3,:3]=R0.T@f['ego2global_rotation'];T[:3,3]=R0.T@(f['ego2global_translation']-t0);poses.append(T)
    return dict(sample_idx='sample-0',fut_list=[0,2,4,6],voxel_semantics=sem,mask_camera=masks,fut2cur=poses)


def test_unknown_and_semantic_mismatch_do_not_become_points():
    f=frame();sem=np.full((2,2,1),17,dtype=np.uint8);sem[0,0,0]=4
    mask=np.zeros_like(sem,dtype=bool)
    assert len(endpoint.observed_box_points(sem,mask,f,(0,0,0,2,2,1),(1,1,1))[0])==0
    mask[:]=True;sem[0,0,0]=7
    assert len(endpoint.observed_box_points(sem,mask,f,(0,0,0,2,2,1),(1,1,1))[0])==0


def test_overlapping_compatible_boxes_are_excluded_including_third_owner():
    sem=np.full((2,2,1),17,dtype=np.uint8);sem[0,0,0]=4;mask=np.ones_like(sem,dtype=bool)
    f=frame(('a','b','c'),(4,4,4))
    assert all(len(x)==0 for x in endpoint.observed_box_points(sem,mask,f,(0,0,0,2,2,1),(1,1,1)))
    f=frame(('a','b'),(4,7))
    points=endpoint.observed_box_points(sem,mask,f,(0,0,0,2,2,1),(1,1,1))
    assert len(points[0])==1 and len(points[1])==0


def test_oriented_box_test_rejects_aabb_only_voxels():
    f=frame(centers=[[1,1,.5]])
    theta=np.pi/4;f['rotations'][0]=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    f['sizes'][0]=[2,.2,1]
    sem=np.full((2,2,1),4,dtype=np.uint8)
    points=endpoint.observed_box_points(sem,np.ones_like(sem),(f),(0,0,0,2,2,1),(1,1,1))[0]
    np.testing.assert_allclose(points,[[.5,.5,.5],[1.5,1.5,.5]])


def test_true_instance_endpoints_do_not_fill_hidden_frame_and_use_physical_times(tmp_path):
    frames=[frame() for _ in range(4)]
    for i,f in enumerate(frames):f['timestamp_us']=np.int64([0,990000,2020000,3100000][i]);f['ego2global_translation']=np.array([i*10.,0,0])
    loader=fixture_cache(tmp_path,frames);results=results_for(frames);results['mask_camera'][1][:]=False
    target=loader(results)['endpoint_segments']
    assert target['instance_tokens']==['instance-a']
    np.testing.assert_array_equal(target['observed'],[[True,False,True,True]])
    assert not target['point_valid'][0,1].any() and not target['points'][0,1].any()
    np.testing.assert_allclose(target['times'],[0,.99,2.02,3.1])
    np.testing.assert_allclose(target['points'][0,2,0],[20.5,.5,.5])
    again=loader(results)['endpoint_segments'];np.testing.assert_array_equal(target['points'],again['points'])


def test_single_observation_and_distinct_instance_tokens_do_not_form_a_track(tmp_path):
    frames=[frame((str(i),),(4,)) for i in range(4)]
    for i,f in enumerate(frames):f['timestamp_us']=np.int64(i*1000000)
    target=fixture_cache(tmp_path,frames)(results_for(frames))['endpoint_segments']
    assert target['points'].shape==(0,4,16,3)


def test_wrong_pose_or_split_and_missing_target_fail_closed(tmp_path):
    frames=[frame() for _ in range(4)]
    for i,f in enumerate(frames):f['timestamp_us']=np.int64(i*1000000)
    loader=fixture_cache(tmp_path,frames);r=results_for(frames);r['fut2cur'][2][0,3]=1
    with pytest.raises(ValueError,match='pose differs'):loader(r)
    r=results_for(frames);r['sample_idx']='validation-anchor'
    with pytest.raises(ValueError,match='configured endpoint split'):loader(r)
    with pytest.raises(ValueError,match='already loaded'):loader({})


def test_streaming_json_and_quaternion_conventions(tmp_path):
    records=[{'name':'a, ] escaped " value','v':[1,2]}, {'name':'二','v':[-1,2]}]
    path=tmp_path/'array.json';path.write_text(json.dumps(records))
    assert list(builder.iter_json_array(path,chunk_size=3))==records
    np.testing.assert_allclose(builder.rotation([np.sqrt(.5),0,0,np.sqrt(.5)]),[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-8)
    path.write_text('[{"broken":')
    with pytest.raises(ValueError,match='Incomplete'):list(builder.iter_json_array(path,chunk_size=2))
