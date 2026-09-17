import numpy as np
import torch
from loaders.pipelines.radar import transform_radar, LoadCausalRadar
from models.sparse_world_transformer import SparseWorldTransformer


def test_position_velocity_and_doppler_frames():
    raw = np.zeros((18, 1))
    raw[0] = 1
    raw[8] = 2
    T = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    xyz, vel, los, radial = transform_radar(raw, T)
    np.testing.assert_allclose(xyz, [[10, 21, 3]])
    np.testing.assert_allclose(vel, [[0, 2, 0]])
    np.testing.assert_allclose(los, [[0, 1, 0]])
    np.testing.assert_allclose(radial, [2])


def test_causal_skip_does_not_consume_sweep_budget(monkeypatch, tmp_path):
    import loaders.pipelines.radar as radar_module
    class FakeNuScenes:
        def get(self, table, token):
            if table == 'sample':
                return {'data': {'LIDAR_TOP': 'lidar', **{c:'future' for c in radar_module.RADAR_CHANNELS}}}
            if table == 'sample_data':
                if token == 'lidar': return {'timestamp':1000000,'ego_pose_token':'pose'}
                return {'timestamp':1100000 if token=='future' else 900000,
                    'prev':'past' if token=='future' else '', 'ego_pose_token':'pose',
                    'calibrated_sensor_token':'calib', 'filename':token}
            return {'translation':[0,0,0], 'rotation':[1,0,0,0]}
    reads=[]
    class PointCloud:
        @classmethod
        def from_file(cls,path):
            reads.append(path)
            instance=cls()
            instance.points=np.zeros((18,1));instance.points[0]=5
            return instance
    monkeypatch.setattr(radar_module,'get_nusc',lambda *a: FakeNuScenes())
    monkeypatch.setattr(radar_module,'RadarPointCloud',PointCloud)
    data=LoadCausalRadar(str(tmp_path),sweeps_num=1)({'sample_idx':'sample'})
    assert len(reads)==5 and all(path.endswith('past') for path in reads)
    np.testing.assert_allclose(data['radar_points'][:,6],.1)


def synthetic_inputs(batch, device):
    torch.manual_seed(31)
    feats=[torch.randn(batch,12,256,8//(2**i),8//(2**i),device=device) for i in range(4)]
    points=torch.rand(batch,7,1,3,device=device)
    query=torch.randn(batch,7,256,device=device)
    projection=np.array([[8,0,8,0],[0,8,8,0],[0,0,1,10],[0,0,0,1]],dtype=np.float32)
    meta=[dict(lidar2img=[projection]*12,ego2lidar=np.eye(4), img_shape=[(16,16,3)]*12,
               radar_points=np.zeros((0,10),np.float32)) for _ in range(batch)]
    poses=[torch.eye(4,device=device).repeat(batch,1,1) for _ in range(2)]
    horizons=[torch.full((batch,),t,device=device) for t in (0,2)]
    return points,query,feats,meta,poses,horizons


def test_empty_radar_full_decoder_matches_camera_and_batch_isolation():
    # CPU tests use the corrected all-level PyTorch sampler explicitly.
    import models.sparse_world_transformer as transformer_module
    import models.sparse_world_sampling as sampling_module
    original_cuda=transformer_module.MSMV_CUDA
    transformer_module.MSMV_CUDA=False
    original_sampler=sampling_module.msmv_sampling
    sampling_module.msmv_sampling=sampling_module.msmv_sampling_pytorch
    try:
        args=dict(embed_dims=256,num_frames=2,future_frames=[0,2],num_layers=2,
            num_refines=[1,4],num_classes=17,num_points=2,pc_range=[-40,-40,-1,40,40,5.4])
        torch.manual_seed(10)
        camera=SparseWorldTransformer(**args).eval()
        torch.manual_seed(10)
        radar=SparseWorldTransformer(**args,radar_cfg={}).eval()
        for name, value in camera.state_dict().items():
            torch.testing.assert_close(value, radar.state_dict()[name], rtol=0, atol=0)
        radar.load_state_dict(camera.state_dict(),strict=False)
        inputs=synthetic_inputs(2,'cpu')
        def run(net, inp):
            p,q,f,m,t,h=inp
            return net(p,q,[x.clone() for x in f],m,t,h)
        with torch.no_grad():
            out_camera=run(camera,inputs)
            out_radar=run(radar,inputs)
            for a,b in zip(out_camera[0]+out_camera[1],out_radar[0]+out_radar[1]):
                torch.testing.assert_close(a,b)
            for b in range(2):
                p,q,f,m,t,h=inputs
                alone=run(radar,(p[b:b+1],q[b:b+1],[x[b:b+1] for x in f],[m[b]],
                    [x[b:b+1] for x in t],[x[b:b+1] for x in h]))
                for combined,single in zip(out_radar[0]+out_radar[1],alone[0]+alone[1]):
                    torch.testing.assert_close(combined[[b,b+2]],single,atol=2e-5,rtol=2e-5)
    finally:
        sampling_module.msmv_sampling=original_sampler
        transformer_module.MSMV_CUDA=original_cuda


def test_camera_metadata_rotation_matches_loader_projection():
    import importlib.util
    from pathlib import Path
    from pyquaternion import Quaternion
    path = Path(__file__).parents[1] / 'tools/prepare_m0_data.py'
    spec = importlib.util.spec_from_file_location('prepare_m0_data_test', path)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    class Tables:
        def get(self, table, token):
            if table == 'calibrated_sensor':
                return dict(translation=[1,2,0], rotation=Quaternion(axis=[0,0,1], angle=np.pi/2).elements,
                            camera_intrinsic=np.eye(3).tolist())
            return dict(translation=[0,0,0], rotation=[1,0,0,0])
    cam=prep.camera_info(Tables(),dict(calibrated_sensor_token='c',ego_pose_token='e',timestamp=0,filename='x'),Path('.'),np.eye(4))
    lidar2cam_r=np.linalg.inv(cam['sensor2lidar_rotation'])
    lidar_point=np.array([1.,3.,2.])
    projected=lidar2cam_r @ (lidar_point-cam['sensor2lidar_translation'])
    # 90deg camera-to-ego rotation: (0,1,2) ego offset -> (1,0,2) camera.
    np.testing.assert_allclose(projected,[1,0,2],atol=1e-8)
