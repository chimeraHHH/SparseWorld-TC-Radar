"""Compare the real sm90 sampler's values/gradients against all-level grid_sample."""
import importlib.util
from pathlib import Path
import sys
import types
import torch

root = Path(__file__).resolve().parents[1]
package = types.ModuleType('sampler_check')
package.__path__ = [str(root / 'models/csrc')]
sys.modules['sampler_check'] = package
spec = importlib.util.spec_from_file_location('sampler_check.wrapper', root / 'models/csrc/wrapper.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
assert wrapper.MSMV_CUDA
assert torch.cuda.get_device_capability() == (9, 0)
torch.manual_seed(42)
features = [torch.randn(2, 6, h, h+1, 8, device='cuda', requires_grad=True) for h in (9, 5, 3, 2)]
locations = torch.rand(2, 7, 4, 3, device='cuda') * .8 + .1
locations[..., 2] = torch.randint(0, 6, (2, 7, 4), device='cuda') / 5
locations.requires_grad_()
weights = torch.randn(2, 7, 4, 4, device='cuda').softmax(-1).requires_grad_()
actual = wrapper.msmv_sampling(features, locations, weights)
reference = wrapper.msmv_sampling_pytorch([f.permute(0,4,1,2,3).contiguous() for f in features], locations, weights)
torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
probe = torch.randn_like(actual)
a = torch.autograd.grad((actual * probe).sum(), features + [locations, weights], retain_graph=True)
b = torch.autograd.grad((reference * probe).sum(), features + [locations, weights])
for i, (one, two) in enumerate(zip(a,b)):
    # View selection is discrete by design, so only XY derivatives are comparable.
    if i == len(features): one,two=one[...,:2],two[...,:2]
    torch.testing.assert_close(one,two,atol=1e-4,rtol=1e-4)
print('SAMPLER_CUDA_FORWARD_BACKWARD_PARITY_PASSED', torch.cuda.get_device_name(), flush=True)
