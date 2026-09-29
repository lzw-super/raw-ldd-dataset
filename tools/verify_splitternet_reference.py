"""Verify RGB PyTorch SplitterNet against official H5/TF reference (no training)."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py,numpy as np,torch
parser=argparse.ArgumentParser()
parser.add_argument("repository",type=Path,help="Local clone of official SplitterNet repository")
args=parser.parse_args()
from models.paper_denoisers import SplitterNet
torch.set_num_threads(2)
f=h5py.File(args.repository/'model_weights/SplitterNet_MIDD_model.h5')
m=SplitterNet(channels=3)
for typ,prefix in [(torch.nn.Conv2d,'conv2d'),(torch.nn.ConvTranspose2d,'conv2d_transpose')]:
 modules=[v for v in m.modules() if isinstance(v,typ)]
 for i,layer in enumerate(modules):
  name=prefix+('_'+str(i) if i else '')
  grp=f['model_weights'][name]
  names=grp.attrs['weight_names']
  kernel=np.array(grp[names[0]]);bias=np.array(grp[names[1]])
  with torch.no_grad():
   layer.weight.copy_(torch.from_numpy(kernel.transpose(3,2,0,1)))
   layer.bias.copy_(torch.from_numpy(bias))
r=np.load(args.repository/'tests/data/splitternet_midd_reference.npz')
with torch.no_grad():out=m(torch.from_numpy(r['x']).permute(0,3,1,2)).permute(0,2,3,1).numpy()
print('Official TF reference max abs:',np.max(np.abs(out-r['y'])))
np.testing.assert_allclose(out,r['y'],atol=1e-5,rtol=1e-5)
