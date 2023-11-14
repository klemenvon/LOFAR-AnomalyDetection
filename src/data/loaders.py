import math
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import interpolate
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from .utils import *

log = logging.getLogger(__name__)

class LOFARDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_conf: DictConfig,
            preproc_conf: DictConfig,
            loader_conf: DictConfig,
    ):
        """
        The PyTorch Lightning DataModule for LOFAR data which will be used for training.
        data_conf - (DictConfig) configuration for the data module
        preproc_conf - (DictConfig) configuration for the preprocessor
        loader_conf - (DictConfig) configuration for the data loader
        """
        super().__init__()
        preprocessor = Preprocessor(**preproc_conf)
        data_store = DataStore(**data_conf)
        self.full_data = LOFARDataset(data_store,preprocessor,uv=True)
        self.train_data, self.val_data, self.test_data = random_split(self.full_data,[0.8,0.1,0.1])
        self.loader_conf = OmegaConf.to_container(loader_conf,resolve=False)
        self.loader_conf['collate_fn'] = collate_fn[loader_conf.collate_fn]

    def train_dataloader(self):
        return DataLoader(self.train_data,**self.loader_conf)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,**self.loader_conf)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,**self.loader_conf)


class DataReader():
    def __init__(self,data_dir,uv=True,exclude=None,include=None,device=None):
        """
        Data reader class. Maps dataset and fetches data on demand from disk.
        data_dir - (str) string specifying where to look for data
        exclude - (str) specify string or substring to filter out from map
        include - (str) specify string or substring to include in map
        """
        self.device = device
        self.uv = uv
        log.info(f"Data reader will read all data to device: {self.device}")
        self.bl_list,self.map = get_dataset_map(data_dir,exclude=exclude,include=include)
        log.info(f"Mapped dataset in {data_dir}. {len(self.bl_list)} baselines.")
    
    def _get_baseline(self,handle,sap,bl_idx):
        vis = handle[f'measurement/saps/{sap}/visibilities']
        scale_fac = handle[f'measurement/saps/{sap}/visibility_scale_factors']
        baseline = torch.empty(size=vis.shape,dtype=torch.float64,device=self.device)
        sfac_torch = torch.from_numpy(scale_fac[bl_idx,:,:])
        baseline = torch.from_numpy(vis[bl_idx,:,:,:,:]) * sfac_torch[None,:,:,None]
        #baseline = torch.from_numpy(vis[bl_idx,:,:,:,:] * scale_fac[bl_idx,:,:])
        return baseline
    
    def _get_uv_coords(self,handle,sap,bl_idx):
        # Speed of light
        c=2.99792458e8
        # observation start time
        hms=handle['measurement']['info']['start_time'][0].decode('ascii').split()[1].split(sep=':')
        # time in hours, in [0,24]
        start_time=float(hms[0])+float(hms[1])/60.0+float(hms[2])/3600
        # convert to radians
        theta=start_time/24.0*(2*math.pi)
        # frequencies in Hz
        frq=handle['measurement']['saps'][sap]['central_frequencies']
        Nf0=frq.shape[0]//2
        # central frequency
        freq0=frq[Nf0]
        # 1/lambda=freq0/c
        inv_lambda=freq0/c
        # rotation matrix =[cos(theta) sin(theta); -sin(theta) cos(theta)]
        rot00=math.cos(theta)*inv_lambda
        rot01=math.sin(theta)*inv_lambda
        baselines=handle['measurement']['saps'][sap]['baselines']
        xyz=handle['measurement']['saps'][sap]['antenna_locations']['XYZ']
        # Make uv array
        uv = torch.zeros(1,2,device=self.device)
        xx=xyz[baselines[bl_idx][0]][0]-xyz[baselines[bl_idx][1]][0]
        yy=xyz[baselines[bl_idx][0]][1]-xyz[baselines[bl_idx][1]][1]
        uv[0,0]=xx*rot00+yy*rot01
        uv[0,1]=-xx*rot01+yy*rot00
        return uv
    
    def get_baseline(self,baseline_tuple,uv=None):
        sas,sap,bl = baseline_tuple
        ftgt = self.map[sas]
        with h5py.File(ftgt,'r') as f:
            baseline = self._get_baseline(f,sap,bl)
            uv = self._get_uv_coords(f,sap,bl)
        if uv is not None:
            return baseline,uv
        return baseline

class LOFARDataset(Dataset):
    def __init__(self,data,preprocessor,uv=False):
        self.uv = uv
        self.dataset = data
        self.preproc = preprocessor
    
    def __len__(self):
        return len(self.dataset.bl_list)
    
    def __getitem__(self,idx):
        bl_tup = self.dataset.bl_list[idx]
        if self.uv:
            baseline, uv = self.dataset.get_baseline(bl_tup,uv=self.uv)
            baseline, uv = self.preproc(baseline,uv)
            return baseline,uv
        baseline = self.dataset.get_baseline(bl_tup,uv=self.uv)
        baseline = self.preproc(baseline)
        return baseline

class Preprocessor():
    def __init__(self,mode='patch',size=(128,128),channels=4,device=None):
        self._processors = {
            'patch' : self._patch,
            'interp' : self._interpolate,
            'plain' : self._plain
        }
        self.size = size
        self.channels = channels
        self.processor = self._processors.get(mode)
        self.device = device
    
    @staticmethod
    def _flatten_complex(baseline):
        # Takes (ntime,nfreq,npol,ncomp)
        # Gives (ntime,nfreq,nchan)
        ntime,nfreq,npol,ncomp = baseline.shape
        baseline = torch.reshape(baseline,(ntime,nfreq,npol*ncomp))
        return baseline
    
    @staticmethod
    def _tile_uv(uv,shape):
        # Takes (coords)
        # Gives (repetitions,coords)
        if len(shape) == 3:
            return uv
        repl,ntime,nfreq,nchan = shape
        #uv = uv[None,:]
        if uv.shape[0] > repl:
            uv = uv[:repl]
        uv = torch.broadcast_to(uv,(repl,2))
        return uv
    
    @staticmethod
    def _cut_channels(baseline,n_channels=4):
        # Takes (ntime,nfreq,npol,ncomp)
        # Gives (ntime,nfreq,npol,ncomp)
        if n_channels == 4:
            baseline = baseline[:,:,[0,3],:]
        return baseline
    
    @staticmethod
    def _normalize(baseline):
        bmean = baseline.mean()
        bstd = baseline.std()
        baseline.sub_(bmean).div_(bstd)
        return baseline

    def _plain(self,baseline,uv=None):
        # Gets (nchan,ntime,nfreq)
        return baseline,uv
    
    def _interpolate(self,baseline,uv=None):
        # Gets (nchan,ntime,nfreq)
        baseline = torch.unsqueeze(baseline,0)
        # (1,nchan,ntime,nfreq)
        size_x,size_y = self.size
        tgt_shape = (size_x,size_y)
        baseline = interpolate(baseline,size=tgt_shape,mode='bilinear')
        #baseline = torch.squeeze(baseline)
        return baseline,uv
    
    def _patch(self,baseline,uv=None):
        # Gets (nchan,ntime,nfreq)
        nchan,ntime,nfreq = baseline.shape
        patch_size = self.size[0]
        stride = patch_size//2
        new_base = torch.zeros(nchan,max(ntime,patch_size),max(nfreq,patch_size),device=self.device)
        new_base[:,:ntime,:nfreq] = baseline[:,:,:]
        baseline = new_base
        #baseline = baseline.unfold(2,patch_size,stride).unfold(3,patch_size,stride)
        baseline = baseline.unfold(1,patch_size,stride).unfold(2,patch_size,stride)
        # (nchan,patchx,patchy,ntime,nfreq)
        baseline = torch.moveaxis(baseline,1,0)
        # (patchx,nchan,patchy,ntime,nfreq)
        baseline = torch.moveaxis(baseline,2,1)
        # (patchx,patchy,nchan,ntime,nfreq)
        px,py,nchan,ntime,nfreq = baseline.shape
        baseline = baseline.reshape(px*py,nchan,ntime,nfreq)
        # (npatch,nchan,ntime,nfreq)
        if uv is not None:
            uv = self._tile_uv(uv,baseline.shape)
        return baseline,uv

    def __call__(self,baseline,uv=None):
        """
        Given a baseline in the format (-1,ntime,nfreq,npol,comp)
        preprocess to the specified dimensions with the initialized method
        """
        # Basic pre-processing
        # (ntime,nfreq,npol,ncomp)
        baseline = self._cut_channels(baseline,self.channels)
        # (ntime,nfreq,npol,ncomp)
        baseline = self._flatten_complex(baseline)
        # (ntime,nfreq,nchan)
        baseline = baseline.moveaxis(-1,0)
        # (nchan,ntime,nfreq)
        # Final Pre-processing (depends on function)
        baseline,uv = self.processor(baseline,uv)
        # Normalize the data to avoid the issue of no convergence
        baseline = self._normalize(baseline)
        if uv is not None:
            return baseline,uv
        return baseline

class Observation():
    def __init__(self,path,sap,device=None):
        """
        Get an specific SAP of an observation and load it onto a device
        path - (str) path to the observation
        sap - (str/int) SAP to get data for
        device - (torch.device) device to store data on
        """
        self.sap = sap
        self.device = device
        with h5py.File(path,'r') as f:
            # Get the data itself
            self.vis,self.scale_fac = self._get_cube(f,sap)
            # Get the uv coordinates
            self.uv_coords = self._get_uv_coords(f,sap)
    
    def _get_cube(self,handle,sap):
        # Doesn't actually get cube, gets cube components
        vis = handle[f'measurement/saps/{sap}/visibilities']
        scale_fac = handle[f'measurement/saps/{sap}/visibility_scale_factors']
        # Create and fill visibilities
        vis_torch = torch.from_numpy(vis[:,:,:,:,:]).to(self.device,non_blocking=True)
        # Get scale factors too
        scale_fac_torch = torch.from_numpy(scale_fac[:,:,:]).to(self.device,non_blocking=True)
        return vis_torch,scale_fac_torch

    def _get_uv_coords(self,handle,sap):
        # Speed of light
        c=2.99792458e8
        # observation start time
        hms=handle['measurement']['info']['start_time'][0].decode('ascii').split()[1].split(sep=':')
        # time in hours, in [0,24]
        start_time=float(hms[0])+float(hms[1])/60.0+float(hms[2])/3600
        # convert to radians
        theta=start_time/24.0*(2*math.pi)
        # frequencies in Hz
        frq=handle['measurement']['saps'][sap]['central_frequencies']
        Nf0=frq.shape[0]//2
        # central frequency
        freq0=frq[Nf0]
        # 1/lambda=freq0/c
        inv_lambda=freq0/c
        # rotation matrix =[cos(theta) sin(theta); -sin(theta) cos(theta)]
        rot00=math.cos(theta)*inv_lambda
        rot01=math.sin(theta)*inv_lambda
        baselines=handle['measurement']['saps'][sap]['baselines']
        nbase = baselines.shape[0]
        xyz=handle['measurement']['saps'][sap]['antenna_locations']['XYZ']
        # Make uv array
        uv = torch.zeros(nbase,2,device=self.device)
        for b in range(nbase):
            xx=xyz[baselines[b][0]][0]-xyz[baselines[b][1]][0]
            yy=xyz[baselines[b][0]][1]-xyz[baselines[b][1]][1]
            uv[b,0]=xx*rot00+yy*rot01
            uv[b,1]=-xx*rot01+yy*rot00
        return uv

    def get_baseline(self,bl_idx,uv=True):
        """
        Method that is called from externals to get the baseline
        Calculates the multiplication of the visibilities and the scale factor
        returns the uv coordinates as well if prompted
        """
        baseline_vis = self.vis[bl_idx]
        baseline_fac = self.scale_fac[bl_idx]
        vis = baseline_fac[None,:,:,None] * baseline_vis
        if uv:
            return vis,self.uv_coords[bl_idx]
        return vis

class DataStore():
    def __init__(self,data_dir,uv=True,include=None,exclude=None,device=None,preprocessor=None):
        """
        data_dir - (str) data directory to look in
        uv - (bool) create UV ccordinate store?
        include - (str) include certain sub-directories specifically?
        exclude - (str) exclude certain sub-strings when searching for observations?
        device - (torch.device) what device to store the data on?
        """
        # Build data store
        self.device = device
        self.uv = uv
        self.preproc = preprocessor
        log.info(f"Creating data store on device: {self.device}")
        self.bl_list,self.map = get_dataset_map(data_dir,exclude=exclude,include=include)
        log.info(f"Mapped dataset in {data_dir}. {len(self.bl_list)} baselines.")
        sap_obs = list(set([tuple(bl[:2]) for bl in self.bl_list]))
        self.data_map = {sap_tup :self._get_obs(sap_tup) for sap_tup in sap_obs}
    
    def _get_obs(self,sap_tuple):
        """
        Gets observation object for a SAP tuple
        sap_tuple - (tuple) (observation,SAP)
        """
        observation, sap = sap_tuple
        fpath = self.map[observation]
        log.debug(f"Getting observation {observation} and SAP {sap}")
        return Observation(fpath,sap,self.device)
    
    def get_baseline(self,baseline_tuple,uv=None):
        obs, sap, bl_idx = baseline_tuple
        if self.preproc is not None:
            data = self.data_map[(obs,sap)].get_baseline(bl_idx,uv)
            if uv:
                return self.preproc(*data)
            else:
                return self.preproc(data)
        return self.data_map[(obs,sap)].get_baseline(bl_idx,uv)