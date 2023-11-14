import os
import glob
import logging
from functools import partial

import h5py
import numpy as np
import torch

log = logging.getLogger(__name__)

def get_metadata(filename,SAP,give_baseline=False):
  # open LOFAR H5 file, read metadata from a SAP,
  # return number of baselines, time, frequencies, polarizations, real/imag
  # if give_baseline=True, also return ndarray of baselines

  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']

  if give_baseline:
    baselines=f['measurement']['saps'][SAP]['baselines']
    (nbase,_)=baselines.shape
    bline=np.ndarray(baselines.shape,dtype=object)
    for ci in range(nbase):
      bline[ci]=baselines[ci]
    return bline,g.shape
  return g.shape

def get_fileSAP(pathname,pattern='L*.MS_extract.h5',exclude=None,include=None,rec_file_search=True,unfiltered=False):
  # search in pathname for files matching 'pattern'
  # test valid SAPs in each file and
  # return file_list,sap_list for valid files and their SAPs
  file_list=[]
  sap_list=[]
  if rec_file_search:
    rawlist = glob.glob(pathname+'**'+os.sep+pattern,recursive=True)
  else:
    rawlist=glob.glob(pathname+os.sep+pattern)
  # open each file and check valid saps
  if exclude:
    # Filter for excluded strings
    rawlist = [f for f in rawlist if exclude not in f]
  if include:
    # Filter for included strings
    rawlist = [f for f in rawlist if include in f]
  # We need to check for duplicates
  nodup = {}
  for f in rawlist:
    nodup[f[-21:]] = f
  rawlist = [v for _,v in nodup.items()]
  # Get meta and see if useful
  for filename in rawlist:
    log.debug(f"[get_fileSAP] Processing file {filename}.")
    f=h5py.File(filename,'r')
    g=f['measurement']['saps']
    SAPs=[SAP for SAP in g]
    # flag to remember if this file is useful
    fileused=False
    if len(SAPs)>0:
     for SAP in SAPs:
      if unfiltered:
        # Just add it
        file_list.append(filename)
        sap_list.append(SAP)
        fileused=True
        continue
      # Otherwise filter for usable
      try:
       vis=f['measurement']['saps'][SAP]['visibilities']
       (nbase,ntime,nfreq,npol,reim)=vis.shape
       # select valid datasets (larger than 90 say)
       if nbase>1 and nfreq>=90 and ntime>=90 and npol==4 and reim==2:
         file_list.append(filename)
         sap_list.append(SAP)
         fileused=True
      except:
       log.error('Failed opening'+filename)
    
    if not fileused:
      # To avoid this being printed every time
      log.debug('File '+filename+' not used') 

  return file_list,sap_list

def get_dataset_map(base_path, exclude=None,include=None,unfiltered=False):
  """
  Maps the dataset so that we know exactly how much data we're working with
  and how to find it.
  :param str base_path: Folder where to look for all the data
  :param str exclude: Folders or sub-paths to exclude from the search
  :return total_list : list (sas_id,sap,baseline), file_map : { sas_id : file_path }
  """
  file_list,sap_list = get_fileSAP(base_path,exclude=exclude,include=include,unfiltered=unfiltered)
  total_list = []
  file_map = {}
  rev_map = {}
  sas_processed = []

  unique_files = list(set(file_list))

  # Map sas_ids and file paths
  for f in unique_files:
    # Read h5 file, extract sas_id and then use that as a key for the map
    sas = h5py.File(f,'r')['measurement/sas_id'][0]
    file_map[sas] = f
    rev_map[f] = sas
  
  for f,s in zip(file_list,sap_list):
    nbase,_,_,_,_ = get_metadata(f,s)
    # Creates a list of tuples of (sas_id,sap,)
    bl_list = [(a,s,b) for a,s,b in zip([rev_map[f]]*nbase,[s]*nbase,range(nbase))]
    total_list += bl_list
  len_before = len(total_list)
  total_list = list(set(total_list))
  len_after = len(total_list)
  log.debug(f"Removed {len_before-len_after} duplicate mapping entries.")
  return total_list,file_map

def collate_batch_patch(batch,mydevice):
    # Unpack batch (keep in mind these are all lists/arrays)
    #patchx,patchy,xlist,uvcoords = batch
    xlist = [x[0] for x in batch]
    uvcoords = [x[1] for x in batch]

    # Get the total size of the batch
    #total_batch = sum([x*y for x,y in zip(patchx,patchy)])
    xtens = torch.cat(xlist,dim=0).to(mydevice,non_blocking=True)
    utens = torch.cat(uvcoords,dim=0).to(mydevice,non_blocking=True)
    return [xtens,utens]

collate_fn = {
    'patch': collate_batch_patch,
}

def setup_collate(fn_name, device):
  """
  Sets up the collate function for the dataset.
  Returns a callable that can be used with the DataLoader.
  """
  return partial(collate_fn[fn_name],mydevice=device)

def update_collate(device):
    global collate_fn
    for name, func in collate_fn.items():
      collate_fn[name] = partial(func,mydevice=device)
