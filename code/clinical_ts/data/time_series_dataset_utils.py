__all__ = [ 'check_and_clean_numpy_files','save_dataset', 'load_dataset','dataset_add_chunk_col', 'dataset_add_length_col', 'dataset_add_labels_col', 'dataset_add_mean_col', 'dataset_add_median_col', 'dataset_add_std_col', 'dataset_add_iqr_col', 'dataset_get_stats', 'append_to_memmap', 'append_to_df_memmap', 'upgrade_memmap_meta', 'npys_to_memmap_batched', 'npys_to_memmap', 'reformat_as_memmap', 'load_memmap_meta_dict']

MEMMAP_META_VERSION = "1.0"

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import iqr


#workaround for windows pickles
from sys import platform
import pathlib
if platform == "linux" or platform == "linux2":
    pathlib.WindowsPath = pathlib.PosixPath

try:
    import pickle5 as pickle
except ImportError as e:
    import pickle

from tqdm.auto import tqdm
import os

def check_and_clean_numpy_files(directory):
    """
    Check .npy files in directory and delete corrupted ones.
    
    Args:
        directory (str): Path to directory to check
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    corrupted_files = []
    checked_files = 0
    deleted_files = 0
    
    print(f"Checking .npy files in: {directory}")
    print("-" * 50)
    
    # Find all .npy files
    numpy_files = list(directory.glob("**/*.npy"))
    
    if not numpy_files:
        print("No .npy files found")
        return
    
    for file_path in tqdm(numpy_files):
        checked_files += 1
        try:
            # Try to load the .npy file
            data = np.load(file_path)
            #print(f"✓ OK: {file_path} (shape: {data.shape})")
                
        except (EOFError, OSError, ValueError) as e:
            # These are the common errors for corrupted numpy files
            error_type = type(e).__name__
            print(f"✗ CORRUPTED: {file_path} - {error_type}: {str(e)}")
            corrupted_files.append((file_path, error_type, str(e)))
            
            try:
                os.remove(file_path)
                print(f"  → Deleted: {file_path}")
                deleted_files += 1
            except OSError as delete_error:
                print(f"  → Failed to delete {file_path}: {delete_error}")
                
        except Exception as e:
            # Catch any other unexpected errors
            print(f"? UNKNOWN ERROR: {file_path} - {type(e).__name__}: {str(e)}")
    
    # Summary
    print("-" * 50)
    print(f"Summary:")
    print(f"  Files checked: {checked_files}")
    print(f"  Corrupted files found: {len(corrupted_files)}")
    print(f"  Files deleted: {deleted_files}")
    
    if corrupted_files:
        print(f"\nCorrupted files details:")
        for file_path, error_type, error_msg in corrupted_files:
            print(f"  - {file_path}: {error_type}")

def save_dataset(df,lbl_itos=None,mean=None,std=None,target_root=".",df_filename="df.pkl",protocol=4):
    target_root = Path(target_root)
    df.to_pickle(target_root/df_filename, protocol=protocol)

    if(isinstance(lbl_itos,dict)):#dict as pickle
        outfile = open(target_root/("lbl_itos.pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    elif(lbl_itos is not None):#array
        np.save(target_root/("lbl_itos.npy"),lbl_itos)

    if(mean is not None):
        np.save(target_root/("mean.npy"),mean)
    if(std is not None):
        np.save(target_root/("std.npy"),std)

def load_dataset(target_root,df_filename="df_memmap.pkl"):
    target_root = Path(target_root)

    df = pd.read_pickle(target_root/df_filename)
    
    if((target_root/("lbl_itos.pkl")).exists()):#dict as pickle
        infile = open(target_root/("lbl_itos.pkl"), "rb")
        lbl_itos=pickle.load(infile)
        infile.close()
    elif((target_root/("lbl_itos.npy")).exists()):
        lbl_itos = np.load(target_root/("lbl_itos.npy"))
    else:#array
        lbl_itos = None

    if((target_root/("mean.npy")).exists()):
        mean = np.load(target_root/("mean.npy"))
    else:
        mean = None
    if((target_root/("std.npy")).exists()):
        std = np.load(target_root/("std.npy"))
    else:
        std = None
    return df, lbl_itos, mean, std

def dataset_add_chunk_col(df, col="data"):
    '''add a chunk column to the dataset df'''
    df["chunk"]=df.groupby(col).cumcount()

def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col+"_length"]=df[col].apply(lambda x: len(np.load(x if data_folder is None else data_folder/x, allow_pickle=True)))

def dataset_add_labels_col(df, col="label", data_folder=None):
    '''add a column with unique labels in column col'''
    df[col+"_labels"]=df[col].apply(lambda x: list(np.unique(np.load(x if data_folder is None else data_folder/x, allow_pickle=True))))

def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_mean"]=df[col].apply(lambda x: np.mean(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_median_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with median'''
    df[col+"_median"]=df[col].apply(lambda x: np.median(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_std"]=df[col].apply(lambda x: np.std(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_iqr_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_iqr"]=df[col].apply(lambda x: iqr(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if(simple):
        return df[col+"_mean"].mean(), df[col+"_std"].mean()
    else:
        #https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        #or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1,x2):
            (mean1,var1,n1) = x1
            (mean2,var2,n2) = x2
            mean = mean1*n1/(n1+n2)+ mean2*n2/(n1+n2)
            var = var1*n1/(n1+n2)+ var2*n2/(n1+n2)+n1*n2/(n1+n2)/(n1+n2)*np.power(mean1-mean2,2)
            return (mean, var, (n1+n2))

        def combine_all_means_vars(means,vars,lengths):
            inputs = list(zip(means,vars,lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result= combine_two_means_vars(result,inputs2)
            return result

        means = list(df[col+"_mean"])
        vars = np.power(list(df[col+"_std"]),2)
        lengths = list(df[col+"_length"])
        mean,var,length = combine_all_means_vars(means,vars,lengths)
        return mean, np.sqrt(var)


def npys_to_memmap_batched(npys, target_filename, fs, channel_itos=None, max_len=0, delete_npys=True, batched_npy=False, batch_length=900000):
    '''
    analogous to npys_to_memmap but processes batches of files before flushing them into memmap for faster processing
    '''
    memmap = None
    start = np.array([0])#start_idx in current memmap file (always already the next start- delete last token in the end)
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]#shapes of all memmap files

    data = []
    data_lengths=[]
    dtype = None

    target_filename = Path(target_filename)

    for idx,npy in tqdm(enumerate(npys),total=len(npys)):
        data_batched = np.load(npy, allow_pickle=True)

        for data_tmp in (tqdm(data_batched,leave=False) if batched_npy else [data_batched]):
            data.append(data_tmp)
            data_lengths.append(len(data[-1]))
            if(idx==len(npys)-1 or np.sum(data_lengths)>batch_length):#flush
                data = np.concatenate(data,axis=0)#concatenate along time axis (still axis 0 at this stage)
                if(memmap is None or (max_len>0 and start[-1]>max_len)):#new memmap file has to be created
                    if(max_len>0):
                        filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames))+".npy"))
                    else:
                        filenames.append(target_filename)

                    shape.append([np.sum(data_lengths)]+[l for l in data.shape[1:]])#insert present shape

                    if(memmap is not None):#an existing memmap exceeded max_len
                        del memmap
                    #create new memmap
                    start[-1] = 0
                    start = np.concatenate([start,np.cumsum(data_lengths)])
                    length = np.concatenate([length,data_lengths])

                    memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
                else:
                    #append to existing memmap
                    start = np.concatenate([start,start[-1]+np.cumsum(data_lengths)])
                    length = np.concatenate([length,data_lengths])
                    shape[-1] = [start[-1]]+[l for l in data.shape[1:]]
                    memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple(shape[-1]))

                #store mapping memmap_id to memmap_file_id
                file_idx=np.concatenate([file_idx,[(len(filenames)-1)]*len(data_lengths)])
                #insert the actual data
                memmap[start[-len(data_lengths)-1]:start[-len(data_lengths)-1]+len(data)]=data[:]
                memmap.flush()
                dtype = data.dtype
                data = []#reset data storage
                data_lengths = []

    start= start[:-1]#remove the last element
    #cleanup
    for npy in npys:
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    meta = {
        "version": MEMMAP_META_VERSION,
        "fs": fs,
        "channel_itos": channel_itos
    }
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=dtype,filenames=filenames,meta=meta)

def append_to_memmap(memmap1, memmap2, file_id1=0, file_id2=0):
    '''
    appends the contents of memmap2(file_id2) to memmap1(file_id1 in case of split files)
    '''
    memmap1 = Path(memmap1)
    memmap2 = Path(memmap2)

    meta1 = np.load(memmap1.parent/(memmap1.stem+"_meta.npz"),allow_pickle=True)
    meta2 = np.load(memmap2.parent/(memmap2.stem+"_meta.npz"),allow_pickle=True)
    meta1_shape = meta1["shape"][file_id1]
    meta2_shape = meta2["shape"][file_id2]
    assert((len(meta1_shape)==1 and len(meta2_shape)==1) or meta1_shape[1:]==meta2_shape[1:])#shapes have to match up to length
    assert(meta1["dtype"]==meta2["dtype"])#dtypes have to agree
    mask = np.where(meta2["file_idx"]==file_id2)[0]
    lengths2 = np.array(meta2["length"])[mask]
    shape = np.concatenate(([meta1_shape[0]+np.sum(lengths2)],meta1_shape[1:])).astype(np.int64)
    full_shape=[(shape if i==file_id1 else m) for i,m in enumerate(meta1["shape"])]
    starts2 = meta1_shape[0]+np.concatenate(([0],np.cumsum(lengths2)[:-1]))
    start = np.concatenate((meta1["start"],starts2))
    length = np.concatenate((meta1["length"],lengths2))
    file_idx= np.concatenate((meta1["file_idx"],np.array([file_id1]*len(mask))))
    print("Appending",memmap2,"to",memmap1,"...")
    memmap_extended = np.memmap(memmap1.parent/(meta1["filenames"][file_id1]), dtype=np.dtype(str(meta1["dtype"])), mode='r+', shape=tuple(shape))
    memmap_source = np.memmap(memmap2.parent/(meta2["filenames"][file_id2]), dtype=np.dtype(str(meta2["dtype"])), mode="r", shape=tuple(meta2_shape))
    memmap_extended[meta1_shape[0]:] = memmap_source
    memmap_extended.flush()

    np.savez(memmap1.parent/(memmap1.stem+"_meta.npz"),start=start,length=length,shape=full_shape,file_idx=file_idx,dtype=meta1["dtype"],filenames=meta1["filenames"])
    print("done.")

def append_to_df_memmap(path_df_memmap1,path_df_memmap2,path_memmap1,path_memmap2,file_id1=0,file_id2=0,col_data="data"):
    df_memmap1 = pd.read_pickle(path_df_memmap1).sort_values(by=[col_data])
    df_memmap2 = pd.read_pickle(path_df_memmap2).sort_values(by=[col_data])
    path_memmap1 = Path(path_memmap1)
    path_memmap2 = Path(path_memmap2)
    
    meta1 = np.load(path_memmap1.parent/(path_memmap1.stem+"_meta.npz"),allow_pickle=True)
    meta2 = np.load(path_memmap2.parent/(path_memmap2.stem+"_meta.npz"),allow_pickle=True)
    df_memmap2 = df_memmap2.iloc[np.where(meta2["file_idx"]==file_id2)].copy()
    file_idx1 = meta1["file_idx"]
    assert(len(df_memmap1)==len(file_idx1))# apply append_to_df_memmap before append_to_memmap
    data_idx_start = np.max(np.where(file_idx1<=file_id1)[0])+1
    df_memmap1.loc[df_memmap1[col_data]>=data_idx_start,col_data]=df_memmap1.loc[df_memmap1[col_data]>=data_idx_start,col_data]+len(df_memmap2)
    
    df_memmap2[col_data]=df_memmap2[col_data]-df_memmap2[col_data].min()+data_idx_start
    df_memmap1 = pd.concat((df_memmap1,df_memmap2))
    return df_memmap1

def upgrade_memmap_meta(npz_file, fs, channel_itos=None):
    '''adds or updates meta dict (with version, fs, channel_itos) in an existing memmap_meta file, and removes any old fs field'''
    # Load the existing NPZ file
    data = np.load(npz_file, allow_pickle=True)
    
    # Create a dictionary with all existing arrays except 'fs' and 'meta'
    arrays_dict = {}
    for key in data.files:
        if key not in ["fs", "meta"]:
            arrays_dict[key] = data[key]
    
    # Add the new 'meta' parameter
    meta = {
        "version": MEMMAP_META_VERSION,
        "fs": fs,
        "channel_itos": channel_itos
    }
    arrays_dict['meta'] = meta
    
    # Close the original file before overwriting
    data.close()
    
    # Save the updated data back to the same file
    np.savez(npz_file, **arrays_dict)
    
    print(f"Successfully upgraded meta dict in {npz_file}")
    return list(arrays_dict.keys())

def npys_to_memmap(npys, target_filename, fs, channel_itos=None, max_len=0, delete_npys=True, batched_npy=False):
    '''
    fs: sampling frequency now required argument
    channel_itos: list of channel names (for mapping to integers)
    converts list of filenames pointing to npy files into a memmap file with target_filename
    max_len: restricts filesize per memmap file (0 no restriction)
    delete_npys: deletes original npys after processing to save space
    batched_npy: assumes first axis in the npy file enumerates samples (otherwise just a single sample per npy file)
    '''
    memmap = None
    start = []#start_idx in current memmap file
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]

    target_filename = Path(target_filename)

    for _,npy in tqdm(enumerate(npys),total=len(npys)):
        data_batched = np.load(npy, allow_pickle=True)
        for data in tqdm(data_batched,leave=False) if batched_npy else[data_batched]:
            if(memmap is None or (max_len>0 and start[-1]+length[-1]>max_len)):
                if(max_len>0):
                    filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames))+".npy"))
                else:
                    filenames.append(target_filename)

                if(memmap is not None):#an existing memmap exceeded max_len
                    shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
                    del memmap
                #create new memmap
                start.append(0)
                length.append(data.shape[0])
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
            else:
                #append to existing memmap
                start.append(start[-1]+length[-1])
                length.append(data.shape[0])
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple([start[-1]+length[-1]]+[l for l in data.shape[1:]]))

            #store mapping memmap_id to memmap_file_id
            file_idx.append(len(filenames)-1)
            #insert the actual data
            memmap[start[-1]:start[-1]+length[-1]]=data[:]
            memmap.flush()
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #append final shape if necessary
    if(len(shape)<len(filenames)):
        shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    meta = {
        "version": MEMMAP_META_VERSION,
        "fs": fs,
        "channel_itos": channel_itos
    }
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=data.dtype,filenames=filenames,meta=meta)

def reformat_as_memmap(df, target_filename, fs, channel_itos=None, fs_annotation=None, channel_itos_annotation=None, data_folder=None, annotation=False, max_len=900000000, delete_npys=True, col_data="data", col_lbl="label", batch_length=0, skip_export_signals=False):
    #note: max_len refers to time steps to keep consistency between labels and signals
    target_filename = Path(target_filename)
    data_folder = Path(data_folder)
    
    npys_data = []
    npys_label = []

    for _,row in df.iterrows():
        npys_data.append(data_folder/row[col_data] if data_folder is not None else row[col_data])
        if(annotation):
            npys_label.append(data_folder/row[col_lbl] if data_folder is not None else row[col_lbl])
    if(not(skip_export_signals)):
        if(batch_length==0):
            npys_to_memmap(npys_data, target_filename, fs=fs, channel_itos=channel_itos, max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_data, target_filename, fs=fs, channel_itos=channel_itos, max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)
    if(annotation):
        assert(fs_annotation is not None)
        if(batch_length==0):
            npys_to_memmap(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), fs=fs_annotation, channel_itos=channel_itos_annotation, max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), fs=fs_annotation, channel_itos=channel_itos_annotation, max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)

    #replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped[col_data+"_original"]=df_mapped[col_data]
    df_mapped[col_data]=np.arange(len(df_mapped))
    if(annotation):#for consistency also map labels (even though indexing will be via col_data)
        df_mapped[col_lbl+"_original"]=df_mapped[col_lbl]
        df_mapped[col_lbl]=np.arange(len(df_mapped))
        
    df_mapped.to_pickle(target_filename.parent/("df_"+target_filename.stem+".pkl"))
    return df_mapped

def load_memmap_meta_dict(memmap_filename):
    """
    Loads the 'meta' dictionary from the memmap meta file associated with the given memmap_filename.
    Also attempts to load annotation meta if a corresponding label meta file exists.
    If the meta file is not found, raises FileNotFoundError.
    If the meta file exists but contains no 'meta' key, prints a warning and returns an empty dict.
    If the annotation meta file is not found, skips annotation keys.
    """
    memmap_filename = Path(memmap_filename)
    meta_file = memmap_filename.parent / f"{memmap_filename.stem}_meta.npz"
    label_meta_file = memmap_filename.parent / f"{memmap_filename.stem}_label_meta.npz"

    if not meta_file.exists():
        raise FileNotFoundError(f"Meta file '{meta_file}' not found.")
    meta_npz = np.load(meta_file, allow_pickle=True)
    if "meta" not in meta_npz:
        print(f"Warning: No 'meta' key found in {meta_file}. Returning empty dict.")
        return {}
    meta = meta_npz["meta"].item()
    combined_meta = dict(meta)

    if label_meta_file.exists():
        annotation_npz = np.load(label_meta_file, allow_pickle=True)
        if "meta" in annotation_npz:
            annotation_meta = annotation_npz["meta"].item()
            for k, v in annotation_meta.items():
                if k == "version":
                    continue  # skip annotation version
                combined_meta[f"annotation_{k}"] = v
    return combined_meta

