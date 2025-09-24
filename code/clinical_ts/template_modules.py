__all__ = ['_string_to_class', 'BaseConfig', 'BaseConfigData', 'TrainerConfig', 'TaskConfig', 'LossConfig', 'SSLLossConfig', 'ShapeConfig', 'StaticStats', 'EncoderBase', 'EncoderBaseConfig', 'PrePostBase', 'PrePostBaseConfig', 'EncoderStaticBase', 'EncoderStaticBaseConfig', 'PredictorBase', 'PredictorBaseConfig', 'HeadBase', 'HeadBaseConfig', 'QuantizerBase', 'QuantizerBaseConfig', 'MaskingBaseConfig', 'TimeSeriesEncoder', 'TimeSeriesEncoderConfig', 'SequentialTimeSeriesEncoder', 'EMATimeSeriesEncoderBaseConfig']

from torch import nn

import dataclasses
from dataclasses import dataclass, field
from typing import List, Any, Union

import importlib

############################################################################################################

def _string_to_class(target: str) -> type:
    """Convert a fully qualified class name string to its corresponding class.
    
    Args:
        target: A string representing a class name. Can be either:
            - A simple class name (e.g., 'MyClass') - will be looked up in the current module
            - A fully qualified class name (e.g., 'package.module.MyClass')
    
    Returns:
        The class object corresponding to the given string.
        
    Raises:
        ValueError: If the input string is empty or malformed.
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
        KeyError: If the class cannot be found in the current module's globals.
    """
    if not target or not isinstance(target, str):
        raise ValueError("Input must be a non-empty string")
        
    parts = target.split(".")
    if not parts:
        raise ValueError("Invalid class name format")
        
    try:
        if len(parts) == 1:
            # Look up in current module's globals
            if target not in globals():
                raise KeyError(f"Class '{target}' not found in current module")
            return globals()[target]
        else:
            # Import the module and get the class
            module_path = ".".join(parts[:-1])
            class_name = parts[-1]
            
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(f"Failed to import module '{module_path}': {str(e)}")
                
            if not hasattr(module, class_name):
                raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")
                
            return getattr(module, class_name)
            
    except Exception as e:
        raise type(e)(f"Failed to convert string '{target}' to class: {str(e)}")
        
############################################################################################################
@dataclass
class BaseConfig:
    #optimizer
    optimizer:str ='adam' # sgd/adam
    auc_maximization:bool=False # direct auc maximization
    lr:List[float] = field(default_factory=lambda: [1e-3]) # initial learning rate, can pass a list of three element to specify learning rates for encoder, predictor, head separately
    weight_decay:float = 1e-3 # weight decay
    lr_schedule:str = "const" # const/const-plateau/warmup-const/warmup-cos/warmup-cos-restart/warmup-poly
    lr_num_warmup_steps:int =1000 # number of linear lr warmup steps
    discriminative_lr_factor:float = 0.1 # factor by which the lr decreases per layer group during finetuning

    train_head_only:bool = False # freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)
    linear_eval:bool = False # linear evaluation instead of full finetuning
    
    #for dataloader preparation
    batch_size:int = 64 # mini-batch size
    #metrics
    metric_checkpointing:str = "" # metric used for model selection/checkpointing (by default first entry from metrics)
    aggregate_strided_multi_predictions:bool = True # aggregate overlapping predictions (if stride_valtest<input_size for multi_predictions)

    #input sizes etc
    fs:float=100. # sampling frequency
    input_size: int | float = 1000 # number of input tokens (int) or length in seconds (float) passed to the model
    input_size_dl: int | float = -1 # number of input tokens (int) or length in seconds (float) extracted by the dataloader (-1: means input_size); e.g. in case transformations (such as conversion to spectrogram or resampling on the fly) are applied the two might differ
    #input_size_max: Union[int, float] = 0 # maximum input size the model is able to process (int: tokens, float: seconds); default value 0 means setting this to input_size
    chunk_length_train: int | float = 0 # 0: no chunkify, -1: input_size; can be int (tokens) or float (seconds)
    chunk_length_val: int | float = -1 # 0: no chunkify, -1: input_size (for validation during training); can be int (tokens) or float (seconds)
    chunk_length_valtest: int | float = -1 # (both val and test during inference); can be int (tokens) or float (seconds)
    stride_train: int | float = -1 # -1: chunk_length_train; can be int (tokens) or float (seconds)
    stride_val: int | float = -1 # -1: chunk_length_valtest (for validation during training); can be int (tokens) or float (seconds)
    stride_valtest: int | float = -1 # (both val and test during inference); can be int (tokens) or float (seconds)
    stride_export: int | float = -1 # -1: input_size; can be int (tokens) or float (seconds)

    input_channels:int = 12 # NOTE: refers to the input channels passed to the model i.e. should coincide with len(input_channels_filter) if non-zero; should be freq_bins*channels if passed to a NoEncoder
    freq_bins:int = 0 # number of frequency bins in the case of spectrogram input
    input_channels_cat: int = 0 # nnumber of categorical input channels
    input_channels_cont: int = 0 # number of continuous input channels
    normalize:bool = True # normalize input signal

    #label aggregation i.e. aggregate some number of labels into a new one
    label_aggregation_epoch_length:int= -1 # how labels in the original sequence should be aggregated- 0: all -1: no label_aggregation
    label_aggregation_majority_vote:bool = False # decide on segment label via majority vote (i.e. single label per segment)
    label_aggregation_binary:bool = False # only count which segments are present (multi-label) rather than for what fraction of steps

    #global dataloader setting mostly related to contrastive losses   
    return_idxs:bool = False # return df_idx and start_idx and end_idx inside the sequence as part of the data loader (e.g. for spectra)
    sample_items_per_record:int = 1 # number of samples to be returned per record (use 2 for contrastive approaches)

@dataclass
class BaseConfigData:
    _target_:str = "clinical_ts.models.ecg_foundation_models.ecg_cpc.config.BaseConfigData" # just used to filter out data configs from kwargs
    name:str = "" # dataset name (only for supervised training during preprocessing)
    path:str = "" # path to dataset
    df_filename:str = "df_memmap.pkl" # filename of dataframe
    path_label:str= "" # separate path to annotations (by default will be inferred from path)
    fs:float = 100. # input sampling frequency
    col_lbl:str = "label" # df column that contains the label
    cols_static:List[str]=field(default_factory=lambda: []) # columns to be used as static data (continuous)
    cols_static_cat:List[str]=field(default_factory=lambda: []) # columns to be used as static data (categorical)

    col_train_fold:str = "strat_fold" # column in the dataset used to select the training set
    col_val_fold:str ="strat_fold" # column in the dataset used to select the validation set
    col_test_fold:str = "strat_fold" # column in the dataset used to select the test set
    train_fold_ids:List[str]=field(default_factory=lambda: ["x-1","x-2"]) # can use positive number to select folds (also supports negative numbers counting from backwards), using a x as prefix excludes a fold that follows the x
    val_fold_ids:List[str]=field(default_factory=lambda: ["-2"])
    test_fold_ids:List[str]=field(default_factory=lambda: ["-1"])

    input_channels_filter:List[int] = field(default_factory=lambda: []) # integer array to specify which channels to select []: use all

    label_filter:List[int] = field(default_factory=lambda: []) # supervised only: filter out certain labels from loss calculation and evaluation [] to include all
    annotation:bool = False # True for sequence annotation
    fs_annotation:float = -1#sampling frequency for annotations; for PSG this should be 1./(30.*fs);-1: is set to fs by default 
    allow_multiple_keys:bool = False #allow duplicate keys in the dataset

@dataclass
class TrainerConfig:
    executable:str = ""
    revision:str = ""
    username:str = ""

    export_features:str = "" # specify layer for which features are supposed to be exported e.g. "ts_encoder.predictor"
    export_predictions:bool = False
    
    epochs:int=30 # number of total epochs to run
    frozen_epochs:int=0 # number of epochs of the above during which only the head is trained
    num_workers:int=4 # number of works in the dataloader

    resume:str='' # path to latest checkpoint (default: none)
    pretrained:str='' # path to pretrained checkpoint (default: none)
    pretrained_keys_filter:List[str]=field(default_factory=lambda: ["ts_encoder.encoder.", "ts_encoder.predictor."]) # if pretrained is set load only weights corresponding to keys that start with strings specified in this list (by default sequence encoder and predictor only)
    eval_only:str='' # path to checkpoint for evaluation
       
    output_path:str='.' # output path
    metadata:str='' # metadata for output 
    
    gpus:int=1 # number of gpus
    num_nodes:int=1 # number of compute nodes
    precision:int=16 # 16/32
    strategy:Any="auto" # None/ddp
    accumulate:int=1 # accumulate grad batches (total-bs=accumulate-batches*bs)

    #checkpointing
    checkpoint_every_n_train_steps:int=0 #Model checkpointing according to train steps (0:disabled)
    checkpoint_every_n_epochs:int=1 #Model checkpointing according to epochs (0:disabled)

    fp32_matmul_precision:str = "highest" # highest corresponds to default behavior https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        
    lr_find:bool=False # run lr finder before training run
    auto_batch_size:bool=False # run batch size finder before training run

    refresh_rate:int=0 # progress bar refresh rate (0 to disable)

############################################################################################################
@dataclass
class TaskConfig:
    """task config: add extra fields in derived classes if desired"""
    mainclass:str= "clinical_ts.template_model.SSLModel"

###########################################################################################################
@dataclass
class LossConfig:
    _target_:str = ""
    loss_type:str="" # supervised vs. cpc vs. masked_pred vs. masked_rec vs. lm

@dataclass
class SSLLossConfig(LossConfig):
    _target_:str = ""
    pretraining_targets:int=0 # 0:continuous; 1: continuous quantized; 2: discrete quantized

###########################################################################################################
@dataclass
class ShapeConfig:
    channels:int=0 # ordinary input channels
    length:int=0 # 0 means no sequence axis 
    sequence_last:bool = False # sequence axis last vs. sequence axis first
    static_dim:int = 0 # continuous static features
    channels2:int=0 # frequency channels
    static_dim_cat:int = 0 # categorical static features

    def __str__(self):
        res ="["
        if(self.channels>0 or self.channels2>0):
            if(self.sequence_last):
                res+="ch="+str(self.channels)+","+("ch2="+str(self.channels2)+"," if self.channels2>0 else "")+"seq="+str(self.length)+"]"
            else:
                res+="seq="+str(self.length)+",ch="+str(self.channels)+(",ch2="+str(self.channels2)+"," if self.channels2>0 else "")+"]"
        if(self.static_dim>0 or self.static_dim_cat>0):
            res+="["
            if(self.static_dim>0 or self.static_dim_cat>0):
                res+="cont="+str(self.static_dim)+",cat="+str(self.static_dim_cat)
            elif(self.static_dim>0 ):
                res+="cont="+str(self.static_dim)
            else:
                res+="cat="+str(self.static_dim_cat)
            res+="]" 
        return res
############################################################################################################
@dataclass
class StaticStats:
    median:float =0. # feature-wise median
    quantile25:float=0. # 25th percentile
    quantile75:float=0. # 75th percentile
    min:float=0. # minimum value
    max:float=0. # maximum value
    nans:int=0 # number of NaN values

############################################################################################################
class EncoderBase(nn.Module):
    '''Encoder base class'''
    def __init__(self, hparams_encoder, static_stats_train, hparams_input_shape):
        '''
        input shape: (bs,channels,seq) + optional static (bs,feat)
        selected encoders e.g. NoEncoder also accept (bs,channels,freq,seq) for the first argument
        output shape: bs,seq,feat
        static_stats_train: static stats (for compatibility with ts encoder)
        '''
        super().__init__()

    def get_output_shape(self):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class EncoderBaseConfig:
    _target_:str = ""
    timesteps_per_token: int = 1 # timesteps per token a la vision transformer
    input_format_seq_last: bool = True # seq axis last e.g. bs, ch, seq instead of second e.g. bs, seq, ch


class PrePostBase(nn.Module):
    '''Pre- postprocessing base class (to be applied before encoder/after head in TimeSeriesEncoder)'''
    def __init__(self, hparams_encoder, hparams_input_shape, hparams_input_shape_pre=None):
        '''
        input shape: (bs,channels,seq) + optional static (bs,feat)
        selected encoders e.g. NoEncoder also accept (bs,channels,freq,seq) for the first argument
        output shape: bs,seq,feat
        '''
        super().__init__()
    
    def get_output_shape(self):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class PrePostBaseConfig:
    _target_:str = ""

class EncoderStaticBase(nn.Module):
    '''Static encoder base class'''
    def __init__(self, hparams_encoder, hparams_base, static_stats_train, target_dim=None):
        '''
        if target_dim is not None, the output dimension should be target_dim
        input shape: bs, channels
        output shape: bs, feat
        static_stats_train: StaticStats inferred from the training set
        '''
        super().__init__()
    
    def get_output_shape(self):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class EncoderStaticBaseConfig:
    _target_:str = ""

class PredictorBase(nn.Module):
    '''Predictor base class'''
    def __init__(self, hparams_predictor, hparams_input_shape):
        '''
        input shape: bs, seq, feat
        output shape: bs, seq, feat
        '''
        super().__init__()
        self.model_dim = hparams_predictor.model_dim
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.model_dim
    
    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class PredictorBaseConfig:
    _target_:str = ""
    model_dim: int = 512 # model hidden units/internal dimension (typical 512 for RNNs, 768 for transformers)
    causal: bool = False # use unidirectional predictor

class HeadBase(nn.Module):
    '''Head base class'''
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''
        input shape: bs, seq, feat
        output shape: bs,seq,nc for multi_prediction else bs,nc for global_pool
        '''
        super().__init__()
        self.target_dim = target_dim
        self.multi_prediction = hparams_head.multi_prediction
        
    def get_output_shape(self):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())
    
@dataclass
class HeadBaseConfig:
    _target_:str = ""
    multi_prediction: bool = False # prediction for every token/set of pooled tokens

class QuantizerBase(nn.Module):
    '''Quantizer base class'''
    def __init__(self, hparams_quantizer, hparams_input_shape):
        super().__init__()
        self.output_shape = dataclasses.replace(hparams_input_shape)

        if(hparams_quantizer.target_dim>0):
            self.output_shape.channels = hparams_quantizer.target_dim
        else:
            self.output_shape.channels = hparams_quantizer.embedding_dim*hparams_quantizer.num_codebooks
        
    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class QuantizerBaseConfig:
    _target_:str = "" # clinical_ts.cpc_template.QuantizerBase
    embedding_dim: int = 128 # model hidden units/internal dimension (typical 512 for RNNs, 768 for transformers)
    vocab_size: int = 320 # number of items in the vocabulary (in each codebook)
    target_dim: int = 0 # can either be set explicitly or will just be dynamically calculated
    num_codebooks: int = 2 # number of codebooks

@dataclass
class MaskingBaseConfig:
    _target_:str = ""
    mask_probability:float = 0.065 # wav2vec presets
    mask_span: int = 10 # mask span length

class TimeSeriesEncoder(EncoderBase):
    '''A time series encoder consisting of encoder, predictor, head'''
    def __init__(self, hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=None, has_ema=True):
        super().__init__(hparams_seqenc, static_stats_train, hparams_input_shape)
        # save basic vars
        self.pass_static = hparams_seqenc.pass_static
        self.name = hparams_seqenc.name
        hparams_input_shape_original = hparams_input_shape

        #(optional) preprocessing
        self.pre = _string_to_class(hparams_seqenc.pre._target_)(hparams_seqenc.pre, hparams_input_shape) if hparams_seqenc.pre._target_!="" else None
        hparams_input_shape_pre=dataclasses.replace(hparams_input_shape)#store input shape
        if(self.pre):
            hparams_input_shape = self.pre.get_output_shape()
        
        #(optional) static encoder
        self.static_encoder = _string_to_class(hparams_seqenc.static._target_)(hparams_seqenc.static, hparams_input_shape, static_stats_train) if hparams_seqenc.static._target_!="" else None
        if(self.static_encoder):
            hparams_input_shape = self.static_encoder.get_output_shape()

        #encoder
        self.encoder = _string_to_class(hparams_seqenc.enc._target_)(hparams_seqenc.enc, hparams_input_shape, static_stats_train) if hparams_seqenc.enc._target_!="" else None
        #import pdb; pdb.set_trace()
        if(isinstance(self.encoder,TimeSeriesEncoder)):
            assert(self.encoder.name!=self.name)#names have to be different to be able to distinguish losses etc
        hparams_input_shape = self.encoder.get_output_shape()
        if(not self.pass_static):
            hparams_input_shape.static_dim = 0
            hparams_input_shape.static_dim_cat = 0
        hparams_output_shape_encoder = hparams_input_shape

        #(optional) quantizer
        self.quantizer = _string_to_class(hparams_seqenc.quant._target_)(hparams_seqenc.quant, hparams_output_shape_encoder) if hparams_seqenc.quant._target_!="" else None
        self.encoder_output_shape = self.quantizer.get_output_shape() if self.quantizer else hparams_output_shape_encoder

        #(optional) masking module
        self.masking = _string_to_class(hparams_seqenc.mask._target_)(hparams_seqenc.mask, hparams_output_shape_encoder) if hparams_seqenc.mask._target_!="" else None

        #predictor
        self.predictor = _string_to_class(hparams_seqenc.pred._target_)(hparams_seqenc.pred, hparams_input_shape) if hparams_seqenc.pred._target_!="" else None
        hparams_input_shape = self.predictor.get_output_shape()
        self.output_dim = hparams_seqenc.pred.model_dim
        hparams_output_shape_predictor = hparams_input_shape

        #(optional) head
        self.head = _string_to_class(hparams_seqenc.head._target_)(hparams_seqenc.head, hparams_output_shape_predictor, self.output_dim if target_dim is None else target_dim) if hparams_seqenc.head._target_!="" else None
        if(self.head):
            hparams_input_shape = self.head.get_output_shape()
        
        #(optional) postprocessing
        self.post = _string_to_class(hparams_seqenc.post._target_)(hparams_seqenc.post, hparams_input_shape, hparams_input_shape_pre) if hparams_seqenc.post._target_!="" else None
        
        if(self.post):
            hparams_input_shape = self.post.get_output_shape()
        
        self.output_shape = hparams_input_shape
        
        #(optional) ssl head (if also ssl loss is specified)
        ssl_target_dim = self.encoder_output_shape.channels
        if(hparams_seqenc.loss.loss_type.startswith("instance_contrastive")):
            ssl_target_dim = hparams_seqenc.loss.target_dim 
        elif(hparams_seqenc.loss.loss_type.startswith("masked_pred_hubert")):
            ssl_target_dim = self.output_dim
        elif(hparams_seqenc.loss.loss_type.startswith("masked_pred_capi")):
            ssl_target_dim = list(hparams_seqenc.loss.kmeans_ks)
        
        self.head_ssl = _string_to_class(hparams_seqenc.head_ssl._target_)(hparams_seqenc.head_ssl, hparams_output_shape_predictor, ssl_target_dim) if (hparams_seqenc.head_ssl._target_!="" and hparams_seqenc.loss._target_!="") else None
        
        #(optional) ssl loss
        self.loss = _string_to_class(hparams_seqenc.loss._target_)(hparams_seqenc.loss) if hparams_seqenc.loss._target_!="" else None
        
        #(optional) ema (only keep the components we need for the standard forward pass)
        if(has_ema and hparams_seqenc.ema._target_!=""): #avoid infinite recursion
            self.ema = _string_to_class(hparams_seqenc.ema._target_)(hparams_seqenc, hparams_input_shape_original, static_stats_train, target_dim, has_ema=False)
            self.ema.initialize_params(self)
        else:
            self.ema = None

    def update_ema_params(self):
        if(self.ema):
            self.ema.update_params(self)
            
    def forward(self, **kwargs): 
        #input shape: (bs,channels,seq) or (bs,ch,freq,seq) + optional static (bs,feat)
        #output shape: bs,seq'',feat

        #evaluate ema if required
        if(self.ema):
            kwargs_ema = kwargs.copy()
            input_predicted_ema = self.ema.forward_features(**kwargs_ema)
        else:
            input_predicted_ema = None

        if(self.loss is None): #standard supervised forward
            seq_w_static_enc_res = self.static_encoder(**kwargs) if self.static_encoder else {}
            seq_w_static_enc = kwargs.copy()
            seq_w_static_enc.update(seq_w_static_enc_res)

            if(self.pre):
                seq_w_static_enc_res = self.pre(**seq_w_static_enc)
                seq_w_static_enc.update(seq_w_static_enc_res)
            
            if("static" in kwargs.keys()):
                seq_w_static_enc["static_encoded"]= seq_w_static_enc["static"]
                if(not self.pass_static):
                    del seq_w_static_enc["static"]
            
            seq_enc_w_static_enc_res = self.encoder(**seq_w_static_enc)
            seq_enc_w_static_enc = seq_w_static_enc.copy()
            seq_enc_w_static_enc.update(seq_enc_w_static_enc_res)

            seq_pred_res = self.predictor(**seq_enc_w_static_enc)
            seq_pred = seq_enc_w_static_enc.copy()
            seq_pred.update(seq_pred_res)
            
            if(self.head):
                seq_pred_res = self.head(**seq_pred) #bs*epochs, seq', feat or bs*epochs, feat for global pooling
                seq_pred.update(seq_pred_res)
            if(self.post):
                seq_pred_res = self.post(**seq_pred)
                seq_pred.update(seq_pred_res)

            seq_pred["input_predicted"] = seq_pred["seq"]#for loss calculation

            # only return changed keys
            return {k:v for k,v in seq_pred.items() if (k not in kwargs.keys() or not kwargs[k] is v)} #bs,seq'',feat
    
        else:#forward including ssl loss
            seq_w_static_enc_res = self.static_encoder(**kwargs) if self.static_encoder else kwargs
            seq_w_static_enc = kwargs.copy()
            seq_w_static_enc.update(seq_w_static_enc_res)

            if(self.static_encoder is not None):
                seq_w_static_enc["static_encoded"]= seq_w_static_enc["static"]
            
            if(self.pre):
                seq_w_static_enc_res = self.pre(**seq_w_static_enc)
                seq_w_static_enc.update(seq_w_static_enc_res)
            
            if(not self.pass_static and "static" in kwargs.keys()):
                del seq_w_static_enc["static"]

            seq_enc_w_static_enc_res = self.encoder(**seq_w_static_enc)
            seq_enc_w_static_enc = seq_w_static_enc.copy()
            seq_enc_w_static_enc.update(seq_enc_w_static_enc_res)

            #store (original) prediction targets (note: "seq" is left intact)
            seq_enc_w_static_enc2 = seq_enc_w_static_enc.copy()
            if(self.quantizer is not None):
                quantizer_res = self.quantizer(**seq_enc_w_static_enc2)
                seq_enc_w_static_enc2["input_encoded"] = quantizer_res["seq"]
                seq_enc_w_static_enc2["loss_quantizer_"+self.name] = quantizer_res["loss_quantizer"]
            else:
                seq_enc_w_static_enc2["input_encoded"] = seq_enc_w_static_enc2["seq"].clone()

            if(self.masking is not None):
                masking_res = self.masking(**seq_enc_w_static_enc)
                seq_enc_w_static_enc2["mask_ids"] = masking_res["mask_ids"]
                seq_enc_w_static_enc2["seq"] = masking_res["seq"]

            
            input_predicted_res = self.predictor(**seq_enc_w_static_enc2)
            input_predicted = seq_enc_w_static_enc2.copy()
            input_predicted.update(input_predicted_res)

            if(input_predicted_ema is not None):
                input_predicted.update(input_predicted_ema)
                
            if(self.head_ssl is not None):
                input_predicted_final_res = self.head_ssl(**input_predicted) #bs*epochs, seq', feat or bs*epochs, feat for global pooling
                input_predicted_final = input_predicted.copy()
                input_predicted_final.update(input_predicted_final_res)
                
                if(self.post):
                    input_predicted_final_res = self.post(**input_predicted_final)
                    input_predicted_final.update(input_predicted_final_res)    
                
                input_predicted_final["input_predicted"]=input_predicted_final["seq"]
            
            loss_dict={k+"_"+self.name:v for k,v in self.loss(**input_predicted_final).items()}
            input_predicted_final.update(loss_dict)

            # another head for a potentially following layer    
            if(self.head is not None):
                if(self.masking is not None):#recalculate predictor output for clean input
                    input_predicted_clean = self.predictor(**seq_enc_w_static_enc)
                else:
                    input_predicted_clean = input_predicted
                input_predicted_clean_res = self.head(**input_predicted_clean)
                input_predicted_clean.update(input_predicted_clean_res)
                if(self.post):
                    input_predicted_clean_res = self.post(**input_predicted_clean)
                    input_predicted_clean.update(input_predicted_clean_res)
                input_predicted_clean["input_predicted"]=input_predicted_clean["seq"]
            else:
                input_predicted_clean = input_predicted #just a dummy- will not be used

            input_predicted_clean.update({k:v for k,v in input_predicted_final.items() if k not in input_predicted_clean.keys()}) #add encoder loss- if present
            return input_predicted_clean
        
    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        txt = self.__class__.__name__ + "\toutput shape:" +str(self.get_output_shape())
        txt+="\n["
        lst_submodules = [self.static_encoder,self.pre,self.encoder,self.quantizer,self.masking,self.predictor,self.head_ssl,self.head,self.post]
        lst_text = ["static_encoder","pre","encoder","quantizer","masking","predictor","head_ssl","head","post"]
        for m,t in zip(lst_submodules,lst_text):
            if(m is not None):
                txt+="\n-"+t+":"+"\t"+str(m)
        return txt+"\n]"
    
        
    
    #def get_encoder_output_shape(self):
    #    return self.encoder_output_shape
    
    def get_modules(self):
        '''returns submodules assigned to encoder, predictor and head for discriminative learning rates'''
        encoder_modules=[]
        predictor_modules=[]
        head_modules=[]

        if(self.encoder):
            encoder_modules.append(self.encoder)
        if(self.predictor):
            predictor_modules.append(self.predictor)
        if(self.head):
            head_modules.append(self.head)
        if(self.pre):
            encoder_modules.append(self.pre)
        if(self.head_ssl):
            head_modules.append(self.head_ssl)
        if(self.post):
            head_modules.append(self.post)
        if(self.loss):
            head_modules.append(self.loss)    
        if(self.static_encoder):
            encoder_modules.append(self.static_encoder)
        if(self.quantizer):
            predictor_modules.append(self.quantizer)
        if(self.masking):
            predictor_modules.append(self.masking)
        return encoder_modules, predictor_modules, head_modules
        
@dataclass
class EMATimeSeriesEncoderBaseConfig:
    '''base class for the EMA time series encoder'''
    _target_:str=""

@dataclass
class TimeSeriesEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.template_modules.TimeSeriesEncoder"
    name:str = "tsenc" #unique name appended to losses etc and other outputs

    enc: EncoderBaseConfig = field(default_factory=EncoderBaseConfig)
    pred: PredictorBaseConfig = field(default_factory=PredictorBaseConfig)
    head: HeadBaseConfig = field(default_factory=HeadBaseConfig)

    pre: PrePostBaseConfig = field(default_factory=PrePostBaseConfig)
    post: PrePostBaseConfig = field(default_factory=PrePostBaseConfig)
    head_ssl: HeadBaseConfig = field(default_factory=HeadBaseConfig)
    static: EncoderStaticBaseConfig = field(default_factory=EncoderStaticBaseConfig)
    quant: QuantizerBaseConfig = field(default_factory=QuantizerBaseConfig) #NOTE: quantizer only active to produce targets during pretraining
    mask: MaskingBaseConfig = field(default_factory=MaskingBaseConfig)
    loss: SSLLossConfig = field(default_factory=SSLLossConfig)

    ema: EMATimeSeriesEncoderBaseConfig = field(default_factory=EMATimeSeriesEncoderBaseConfig)

    pass_static: bool = True #decide if static gets passed to enc, pred, head (disable for example for CLIP)


class SequentialTimeSeriesEncoder(EncoderBase):
    '''A sequence of TimeSeriesEncoders (somewhat similar to nn.Sequential)
    Note: this is the only module that gets instantiated directly in template_model.py and not through a corresponding config
    '''
    def __init__(self, lst_hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=None):
        super().__init__(lst_hparams_seqenc, static_stats_train, hparams_input_shape)
        ts_encoder = []
        for i,hparams in enumerate(lst_hparams_seqenc):
            ts_encoder.append(_string_to_class(hparams._target_)(hparams, hparams_input_shape, static_stats_train, None if i!=len(lst_hparams_seqenc)-1 else target_dim))
            hparams_input_shape = ts_encoder[-1].get_output_shape()
        self.models = nn.ModuleList(ts_encoder)

        self.output_shape = hparams_input_shape

    def forward(self, **kwargs):
        current = kwargs.copy()
        for m in self.models:
            res = m(**current)
            current.update(res)
        return res
    
    def update_ema_params(self):
        for m in self.models:
            m.update_ema_params()
    
    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        txt = self.__class__.__name__ + "\t" +str(self.get_output_shape())
        txt+="\n["
        for i,m in enumerate(self.models):
            txt+="\n-"+str(i)+":"+str(m)
        return txt+"\n]"
    
    def get_modules(self):
        encoder_modules = []
        predictor_modules = []
        head_modules = []

        for i,m in enumerate(self.models):
            em,pm,hm = m.get_modules()
            if(i<len(self.models)-1):#everything as encoder except for the last layer
                encoder_modules=encoder_modules+em
                encoder_modules=encoder_modules+pm
                encoder_modules=encoder_modules+hm
            else:
                encoder_modules=encoder_modules+em
                predictor_modules=predictor_modules+pm
                head_modules=head_modules+hm
        return encoder_modules, predictor_modules, head_modules
