__all__ = ['MetricConfig', 'MetricBase', 'MetricAUROC', 'MetricAUROCConfig', 'MetricAUROCAggConfig','MetricAUPRConfig', 'MetricAUPRAggConfig', "MetricMAE", "MetricMAEConfig", "MetricMAEAggConfig", "MetricFbeta", "MetricFbetaConfig", "MetricFbetaAggConfig", "MetricF1Config", "MetricF1AggConfig", "MetricAccuracy", "MetricAccuracyConfig", "MetricAccuracyAggConfig","MetricSensitivitySpecificity","MetricSensitivitySpecificityConfig","MetricSensitivitySpecificityAggConfig"]

import numpy as np

from dataclasses import dataclass, field

from clinical_ts.utils.eval_utils_cafa import multiclass_roc_curve
from clinical_ts.utils.bootstrap_utils import empirical_bootstrap

from sklearn.metrics import mean_absolute_error, accuracy_score, fbeta_score, recall_score, matthews_corrcoef
from typing import List

import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Filter out the warnings due to not enough positive/negative samples during bootstrapping
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

@dataclass
class MetricConfig:
    _target_:str = ""
    
    name:str = ""#name of the metric e.g. auroc

    aggregation:str = "" #"" means no aggregation across segments of the same sequence, other options: "mean", "max"
    
    key_summary_metric:str = "" #key into the output dict that can serve as summary metric for early stopping etc e.g. (without key_prefix and key_postfix and aggregation type)
    mode_summary_metric:str ="max" #used to determine if key_summary_metric is supposed to be maximized or minimized
    
    verbose:str = "" # comma-separated list of keys to be printed after metric evaluation (without key_prefix and key_postfix and aggregation type)
    
    bootstrap_report_nans:bool = False #report nans during bootstrapping (due to not enough labels of a certain type in certain bootstrap iterations etc)
    bootstrap_iterations:int = 0 #0: no bootstrap
    bootstrap_alpha:float= 0.95 # bootstrap alpha

def _reformat_lbl_itos(k):
    #    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower() #openclip
    return k.replace(" ","_").replace("|","_").replace("(","").replace(")","")
    
class MetricBase:
    def __init__(self, hparams_metric, lbl_itos, key_prefix, key_postfix, test=True):
        self.key_prefix = (key_prefix+"_" if len(key_prefix)>0 else "")+hparams_metric.name+"_"
        self.key_postfix = key_postfix
        self.aggregation = hparams_metric.aggregation
        self.aggregation_txt = ("_agg" if hparams_metric.aggregation=="mean" else "_agg"+hparams_metric.aggregation) if hparams_metric.aggregation!="" else ""
        self.key_summary_metric = self.key_prefix+hparams_metric.key_summary_metric+self.aggregation_txt+self.key_postfix #data loader id added by default
        self.mode_summary_metric = hparams_metric.mode_summary_metric
        self.verbose = [x for x in hparams_metric.verbose.split(",") if x!=""]

        self.bootstrap_iterations = hparams_metric.bootstrap_iterations if test else 0 #disable bootstrap during training
        self.bootstrap_alpha = hparams_metric.bootstrap_alpha
        self.bootstrap_report_nans = hparams_metric.bootstrap_report_nans

        self.lbl_itos = [_reformat_lbl_itos(l) for l in lbl_itos]
        self.keys = self.get_keys(self.lbl_itos)


    def get_keys(self, lbl_itos):
        '''returns metrics keys in the order they will later be returned by _eval'''
        raise NotImplementedError
    
    def __call__(self,targs,preds):
        
        if(self.bootstrap_iterations==0):
            point = self._eval(targs,preds)
        else:
            point,low,high,nans = empirical_bootstrap((targs,preds), self._eval, n_iterations=self.bootstrap_iterations , alpha=self.bootstrap_alpha,ignore_nans=True)#score_fn_kwargs={"classes":self.lbl_itos}
        res = {self.key_prefix+k+self.aggregation_txt+self.key_postfix:v for v,k in zip(point,self.keys)}
        if(self.bootstrap_iterations>0):
            res_low = {self.key_prefix+k+self.aggregation_txt+self.key_postfix+"_low":v for v,k in zip(low,self.keys)}
            res_high = {self.key_prefix+k+self.aggregation_txt+self.key_postfix+"_high":v for v,k in zip(high,self.keys)}
            res_nans = {self.key_prefix+k+self.aggregation_txt+self.key_postfix+"_nans":v for v,k in zip(nans,self.keys)}
            res.update(res_low)
            res.update(res_high)
            if(self.bootstrap_report_nans):
                res.update(res_nans)

        if(len(self.verbose)>0):
            for k in self.verbose:
                print("\n"+self.key_prefix+k+self.aggregation_txt+self.key_postfix+":"+str(res[self.key_prefix+k+self.aggregation_txt+self.key_postfix]))
        
        return res

    def _eval(self,targs,preds):
        # should return an array of results ordered according to the entries returned by get_keys()
        raise NotImplementedError


    
class MetricAUROC(MetricBase):
    '''provides class-wise+macro+micro AUROC/AUPR scores'''
    def __init__(self, hparams_metric, lbl_itos, key_prefix="", key_postfix="0", test=True):
        super().__init__(hparams_metric, lbl_itos=lbl_itos, key_prefix=key_prefix, key_postfix=key_postfix, test=test)
        self.precision_recall = hparams_metric.precision_recall

    def get_keys(self, lbl_itos):
        return list(lbl_itos)+["micro","macro"]
    
    def _eval(self,targs,preds):
        if(self.precision_recall):
            _,_,res = multiclass_roc_curve(targs,preds,classes=self.lbl_itos,precision_recall=True)
            return np.array(list(res.values()))
        else:
            _,_,res = multiclass_roc_curve(targs,preds,classes=self.lbl_itos)
            return np.array(list(res.values()))
        

@dataclass
class MetricAUROCConfig(MetricConfig):
    _target_:str = "clinical_ts.metric.base.MetricAUROC"
    key_summary_metric:str = "macro"
    verbose:str="macro" #by default print out macro auc
    precision_recall:bool = False #calculate the area under the precision recall curve instead of the ROC curve
    name:str = "auroc"
    bootstrap_report_nans:bool = True #by default report number of bootstrap iterations where the score was nan (due to insufficient number of labels etc)

#shorthand for mean aggregation
@dataclass
class MetricAUROCAggConfig(MetricAUROCConfig):
    aggregation:str="mean"

#shorthand for AUPR
@dataclass
class MetricAUPRConfig(MetricAUROCConfig):
    name:str = "aupr"
    precision_recall:bool = True

#shorthand for mean aggregation
@dataclass
class MetricAUPRAggConfig(MetricAUPRConfig):
    aggregation="mean"

class MetricMAE(MetricBase):
    '''provides MAE score with masking token (nan)'''
    def __init__(self, hparams_metric, lbl_itos, key_prefix="", key_postfix="0", test=True):
        self.return_nan_fraction = hparams_metric.return_nan_fraction
        super().__init__(hparams_metric, lbl_itos=lbl_itos, key_prefix=key_prefix, key_postfix=key_postfix, test=test)

    def get_keys(self, lbl_itos):
        keys = list(lbl_itos)+["sum"]
        if(self.return_nan_fraction):
            keys = keys+[l+"_nan_fraction" for l in lbl_itos]
        return keys
        
    def _eval(self,targs,preds):
        if(len(targs.shape)==1):#in case of a single output: add a dummy unit axis
            targs=np.expand_dims(targs)
            preds=np.expand_dims(preds)
        if(preds.shape[1]!=len(self.lbl_itos)):#quantile prediction
            preds = preds.reshape(preds.shape[0],len(self.lbl_itos),-1)[:,:,0]#by default the first entry is the 0.5 quantile- use that for evaluation
        res=[]
        res_nans=[]
        res_sum=0
        if(np.sum(np.isnan(targs))==0):
            res = mean_absolute_error(targs,preds,multioutput="raw_values") # label MAEs
            res_sum = mean_absolute_error(targs,preds,multioutput="uniform_average") #sum
            if(self.return_nan_fraction):
                res_nans = np.zeros_like(res)
                return np.concatenate((res,[res_sum],res_nans),axis=0)
            else:
                return np.concatenate((res,[res_sum]),axis=0)
        else:#nan targets
            for i,l in enumerate(self.lbl_itos):
                targsi = targs[:,i]
                predsi = preds[:,i]
                mask = ~np.isnan(targsi)
                if(len(targsi[mask])>0):
                    res.append(mean_absolute_error(targsi[mask],predsi[mask]))
                    res_sum=res_sum+res[-1]
                else:
                    res.append(np.nan)
                if(self.return_nan_fraction):
                    res_nans.append(1-np.sum(mask)/len(targs))
        res.append(res_sum)
        if(self.return_nan_fraction):
            res = res + res_nans
        return np.array(res)
    
@dataclass
class MetricMAEConfig(MetricConfig):
    _target_:str = "clinical_ts.metric.base.MetricMAE"
    name:str = "mae"
    key_summary_metric:str = "sum"
    mode_summary_metric:str = "min" #minimize mae in case of checkpointing
    return_nan_fraction:bool = False #return fraction of nans per label

#shorthands for mean aggregation
@dataclass
class MetricMAEAggConfig(MetricMAEConfig):
    aggregation:str="mean"

class MetricFbeta(MetricBase):
    '''provides Fbeta score (based on argmax i.e. threshold 0.5, suitable for multi-class)'''
    def __init__(self, hparams_metric, lbl_itos, key_prefix="", key_postfix="0", test=True):
        super().__init__(hparams_metric, lbl_itos=lbl_itos, key_prefix=key_prefix, key_postfix=key_postfix, test=test)
        self.beta = hparams_metric.beta
    
    def get_keys(self, lbl_itos):
        return list(lbl_itos)+["macro"]
        
    def _eval(self,targs,preds):
        targs_argmax = np.argmax(targs,axis=-1)
        preds_argmax = np.argmax(preds,axis=-1)

        return np.concatenate((fbeta_score(targs_argmax,preds_argmax,beta=self.beta,average=None),[fbeta_score(targs_argmax,preds_argmax,beta=self.beta,average="macro")]))
        
@dataclass
class MetricFbetaConfig(MetricConfig):
    _target_:str = "clinical_ts.metric.base.MetricFbeta"
    name:str = "fbeta"
    key_summary_metric:str = "macro"
    beta:float = 1.

#shorthands for mean aggregation
@dataclass
class MetricFbetaAggConfig(MetricFbetaConfig):
    aggregation:str="mean"

#shorthands for f1 (which show up with the name f1)
@dataclass
class MetricF1Config(MetricFbetaConfig):
    name:str = "f1"
    beta:float = 1.

@dataclass
class MetricF1AggConfig(MetricF1Config):
    aggregation:str="mean"

class MetricAccuracy(MetricBase):
    '''provides accuracy score (based on argmax i.e. threshold 0.5, suitable for multi-class)'''
    def __init__(self, hparams_metric, lbl_itos, key_prefix="", key_postfix="0", test=True):
        super().__init__(hparams_metric, lbl_itos=lbl_itos, key_prefix=key_prefix, key_postfix=key_postfix, test=test)

    def get_keys(self, lbl_itos):
        return ["all"]
      
    def _eval(self,targs,preds):
        targs_argmax = np.argmax(targs,axis=-1)
        preds_argmax = np.argmax(preds,axis=-1)
        return np.array([accuracy_score(targs_argmax,preds_argmax)])
    
@dataclass
class MetricAccuracyConfig(MetricConfig):
    _target_:str = "clinical_ts.metric.base.MetricAccuracy"
    name:str="accuracy"
    key_summary_metric:str = "all"

#shorthand for mean aggregation
@dataclass
class MetricAccuracyAggConfig(MetricAccuracyConfig):
    aggregation:str="mean"

################################################################
def specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)

def fix_threshold(targs, preds, metric, targets, greater_than=True):
    #preds: N, classes
    #labels: N (binary) or N,classes (multilabel)
    if(len(targs.shape)==1 and len(preds.shape)==2):
        preds = preds[:,0][None]
        targs = targs[None]
    thresholds_res = []
    for i in range(preds.shape[1]):
        thresholds = np.unique(preds[:,i])
        #assuming a monotonic behavior
        if(metric(targs[:,i],preds[:,i]>thresholds[0])>metric(targs[:,i],preds[:,i]>thresholds[-1])):
            if(greater_than):
                thresholds = thresholds[::-1]
        else:
            if(not(greater_than)):
                thresholds = thresholds[::-1]
        for t in thresholds:
            preds_binary = preds[:,i]> t
            res = metric(targs[:,i], preds_binary)
            if((greater_than and res > targets[i]) or not(greater_than) and res < targets[i]):
                thresholds_res.append(t)
                break
        if(len(thresholds_res)<i+1):#no threshold found
            thresholds_res.append(-1.)
    return thresholds_res

def eval_sensitivity_specificity(targs, preds, metric, targets, greater_than=True):
    if(len(targs.shape)==1 and len(preds.shape)==2):#binary case
        preds = preds[:,1][None]
        targs = targs[None]

    assert(len(targets)==preds.shape[1])
    if(metric is None): #threshold=target
        thresholds=np.array(targets)
    else:
        thresholds=np.array(fix_threshold(targs, preds, metric, targets, greater_than=greater_than))

    sensitivities = []
    specificities = []
    mccs = []
    for i in range(preds.shape[1]):
        preds_binary = preds[:,i]> thresholds[i]
        sensitivities.append(sensitivity(targs[:,i], preds_binary))
        specificities.append(specificity(targs[:,i], preds_binary))
        mccs.append(matthews_corrcoef(targs[:,i],preds_binary))
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    mccs = np.array(mccs)
    youdens = sensitivities+specificities-1
    youdens = np.concatenate((youdens,np.array([np.mean(youdens)])))#macro average as last entry
    return np.concatenate((sensitivities, specificities, mccs, thresholds, youdens))


class MetricSensitivitySpecificity(MetricBase):
    '''provides sensitivity/specificity with/without threshold optimization (binary/multilabel)'''
    def __init__(self, hparams_metric, lbl_itos, key_prefix="", key_postfix="0", test=True):
        super().__init__(hparams_metric, lbl_itos=lbl_itos, key_prefix=key_prefix, key_postfix=key_postfix, test=test)
        if(not hparams_metric.optimize_thresholds):
            self.metric = None
        elif(hparams_metric.optimization_target=="sensitivity"):
            self.metric = sensitivity
        elif(hparams_metric.optimization_target=="specificity"):
            self.metric = specificity
        self.metric_txt = ("_"+hparams_metric.metric+("_gt_" if hparams_metric.greater_than else "_lt_")) if hparams_metric.optimize_thresholds else "threshold_"
        self.targets = list(hparams_metric.targets)
        self.greater_than = hparams_metric.greater_than
    
    def get_keys(self, lbl_itos):
        res = ["sensitivity"+self.metric_txt+str(np.round(t,2))+"_"+l for l,t in zip(lbl_itos,self.targets)]
        res = res + ["specificity"+self.metric_txt+str(np.round(t,2))+"_"+l for l,t in zip(lbl_itos,self.targets)]
        res = res + ["mcc"+self.metric_txt+str(np.round(t,2))+"_"+l for l,t in zip(lbl_itos,self.targets)]
        res = res + ["thresholds"+self.metric_txt+str(np.round(t,2))+"_"+l for l,t in zip(lbl_itos,self.targets)]
        res = res + ["youden"+self.metric_txt+str(np.round(t,2))+"_"+l for l,t in zip(lbl_itos,self.targets)]
        return res + ["youden_macro"]
        
    def _eval(self,targs,preds):
        num_classes = 1 if len(targs.shape)==1 else targs.shape[1]
        if(len(self.targets)==1):#just a single target specified (assume all of them are the same)
            self.targets=self.targets*(num_classes)

        return eval_sensitivity_specificity(targs, preds, self.metric, self.targets, greater_than=self.greater_than)
        
@dataclass
class MetricSensitivitySpecificityConfig(MetricConfig):
    _target_:str = "clinical_ts.metric.base.MetricSensitivitySpecificity"
    name:str = "sens_spec"
    key_summary_metric:str = "youden_macro"
    optimize_thresholds:bool = True
    greater_than:bool = True #optimize for "optimization_target" "greater_than" "targets"
    optimization_target:str = "sensitivity" #"sensitivity" or "specificity"
    targets:List[float] = field(default_factory=lambda: [0.9])#or thresholds in the case of optimize_threshold=False (in case a single element is passed this is used for all elements)

#shorthand for mean aggregation
@dataclass
class MetricSensitivitySpecificityAggConfig(MetricSensitivitySpecificityConfig):
    aggregation:str="mean"