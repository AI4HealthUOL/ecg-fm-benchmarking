import os
import shutil

from matplotlib import pyplot as plt
from pathlib import Path

import torch
import lightning.pytorch as lp
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from clinical_ts.utils.callbacks import LRMonitorCallback, TriggerQuantizerHyperparameterUpdate, UnfreezingFinetuningCallback
from clinical_ts.template_modules import QuantizerBase

import hydra
from hydra.core.hydra_config import HydraConfig

#################
#specific
from clinical_ts.config import *
from clinical_ts.utils.basic_io import get_git_revision_short_hash, get_slurm_job_id, get_work_dir, _string_to_class
from omegaconf import OmegaConf


MLFLOW_AVAILABLE=True
try:
    import mlflow
    from lightning.pytorch.loggers import MLFlowLogger
    from omegaconf import DictConfig, ListConfig

    def log_params_from_omegaconf_dict(params):
        for param_name, element in params.items():
            _explore_recursive(param_name, element)

    def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    _explore_recursive(f'{parent_name}.{k}', v)
                else:
                    if(k!="_target_" and v is not None):
                        mlflow.log_param(f'{parent_name}.{k}'," " if v=="" else v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f'{parent_name}.{i}', " " if v=="" else v)
    from mlflow.exceptions import MlflowException

    class RobustMLFlowLogger(MLFlowLogger):
        '''Robust MLFLowLogger that does not crash if connection is down'''
        def finalize(self, status: str) -> None:
            try:
                super().finalize(status)
            except (MlflowException, ConnectionError, TimeoutError) as e:
                print(f"MLflow finalization failed: {e}")
                print("Training completed successfully despite logging issues")
            except Exception as e:
                print(f"Unexpected error during MLflow finalization: {e}")
    
except ImportError:
    MLFLOW_AVAILABLE=False

###################################################################################################
#MAIN
###################################################################################################
cs = create_default_config()

@hydra.main(version_base=None, config_path="conf",  config_name="config_supervised_ecg")
def run(hparams: FullConfig) -> None:
    hparams.trainer.executable = "main_all"
    hparams.trainer.revision = get_git_revision_short_hash()

    if not os.path.exists(hparams.trainer.output_path):
        os.makedirs(hparams.trainer.output_path)
    
    #determine version/output path
    slurm_job_id = get_slurm_job_id()
    run_name = get_work_dir(hparams.trainer.output_path,pattern=slurm_job_id+"_")

    hparams.trainer.output_path = Path(hparams.trainer.output_path)/(run_name)
    if not os.path.exists(hparams.trainer.output_path):
        os.makedirs(hparams.trainer.output_path)

    logger = [TensorBoardLogger(
        save_dir=Path(hparams.trainer.output_path).parent,
        version=run_name,#hparams.trainer.metadata.split(":")[0],
        name="")]
    
    print("FULL PARSED CONFIG:")
    print(OmegaConf.to_yaml(hparams))

    #get hydra configs
    hydra_cfg = HydraConfig.get()
    config_file = Path(hydra_cfg.runtime.config_sources[1]["path"])/hydra_cfg.job.config_name
    print("Output directory:",hparams.trainer.output_path)
    print("Main config:",config_file)
    print("Overrides:",OmegaConf.to_container(hydra_cfg.overrides.hydra))
    print("Runtime choices:",OmegaConf.to_container(hydra_cfg.runtime.choices))
    #print("Full config:",OmegaConf.to_yaml(hparams))
    
    #copy main config into output dir
    shutil.copyfile(config_file, Path(hparams.trainer.output_path)/(config_file.stem))

    #create the model
    classname = _string_to_class(hparams.task.mainclass)
    model = classname(hparams)
    #model = torch.compile(model)

    print(f"Run name: {run_name}")

    if(MLFLOW_AVAILABLE):
        mlflow.set_experiment(hparams.trainer.executable+"("+hparams.task.mainclass.split(".")[-1]+")")
        
        mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.set_tag("mlflow.runName", run_name)

        mlf_logger = RobustMLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow_run.info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=False,
        )
        mlf_logger._run_id = mlflow_run.info.run_id
        mlf_logger.log_hyperparams = log_params_from_omegaconf_dict       
        logger.append(mlf_logger)

    key_summary_metric = model.metrics_train_val[0].key_summary_metric if len(model.metrics_train_val)>0 else 'val_loss_total0'#use the key_summary_metric of the first metric otherwise val_loss0
    mode_summary_metric = model.metrics_train_val[0].mode_summary_metric if len(model.metrics_train_val)>0 else 'min'

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger[0].log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor=key_summary_metric,
        mode=mode_summary_metric,
        every_n_train_steps = hparams.trainer.checkpoint_every_n_train_steps if hparams.trainer.checkpoint_every_n_train_steps>0 else None,
        every_n_epochs = hparams.trainer.checkpoint_every_n_epochs if hparams.trainer.checkpoint_every_n_epochs>0 else None,)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")
    
    callbacks = [checkpoint_callback,lr_monitor,lr_monitor2]
    if(hparams.trainer.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.trainer.refresh_rate))
    quantizers = [m for m in model.modules() if isinstance(m,QuantizerBase)]
    if(len(quantizers)>0):
        print("Found",len(quantizers),"quantizer modules.")
        callbacks.append(TriggerQuantizerHyperparameterUpdate(quantizers))
    if(hparams.loss.loss_type=="supervised" and hparams.trainer.frozen_epochs>0):
        callbacks.append(UnfreezingFinetuningCallback(unfreeze_epoch=hparams.trainer.frozen_epochs))

    trainer = lp.Trainer(
        #overfit_batches=0.01,
        accumulate_grad_batches=hparams.trainer.accumulate,
        max_epochs=hparams.trainer.epochs if hparams.trainer.eval_only=="" else 0,
    
        default_root_dir=hparams.trainer.output_path,
        
        #debugging flags for val and train
        num_sanity_val_steps=0,
        #overfit_batches=10,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=True,
    
        accelerator="gpu" if hparams.trainer.gpus>0 else "cpu",
        devices=hparams.trainer.gpus if hparams.trainer.gpus>0 else 1,
        num_nodes=hparams.trainer.num_nodes,
        precision=hparams.trainer.precision,
        strategy=hparams.trainer.strategy,
        
        enable_progress_bar=hparams.trainer.refresh_rate>0,
        #weights_summary='top',
        )
    
    if(hparams.trainer.fp32_matmul_precision!="highest"):
        torch.set_float32_matmul_precision(hparams.trainer.fp32_matmul_precision)
        
    if(hparams.trainer.auto_batch_size):#batch size
        tuner=Tuner(trainer)
        tuner.scale_batch_size(model, mode="binsearch")

    if(hparams.trainer.lr_find):# lr find
        tuner=Tuner(trainer)

        #torch.save(model.state_dict(), Path(hparams.trainer.output_path)/(logger.log_dir+"initial_weights.ckpt"))
        # Run learning rate finder
        lr_finder = tuner.lr_find(model)

        # Plot lr find plot
        fig = lr_finder.plot(suggest=True)
        fig.show()
        plt.savefig(Path(hparams.trainer.output_path)/("lrfind.png"))

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("Suggested lr:",new_lr)
        # update hparams of the model
        model.hparams.base.lr = [new_lr]
        model.lr = new_lr

        # there is still some issue with the restored model- therefore just abort the run
        #model.load_state_dict(torch.load(Path(hparams.trainer.output_path)/(logger.log_dir+"initial_weights.ckpt")))
        return

    if(MLFLOW_AVAILABLE):
        #version_number = hparams.trainer.output_path.split('/')[-1].replace('version_', '') # tw: extract version index
        mlflow.log_param("a_slurm_job_id", slurm_job_id)
        mlflow.log_param("a_run_name", run_name)
        mlflow.log_param("a_git_hash", get_git_revision_short_hash())
    
    if(hparams.trainer.epochs>0 and hparams.trainer.eval_only==""):
        trainer.fit(model,ckpt_path= None if hparams.trainer.resume=="" else hparams.trainer.resume)
    trainer.test(model,ckpt_path="best" if hparams.trainer.eval_only=="" else hparams.trainer.eval_only)

    if(MLFLOW_AVAILABLE):
        mlflow.end_run()
    
    if(hparams.trainer.export_features!=""):
        model.export_features(Path(hparams.trainer.output_path)/"features",module=hparams.trainer.export_features)

if __name__ == "__main__": 
    run()
