from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

#for base configs
from clinical_ts.template_modules import (
    BaseConfig,
    BaseConfigData,
    LossConfig,
    SSLLossConfig,
    TrainerConfig,
    TaskConfig,
    TimeSeriesEncoderConfig,
    EncoderStaticBaseConfig,
    HeadBaseConfig
) 
from clinical_ts.ts.encoder import RNNEncoderConfig
from clinical_ts.loss.supervised import BCELossConfig
from clinical_ts.ts.s4 import S4PredictorConfig
from clinical_ts.task.ecg import TaskConfigECG
from clinical_ts.metric.base import MetricConfig, MetricAUROCAggConfig

@dataclass
class FullConfig:

    base: BaseConfig
    data: BaseConfigData
    loss: LossConfig
    metric: MetricConfig
    trainer: TrainerConfig
    task: TaskConfig

    ts: TimeSeriesEncoderConfig
    static: EncoderStaticBaseConfig
    head: HeadBaseConfig
    

def create_default_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=FullConfig)

    cs.store(group="base", name="base", node=BaseConfig)
    cs.store(group="data", name="base", node=BaseConfigData)
    cs.store(group="ts", name="tsenc",  node=TimeSeriesEncoderConfig)
    cs.store(group="ts/enc", name="rnn", node=RNNEncoderConfig)
    cs.store(group="ts/pred", name="s4", node=S4PredictorConfig)
    cs.store(group="ts/head", name="none", node=HeadBaseConfig)
    cs.store(group="ts/head_ssl", name="none", node=HeadBaseConfig)
    cs.store(group="ts/loss", name="none", node=SSLLossConfig)    
    for g in ["static", "ts/static"]:
        cs.store(group=g, name="none", node=EncoderStaticBaseConfig)
    cs.store(group="head", name="none", node=HeadBaseConfig)
    cs.store(group="loss", name="bce", node=BCELossConfig)
    cs.store(group="metric", name="aurocagg", node=MetricAUROCAggConfig)
    cs.store(group="trainer", name="trainer", node=TrainerConfig)
    cs.store(group="task", name="ecg", node=TaskConfigECG)
    
    return cs
