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
from clinical_ts.loss.supervised import BCELossConfig
from clinical_ts.ts.encoder import RNNEncoderConfig
from clinical_ts.ts.head import MLPHeadConfig
from clinical_ts.ts.s4 import S4PredictorConfig
from clinical_ts.task.ecg import TaskConfigECG
from clinical_ts.metric.base import MetricConfig, MetricAUROCAggConfig
from clinical_ts.loss.selfsupervised import CPCLossConfig


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

    ######################################################################
    # base
    ######################################################################
    cs.store(group="base", name="base", node=BaseConfig)

    ######################################################################
    # input data
    ######################################################################
    cs.store(group="data", name="base", node=BaseConfigData)

    ######################################################################
    # time series encoder
    ######################################################################
    cs.store(group="ts", name="tsenc",  node=TimeSeriesEncoderConfig)

    #ENCODER
    cs.store(group="ts/enc", name="rnn", node=RNNEncoderConfig)

    #PREDICTOR
    cs.store(group="ts/pred", name="s4", node=S4PredictorConfig)

    #HEADS
    cs.store(group="ts/head", name="none", node=HeadBaseConfig)

    #SSL HEADS
    cs.store(group="ts/head_ssl", name="none", node=HeadBaseConfig)
    #universal heads
    cs.store(group="ts/head_ssl", name="mlp", node=MLPHeadConfig)

    #LOSS
    cs.store(group="ts/loss", name="none", node=SSLLossConfig)
    cs.store(group="ts/loss", name="cpc", node=CPCLossConfig)

    ######################################################################
    # static encoder
    ######################################################################    
    for g in ["static", "ts/static"]:
        cs.store(group=g, name="none", node=EncoderStaticBaseConfig)
    
    ######################################################################
    # optional multimodal head
    ######################################################################
    cs.store(group="head", name="none", node=HeadBaseConfig)

    ######################################################################
    # loss function
    ######################################################################
    #no global loss
    cs.store(group="loss", name="none", node=LossConfig)
    #supervised losses
    cs.store(group="loss", name="bce", node=BCELossConfig)

    ######################################################################
    # metrics
    ######################################################################
    cs.store(group="metric", name="none", node=MetricConfig)
    cs.store(group="metric", name="aurocagg", node=MetricAUROCAggConfig)

    ######################################################################
    # trainer
    ######################################################################
    cs.store(group="trainer", name="trainer", node=TrainerConfig)

    ######################################################################
    # task
    ######################################################################
    cs.store(group="task", name="ecg", node=TaskConfigECG)
    
    return cs
