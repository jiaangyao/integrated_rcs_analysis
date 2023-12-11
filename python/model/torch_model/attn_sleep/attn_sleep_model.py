from .model_container.model import AttnSleep
from .trainer.trainer import AttnSleepTrainer

from ..base import BaseTorchModel, BaseTorchTrainer
from ..torch_utils import init_gpu

class AttnSleepModel(BaseTorchModel):
    MODEL_ARCHITECURE_KEYS = ["N",
            "d_model",
            "d_ff = 120",
            "h",
            "dropout",
            "num_classes",
            "afr_reduced_cnn_size",
    ]
    
    def __init__(self,
        N = 2,  # number of TCE clones
        d_model = 80,  # set to be 100 for SHHS dataset
        d_ff = 120,   # dimension of feed forward
        h = 5,  # number of attention heads
        dropout = 0.1,
        n_class = 5,
        afr_reduced_cnn_size = 30,
        lam=1e-5,
        n_epoch=20,
        batch_size=32,
        act_func="relu",
        str_reg="L2",
        str_loss="CrossEntropy",
        str_opt="Adam",
        lr=1e-4,
        early_stopping=None,
        bool_verbose=False,
        transform=None,
        target_transform=None,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
        
    ):
        
        self.model_kwargs, self.trainer_kwargs = self.split_kwargs_into_model_and_trainer(locals())
        self.device = init_gpu(use_gpu=bool_use_gpu)

        # initialize the model
        self.model = AttnSleep(
            **self.model_kwargs
        )
        self.model.to(self.device)

        self.early_stopping = self.get_early_stopper(early_stopping)

        self.trainer = BaseTorchTrainer(
            self.model, self.early_stopping, **self.trainer_kwargs
        )

        super().__init__(self.model, self.trainer, self.early_stopping, self.model_kwargs, self.trainer_kwargs)

    def override_model(self, kwargs: dict) -> None:
        model_kwargs, trainer_kwargs = self.split_kwargs_into_model_and_trainer(kwargs)
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.model = AttnSleep(**model_kwargs)
        self.model.to(self.device)
        if self.early_stopping is not None: self.early_stopping.reset()
        self.trainer = BaseTorchTrainer(self.model, self.early_stopping, **trainer_kwargs)
        self.model.to(self.device)
    
    def reset_model(self) -> None:
        #self.override_model(self.model_kwargs | self.trainer_kwargs)
        self.model = AttnSleep(**self.model_kwargs)
        self.model.to(self.device)
        if self.early_stopping is not None: self.early_stopping.reset()
        self.trainer = BaseTorchTrainer(self.model, self.early_stopping, **self.trainer_kwargs)
        self.model.to(self.device)