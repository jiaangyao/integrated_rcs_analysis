import optuna
import wandb
from ray import tune
import numpy as np
from loguru import logger
from analysis_CS.process_classification_results import process_and_log_eval_results_sklearn, process_and_log_eval_results_torch


# TODO: Implement hyperparameter optimization via Optuna, WandB, and/or Ray Tune


class HyperparameterOptimization:
    def __init__(self, model_class, data, eval):
        self.model_class = model_class
        self.data = data  # Should be an instance of dataset.data_class.MLData
        self.evaluation = eval  # Should be an instance of training_eval.model_evaluation.ModelEvaluation

    def initialize_wandb_params(self, output_dir, wandb_group, wandb_tags):
        self.output_dir = output_dir # Typically the local directory
        self.wandb_group = wandb_group
        self.wandb_tags = wandb_tags

    def initialize_ray(self):
        raise NotImplementedError

    def initialize_optuna(self):
        raise NotImplementedError

    def ray_tune():
        raise NotImplementedError

    def optuna(self, config):
        raise NotImplementedError
        # TODO: This is incomplete, acts as placeholder for now
        study = optuna.create_study(direction="maximize")
        objective = (self.optuna_objective(), config)
        study.optimize(objective, n_trials=100)

    def optuna_objective(self, config):
        pass

    def wandb_sweep(self, config=None):
        #X_train, y_train = self.data.get_training_data()
        # Initialize a new wandb run
        with wandb.init(
            config=config,
            dir=self.output_dir,
            group=self.wandb_group,
            tags=self.wandb_tags,
        ):
            # If called by wandb.agent this config will be set by Sweep Controller
            config = wandb.config
            wandb.log({'metadata/local_dir': self.output_dir})

            self.model_class.override_model(config.as_dict())

            # Evaluate predictions
            results, epoch_losses, epoch_val_losses = self.evaluation.evaluate_model(
                self.model_class, self.data
            )
            
            if self.evaluation.model_type == 'torch':
                if epoch_val_losses: raise NotImplementedError("Epoch Validation processing and logging not yet implemented for torch models")
                process_and_log_eval_results_torch(results, config["run_dir"], epoch_losses, epoch_val_losses)
            elif self.evaluation.model_type == 'sklearn':
                process_and_log_eval_results_sklearn(results, config["run_dir"])

            # # Drop prefixes for logging
            # mean_results = {
            #     (f'{k.split("_", 1)[-1]}_mean' if 'test_' in k else f'{k}_mean'): np.mean(v) 
            #     for k, v in results.items()
            # }
            # std_results = {
            #     (f'{k.split("_", 1)[-1]}_std' if 'test_' in k else f'{k}_std'): np.std(v) 
            #     for k, v in results.items()
            # }

            # # Log model performance metrics to W&B
            # wandb.log(std_results)
            # wandb.log(mean_results)

            # TODO: Implement if test_model is true, then log results on hold-out test set
            # if self.test_model:
            #     self.model.fit(X_train, y_train)
            #     X_test, y_test = self.fetch_test_data()
            #     y_preds = self.model.predict(X_test)
