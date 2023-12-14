import optuna
import wandb
from training_eval.process_classification_results import process_and_log_eval_results_sklearn, process_and_log_eval_results_torch
import glob
import shutil

# TODO: Implement hyperparameter optimization via Optuna, Ray Tune


class HyperparameterOptimization:
    def __init__(self, model_class, data, eval):
        self.model_class = model_class
        self.data = data  # Should be an instance of dataset.data_class.MLData
        self.evaluation = eval  # Should be an instance of training_eval.model_evaluation.ModelEvaluation

    def initialize_wandb_params(self, output_dir, wandb_group, wandb_tags, wandb_notes):
        self.output_dir = output_dir # Typically the local directory
        self.wandb_group = wandb_group
        self.wandb_tags = wandb_tags
        self.wandb_notes = wandb_notes
    
    def train_and_eval_no_search(self, config):
        # Train model and Evaluate predictions
        results, epoch_metrics = self.evaluation.evaluate_model(self.model_class, self.data)

        # Log results (to WandB)
        # TODO: Create a logging class to handle this, so that users can pick different logging options/dashboards outside of W&B
        if self.evaluation.model_type == "torch":
            process_and_log_eval_results_torch(
                results, config['setup']['path_run'], epoch_metrics
            )
        elif self.evaluation.model_type == "sklearn":
            process_and_log_eval_results_sklearn(results, config['setup']['path_run'])
            

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

        # Initialize a new wandb run
        with wandb.init(
            config=config,
            dir=self.output_dir,
            group=self.wandb_group,
            tags=self.wandb_tags,
            notes=self.wandb_notes,
        ):  
            # Log the local directory path to wandb, and save log file to run directory (so that it is visible in each run's files dashboard)
            wandb.log({'metadata/local_dir': wandb.run.dir})
            if log_file := glob.glob(f'{self.output_dir}/*.log'): shutil.copy(log_file[0], wandb.run.dir)

            
            # If called by wandb.agent this config will be set by Sweep Controller
            config = wandb.config

            self.model_class.override_model(config.as_dict())

            # Evaluate predictions
            results, epoch_metrics = self.evaluation.evaluate_model(
                self.model_class, self.data
            )
            
            if self.evaluation.model_type == 'torch':
                process_and_log_eval_results_torch(results, self.output_dir, epoch_metrics)
            elif self.evaluation.model_type == 'sklearn':
                process_and_log_eval_results_sklearn(results, self.output_dir)

