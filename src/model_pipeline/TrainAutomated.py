import matplotlib
matplotlib.use('Agg')


import pathlib as pth
import numpy as np
from pprint import pprint
from typing import Union, Sequence
import itertools
import sys
import argparse
import logging
import datetime
import multiprocessing


import torch
import torch.nn as nn
from torchinfo import summary
import optuna

from tqdm import tqdm


src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from _train_single_case import train_model
from utils import load_json, save2json, save_model, convert_str_values
from utils import Plotter

from model import YOLO


def check_models(model_configs_paths: list[pth.Path],
                 max_input_size = (1, 3, 640, 640),
                 max_memory_GB = 20,
                 verbose: bool = False) -> tuple[list[dict], list[pth.Path]]:
    """
    Check if models defined in NeuralNet/Architectures/models_1D/model_configs compile.
    Print models not compiling.
    Return list of model configs that compile.
    """

    # str paths to pth.Path if neccessary
    model_configs_paths = [pth.Path(config) for config in model_configs_paths]

    # check each model if it compiles and take not more than max memory
    model_configs = []
    for index, model_config_path in enumerate(model_configs_paths.copy()):
        model_config = load_json(model_config_path)
        model_config = convert_str_values(model_config)

        try:
            model = YOLO(model_config, 10)
            model.eval()
            model_summary = summary(model, input_size=max_input_size, verbose=0)
            estimated_memory_GB = (model_summary.total_param_bytes + model_summary.total_output_bytes) / (1024 ** 3 )

            if estimated_memory_GB > max_memory_GB:
                    raise MemoryError(f"Estimated memory {estimated_memory_GB:.2f} GB exceeds limit of {max_memory_GB:.2f} GB.")

            del model, model_summary

            if verbose:
                print(f"Model {model_config_path.name} compiled successfully\nEstimated memory: {estimated_memory_GB:.2f} GB.\n")

        except Exception as e:
            if verbose:
                print(f"Error compiling model {model_config_path.name}:\n{e}")
            model_configs_paths.pop(index)
        else:
            model_configs.append(model_config)  
    
    return model_configs, model_configs_paths

def get_step_list(param_value_list: list[Union[int, float]]) -> list[Union[int, float]]:
    """"Generate a list of values based on the given parameter value and type of list elements."""

    start, stop, step = param_value_list
    
    if all(isinstance(x, int) for x in [start, stop, step]):
        return list(range(int(start), int(stop + step), int(step)))
    elif all(isinstance(x, (int, float)) for x in [start, stop, step]):
        return [float(x) for x in np.arange(float(start), float(stop + step), float(step))]
    else:
        raise ValueError(f"Invalid parameter values: {param_value_list}. Must be all int or all float.")

def get_factor_list(param_value_list: list[Union[float]]) -> list[Union[float]]:
    """"Generate a list of values based on the given parameter value of list elements."""

    start, stop, factor = param_value_list
    factor_list = []
    
    num = start
    while num > stop:

        factor_list.append(num)
        num *= factor

    factor_list.sort()
    return factor_list

    

def generate_experiment_configs(training_config: dict,
                                device_name: str = 'cpu') -> list[dict]:
    logger = logging.getLogger(__name__)
    logger.info(f'START: generate_experiment_config.')
    
    device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')
    logger.info(f'Using device: {device}')

    dynamic_params = {}
    static_params = {}

    # Separate dynamic and model, base_path=base_path, existing_ok=Truestatic parameters

    for key, value in training_config.items():
        
        if "comment" in key.lower():
            continue
        
        elif isinstance(value, list) and len(value) > 1 and not 'samples_len' in key.lower():
            if "learning_rate" in key.lower() or "weight_decay" in key.lower():
                dynamic_params[key] = get_factor_list(training_config[key])
            else:
                dynamic_params[key] = get_step_list(value)

        elif 'samples_len' in key.lower():
            samples_len = get_step_list(value)
            static_params[key] = samples_len
        else:
            static_params[key] = value

    static_params['device'] = device
    
    logger.info(f'Generated {len(dynamic_params)} dynamic parameters.')
    logger.info(f'Generated {len(static_params)} static parameters.')

    # Generate all combinations of dynamic parameters
    keys = dynamic_params.keys()
    values_lists = dynamic_params.values()
    combinations = list(itertools.product(*values_lists))

    
    # Create experiment configurations
    exp_configs = []
    for combo in combinations:
        combo_dict = dict(zip(keys, combo))
        combo_dict.update(static_params)

            
        # get model config
        dynamic_config = dict(zip(keys, combo))
        
        exp_config = static_params.copy()
        exp_config.update(dynamic_config)
        
        exp_configs.append(exp_config)
        
    logger.info(f'Generated {len(exp_configs)} experiment configurations.')
    logger.info(f'STOP: generate_experiment_config')


    return exp_configs



def load_config(base_dir: Union[str, pth.Path], device_name: str, mode: int = 0) -> list[dict]:

    """
    Load configuration files and prepare experiment configurations for training.
    mode:
    0 - single_training
    1 - multiple trainings, grid_based
    2 - multiple trainings, with optuna
    
    """
    logger = logging.getLogger(__name__)
    logger.info(f'START: case_based_training.')

    if isinstance(mode, int):
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid mode: {mode}. Must be:\n" \
                             "0 - test" \
                             "1 - single_training," \
                             "2 - multiple trainings, grid_based," \
                             "3 - multiple trainings, with optuna.")

    base_dir = pth.Path(base_dir)
    config_files_dir = base_dir.joinpath('training_configs')

    if mode == 0 or mode == 1:
        training_config = load_json(config_files_dir.joinpath('config_train_single.json'))
    elif mode == 2 or mode == 3:
        training_config = load_json(config_files_dir.joinpath('config_train.json'))
    
    logger.info(f'Loaded training config for mode: {mode}.')
    
    training_config = convert_str_values(training_config)

    if mode == 3:
        device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')
        training_config['device'] = device
        
        logger.info(f'Loaded device: {device}')
        logger.info(f'STOP: load_config. All files loaded.')

        training_config['model'] = None

        return training_config
    else:
        exp_configs = generate_experiment_configs(training_config, device_name = device_name)
        
        logger.info(f'STOP: load_config. All files loaded.')

        return exp_configs

def test_case(exp_config: dict) -> None:
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: test_case.')
    
    """Test case for training model with reduced parameters for quick execution."""
    exp_config['train_repeat'] = 2
    exp_config['learning_rate'] = 0.01
    exp_config['epochs'] = 2
    
    try:
        for _, _ in train_model(training_dict=exp_config):
            pass
    except Exception as e:
        logger.error(f'ERROR: test_case. Error message: {e}')
        print(f"Error training model (TESTING_MODE):\n{e}")
    logger.info('STOP: test_case passed.')

class Checkpoint: 
    def __init__(self, existing_ok: bool = False) -> None:
        self.existing_ok = existing_ok
        self.final_val_best = 0.
        self.save_new = False
    
    def check_checkpoint(self, 
                         model: nn.Module,
                         model_name: str,
                         final_val: float,
                         exp_config: dict,
                         result_hist: dict) -> tuple[nn.Module, dict, pth.Path]:
    
        

        logger = logging.getLogger(__name__)

        model_dir =  pth.Path(__file__).parent.joinpath('training_results').joinpath(f'{model_name.rsplit('_', 1)[0]}')
        model_dir.mkdir(exist_ok=True, parents=True)

        model_path = model_dir.joinpath(f'{model_name}.pt')

        plot_dir = model_dir.joinpath('plots')
        if not plot_dir.exists():
            logger.info(f'Plots directory created: {plot_dir}')

        dict_files_dir = model_dir.joinpath('dict_files')
        if not dict_files_dir.exists():
            logger.info(f'Dict files directory created: {dict_files_dir}')
        
        plot_dir.mkdir(exist_ok=True, parents=True)
        if not plot_dir.exists():
            logger.info(f'Plots directory created: {plot_dir}')

        dict_files_dir.mkdir(exist_ok=True, parents=True)

        config_path = dict_files_dir.joinpath(f'{model_path.stem}_config.json')


        if final_val > self.final_val_best:
            self.save_new = True
            self.final_val_best = final_val

        
        if not self.save_new:
            return model, exp_config, config_path, result_hist

        if not self.existing_ok:
                for file_path in plot_dir.iterdir():
                    file_path.unlink()
                for file_path in dict_files_dir.iterdir():
                    file_path.unlink()

        model_path = save_model(model_path, model, 
                                        existing_ok=self.existing_ok)
        model_name = model_path.stem
        logger.info(f'New best model saved to: {model_path}. model_name: {model_name}')


        best_config = exp_config
        best_hist = result_hist
        best_config['device'] = str(best_config['device'])

        config_path = dict_files_dir.joinpath(f'{model_path.stem}_config.json')
        save2json(best_config, config_path)
        logger.info(f'New config for model {model_name} saved to: {config_path}')

        plotter_obj = Plotter(best_config['num_classes'], plots_dir = plot_dir)
        # iterate over every metric to plot it
        # group into one plot metric and metric val
        for metric_name, metric_values in result_hist.items():
            if 'val' in metric_name:
                continue

            val_metric = None
            if metric_name + '_val' in result_hist.keys():
                val_metric = result_hist[metric_name + '_val']

            plotter_obj.plot_metric_hist(f'{metric_name}_{model_name}.png',
                                         metric_values,
                                         val_metric=val_metric)

            logger.info(f'Metric {metric_name} plot for {model_name} saved to: {plot_dir}')


        self.save_new = False
        return model, best_config, config_path, best_hist

        




def case_based_training(exp_configs: list[dict],
                        model_name: str) -> None:
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: case_based_training.')

    pbar = tqdm(enumerate(exp_configs), total=len(exp_configs), desc="Training models", unit="model", position=0, leave=False)

    model_dir =  pth.Path(__file__).parent.joinpath('training_results').joinpath(f'{model_name.rsplit('_', 1)[0]}')
    model_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Model directory created: {model_dir}')

    model_path = model_dir.joinpath(f'{model_name}.pt')
    logger.info(f'Model path created: {model_path}')

    plot_dir = model_dir.joinpath('plots')
    dict_files_dir = model_dir.joinpath('dict_files')
    

    plot_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Plots directory created: {plot_dir}')
    dict_files_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Dict files directory created: {dict_files_dir}')
    

    logger.info('Case based training starting.')
    checkpoint = Checkpoint(existing_ok=False)


    for i, exp_config in pbar:
        logger.info(f'Case {i+1}/{len(exp_configs)}: {exp_config}')

        for model, result_hist in train_model(training_dict=exp_config):
            logger.info(f'Single model was generated. val_acc: {result_hist["val_acc"][-1]:.3f}  val_loss: {result_hist["val_loss"][-1]:.3f}')

            final_val = result_hist['val_acc'][-1]*0.6 + (1 / (1 + result_hist['val_loss'][-1]))*0.4

            model, best_config, config_path, result_hist = checkpoint.check_checkpoint(model, model_name, final_val, exp_config, result_hist)
    
    logger.info(f'Best model saved to: {model_path}')
    logger.info(f'Best config saved to: {config_path}')
    logger.info('STOP: case_based_training')

    summary(model)


def objective_function(trial: optuna.Trial,
                       exp_config: list[dict], # exp config, converted from str
                       model_name: str,
                       checkpoint: object) -> float:
    
    """
    Objective function for Optuna
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: objective_function')

    existing_ok = False

    batch_size = get_step_list(exp_config['batch_size'])
    epochs = get_step_list(exp_config['epochs'])

    lr = trial.suggest_categorical('learning_rate', get_factor_list(exp_config['learning_rate']))
    weight_decay = trial.suggest_categorical('weight_decay', get_factor_list(exp_config['weight_decay']))

    batch_size = trial.suggest_int('batch_size', exp_config['batch_size'][0], exp_config['batch_size'][1], step=exp_config['batch_size'][2])
    epochs = trial.suggest_int('epochs', exp_config['epochs'][0], exp_config['epochs'][1], step=exp_config['epochs'][2] )
    focal_loss_gamma = trial.suggest_float('focal_loss_gamma', exp_config['focal_loss_gamma'][0], exp_config['focal_loss_gamma'][1], step=exp_config['focal_loss_gamma'][2])

    pc_start = trial.suggest_float('pc_start', exp_config['pc_start'][0], exp_config['pc_start'][1], step=exp_config['pc_start'][2])
    div_factor = trial.suggest_int('div_factor', exp_config['div_factor'][0], exp_config['div_factor'][1], log=True)
    fin_div_factor = trial.suggest_int('final_div_factor', exp_config['final_div_factor'][0], exp_config['final_div_factor'][1], log=True)

    exp_config = exp_config.copy()
    exp_config.update({
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'focal_loss_gamma': focal_loss_gamma,
        'pc_start': pc_start,
        'div_factor': div_factor,
        'final_div_factor': fin_div_factor,
        'train_repeat':1
    })

    logger.info(f'Generated exp_config for trial: {trial.number}')
    for key, value in exp_config.items():
        if key != 'model_config':
            logger.info(f'parameters: {key}: {value}')

    
    # start training
    best_val_loss = float('inf')
    best_mAP05 = 0.
    best_mAP0595 = 0.
    best_precision = 0.
    best_recall = 0.
    best_f1 = 0.

    for epoch_idx, (model, result_hist) in enumerate(train_model(training_dict=exp_config)):
        
        if model is None or result_hist == {}:
            logger.error(f'ERROR: objective_function. Empty result_hist. Trial pruned.')
            trial.report(0.0, step=epoch_idx)
            raise optuna.exceptions.TrialPruned()
            
        
        # loss_hist = loss_hist,
        #   loss_v_hist = loss_v_hist,
        #   mAP05_hist = mAP05_hist,
        #   mAP0595_hist = mAP0595_hist,
        #   precision_hist = precision_hist,
        #   recall_hist = recall_hist,
        #   f1_hist = f1_hist     

        # goal is to maximize val_acc + 1/val_loss

        final_val_loss = result_hist['loss_hist_val'][-1]
        # final_mAP05 = result_hist['mAP05_hist'][-1]
        # final_mAP0595 = result_hist['mAP0595_hist'][-1]
        final_precision = result_hist['precision_hist'][-1]
        final_recall = result_hist['recall_hist'][-1]
        final_f1 = result_hist['f1_hist'][-1]

        best_val_loss = min(best_val_loss, final_val_loss)
        # best_mAP05 = max(best_mAP05, final_mAP05)
        # best_mAP0595 = max(best_mAP0595, final_mAP0595)
        best_precision = max(best_precision, final_precision)
        best_recall = max(best_recall, final_recall)
        best_f1 = max(best_f1, final_f1)



        normalized_loss = 1 / (1 + best_val_loss)
        final_val = 0.3*normalized_loss + 0.2*best_precision + 0.2*best_recall + 0.2*best_f1

        checkpoint.check_checkpoint(model,
                                    model_name,
                                    final_val,
                                    exp_config,
                                    result_hist)

        logger.info(f'Epoch {epoch_idx+1}/{exp_config["epochs"]}: best_precision: {best_precision:.3f}, best_recall: {best_recall:.3f}, best_f1: {best_f1:.3f}, final_val: {final_val:.3f}')

        # report acutal results for prunning
        trial.report(final_val, step=epoch_idx)

        if trial.should_prune():
            logger.info(f'Pruning trial: {trial.number}')
            raise optuna.exceptions.TrialPruned()
        
    logger.info(f'STOP: objective_function')

    return final_val

def optuna_based_training(exp_config: list[dict], # only one, non converted conf given in list
                          model_name: str,
                          n_trials: int = 100) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: optuna_based_training.')

    # able to automatically stop poor working exps
    n_startup = 3
    n_warmup_steps = 25
    interval_steps = 5

    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup_steps, interval_steps=interval_steps)
    logger.info(f'Pruner created: parameters: n_startup_trials: {n_startup}, n_warmup_step: {n_warmup_steps}, interval_steps: {interval_steps}')

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        study_name = f'{model_name}_{timestamp}',
        storage=f'sqlite:///db.sqlite3',
        directions=['maximize'],
        pruner=pruner)
    logger.info(f'Study created. Check ')

    train_repeat_old = exp_config['train_repeat']
    
    # Create progress bar
    pbar = tqdm(total=n_trials, desc="Optuna Optimization", unit="trial")
    
    def callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix({
                "Trial": trial.number,
                "Best Value": f"{study.best_value:.4f}",
                "Current": f"{trial.value:.4f}" if trial.value else "Pruned"
            })
        except ValueError:
            pbar.set_postfix({"Status": "Pruned"})

    checkpoint = Checkpoint(existing_ok=False)                          

    # Single optimize call with callback
    study.optimize(lambda trial: objective_function(trial,
                                                    exp_config=exp_config,
                                                    model_name = model_name,
                                                    checkpoint=checkpoint),
                   n_trials=n_trials,
                   callbacks=[callback])
    
    pbar.close()
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    logger.info(f'Optimization finished. Best value of formula: 0.6 * val_acc + 0.4 * norm_val_loss: {best_value:.4f}')

    print(20*'=')
    print(f'Optuna optimalization finished')
    print(f'Best value of 0.6 * val_acc + 0.4 * norm_val_loss: {best_value:.4f}')
    pprint(f'Best trial params:\n{best_params}')
    print(20*'=')

    final_exp_config = exp_config.copy()
    final_exp_config.update(best_params)
    final_exp_config.update({'train_repeat': train_repeat_old})

    print('Training the best model last time: ')

    case_based_training([final_exp_config],
                        model_name=model_name)
    
    logger.info(f'STOP: optuna_based_training')
    
def argparser():
        
    """
    Parse command-line arguments for automated CNN training pipeline configuration.
    Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """

    default_name = 'ResNet_0'
    parser = argparse.ArgumentParser(
        description="Script for training the model based on predefined range of scenarios",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default=default_name,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID. \n"
            f"If not given, defaults to: {default_name}."
        )
    )

    
    # Flag definition
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'], # choice limit
        help=(
            "Device for tensor based computation.\n"
            "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1, 2, 3], # choice limit
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: test\n'
            '1: single training\n'
            '2: multiple trainings, grid_based\n'
            '3: multiple trainings, with optuna'
        )
    )

    return parser.parse_args()


def main():
    multiprocessing.set_start_method('spawn', force=True)
    args = argparser()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = pth.Path(__file__).parent.joinpath('logs')
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file_name = f'{args.model_name}_{timestamp}_training.log'
    log_file_path = log_dir.joinpath(log_file_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"PROGRAM START: {args.model_name}")


    device = args.device.lower()
    model_name = args.model_name

    base_path = pth.Path(__file__).parent
    if args.mode != 4:
        exp_configs  = load_config(base_path, device, mode = args.mode)

    if args.mode == 0:
        test_case(exp_config=exp_configs[0])
    elif args.mode == 1 or args.mode==2:
        case_based_training(exp_configs=exp_configs,
                            model_name=model_name)
    elif args.mode == 3:
        optuna_based_training(exp_config=exp_configs,
                              model_name=model_name,
                              n_trials=80)

        

if __name__ == '__main__':
    
    main()  