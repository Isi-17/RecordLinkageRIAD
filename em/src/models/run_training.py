import os
import sys
import shlex

# Comment while not running the job
module_path = os.path.join(os.getcwd(), 'em')
sys.path.append(module_path)
os.chdir(module_path)

from src.models.pytorch_model import PyTorchModel
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_train
from src.helpers.logging_helper import setup_logging


setup_logging()


def main(args):
    initialize_gpu_seed(args.model_seed)

    model = PyTorchModel(args)
    model.train()


if __name__ == '__main__':
    env_args = os.environ.get("JOB_ARGUMENTS", None)
    
    if env_args:
        sys.argv = ["run_training.py"] + shlex.split(env_args)
        print(f"Running with JOB_ARGUMENTS: {sys.argv}")
        
    args = read_arguments_train()
    main(args)