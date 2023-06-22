
# Decision Transformer

The code is based on the decision transformer [code](https://github.com/kzl/decision-transformer). I write my own test code in `toy/` directory. 

## Installing dependencies
I listed my conda environment in `toy/dependency.txt`.

## Run code
The file paths below are all relevant to path `toy/`.

There are two main files. `run_toy.py` runs rvs/dt method, while `run_cql.py` runs Q-learning/cql method.

A sample running script is provided in `run.sh`. 

During running, the evaluation result will be printed, and tensorboard logs will be written in `logs/` directory.

## Customize environments and datasets
You can define customized environments by writting a environment description file under `env/` directory. You may see the `env/*.txt` files for the formats.

After defining environment, you should create a dataset by running `create_dataset.py`. Remember you need to provide the environment description file path in the command-line argument. You may also need to modify the code `create_dataset.py` to specify the behavior policy. By default, the dataset file will be written to `dataset/` directory and is stored as a csv file.
