# rl-sandbox

Personal sandbox project for testing reinforcement learning algorithms.

## Setup

### Install dependencies

It is recommended to use a `conda` environment. To create a new environment, run:

```bash
conda create -n rl-sandbox python=3.8
```

Activate the environment:

```bash
conda activate rl-sandbox
```

Install dependencies:

```bash
pip install swig
pip install -r requirements.txt
```

## Run the code

The code is organized by environment and method, for example `lunar_lander/A2C` contains the code for the A2C algorithm on the Lunar Lander environment.
To run or train an algorithm, navigate to the corresponding directory and run:

```bash
python train.py --resume [INSTANCE_NAME] --save_instance [INSTANCE_NAME]
```

Where `INSTANCE_NAME` is the name of the instance to run or resume. If `INSTANCE_NAME` is not provided, a new instance will be created. You can omit the `--resume` flag to start a new instance. `--save_instance` is used to save the instance to a file under `env/method/instances`, this argment is required.

You can tweak the parameters of the algorithm by editing the `parameters.py` file in the corresponding directory. Here you can toggle `render_environment` to render the environment while training, as well as toggle `save_model` to save the model after training

## Plot instance results

To plot the results of an instance, run:

```bash
python plot.py --instance [INSTANCE_NAME]
```


## TODO

- [X] Update `gym` version to maintened fork `gymnasium`
- [ ] Refactor code to remove duplicate code
- [ ] Solve atari games