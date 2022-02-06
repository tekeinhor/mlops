# Technical test

## Goal
The goal of this project is to detect bots that mimic users by producing fake clicks and leads.
The dataset of logs contains the following info: `UserId`, `Event` (action made by a user), `Category` (category she interacts with), `Fake` (indication about user being fake or not).
The task is to create code that:
- takes as input a `.csv` file with the same structure as the dataset.
- output a `.csv` with two columns: `UserId` and `is_fake_probability`.
- recommend threshold to user classification.

This project is done with [python 3.10](https://www.python.org/downloads/).


## Project structures
This project contains the following files:

```bash
├── Dockerfile # dockerfile for prediction
├── data # .csv files for train and test
├── docs # going further discussion
├── features # stored encoder
├── mlruns # stored model using mlflow pkg
├── notebooks # exploration notebooks
├── readme.md # this file
├── requirements.txt #list of required package to make this code works
├── src # main source code
├── testdata # data necessary for test
└── tests # test code
```

## How to run the code
### Requirements :
Please see `environment.yml` for package list necessary to run the code.

I advise running the code inside a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) virtual env.

```bash
# create env with <env-name> name
$ conda create -n <env-name> python=3.10

# make sure your in the working directory (inside /adevinta)
# create env directly from (the env name will be lbc)
# the name can be find in the first line of environment.yml
$ conda env create -f environment.yml

# activate conda env
$ conda activate lbc

# deactivate conda
$ conda deactivate
```

Please follow the guide the official [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more info.

### Notebooks
This project is delivered with two notebooks to illustrate the exploration phase of the work.

PS : Warning !!! If you are using a virtual env, make sure to run your notebook with the appropriate kernel. [For more info](https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab).



## Training
The training is made by running the following code. [Mlflow](https://mlflow.org/) is used for model tracking.
```bash
# make sure you are in the working directory (/adevinta) and run:
python -m src.training
```

The models will be stored in `adevinta/mlruns` folder.

Mlflow has a UI that can be activate as follow:

```bash
# run this command, then go to localhost:5000
$  mlflow ui
```

## Prediction

### Prediction through python
```bash
# make sure you are in the working directory (/adevinta) and run to do a prediction
# use --data to pass the input csv file, --output for the output csv --use-latest
$ python -m src.cli --data "data/test.csv" --output "output.csv" --latest
```
### Prediction through docker

If you have docker installed you can rather use it (or instructions for installation are [here](https://docs.docker.com/get-docker/)) to make the prediction. You can pass the cli arguments through docker.

```bash
# make sure you are in the project folder

# build your docker
$ docker build -t <your_tag> .

# run your docker
$ docker run  <your_tag> --data "data/test.csv" --output "output.csv" --latest
```


### Run your test files
To run your test file, run the following command:

```bash
# make sure you are in the working directory (/adevinta) and run:
$ python -m pytest
```

## Conclusion

For more info about going further please look at [docs/discussion.md](/docs/discussion.md)