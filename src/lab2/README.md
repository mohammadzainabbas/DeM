## Lab 02 - Binary Relations via Linear Programming 👨🏻‍💻

### Table of contents

- [Introduction](#introduction)
  * [NetworkX](#network-x)
- [Dataset](#dataset)
- [Setup](#setup)
  * [Create new enviornment](#create-new-env)
  * [Setup `pre-commit` hooks](#setup-pre-commit)

#

<a id="introduction" />

### 1. Introduction

This lab focuses on implementing a decision model for some given constraints via _linear programming_. You could use the following python packages to solve _linear programs_:

- [x] [Linear Programming with Python and PuLP](http://benalexkeen.com/linear-programming-with-python-and-pulp/)
- [x] [scipy.optimize.linprog](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.linprog.html)

> For more details: checkout this [detailed guidline](https://realpython.com/linear-programming-python/#linear-programming-python-implementation)

You can check a detailed work through of this lab [here](https://github.com/mohammadzainabbas/DeM-Lab/blob/main/docs/binary_relationships_via_linear_programming.md)

<a id="network-x" />

#### 1.1. NetworkX

[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

#

<a id="dataset" />

### 2. Dataset

For the purpose of this lab, we will use _matrix/graph_ datasets. Instead of creating our own graphs (you are more then welcome if you have your own graph datasets), we will use some already existing datasets.

Save your data in this location:

```txt
src
└── lab1
  └── data
    └── data.csv
```

#

<a id="setup" />

### 3. Setup

If you want to follow along, make sure to clone and `cd` to this lab's directory:

```bash
git clone https://github.com/mohammadzainabbas/DeM-Lab.git
cd DeM-Lab/src/lab1
```

<a id="create-new-env" />

#### 3.1. Create new enviornment

Before starting, you may have to create new enviornment for the lab. Kindly, checkout the [documentation](https://github.com/mohammadzainabbas/DeM-Lab/blob/main/docs/SETUP_ENV.md) for creating an new environment.

#

Once, you have activated your new enviornment, we will install all the dependencies of this project:

```bash
pip install -r requirements.txt
```

<a id="setup-pre-commit" />

#### 3.2. Setup `pre-commit` hooks

In order to setup `pre-commit` hooks, please refer to the [documentation](https://github.com/mohammadzainabbas/DeM-Lab/blob/main/docs/SETUP_PRE-COMMIT_HOOKS.md).

#

