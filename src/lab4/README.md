## Lab 01 - Testing Binary Relations 👨🏻‍💻

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

In this lab, we focused on testing binary relations (given a matrix/graph to extract binary relations). You could use [NetworkX](https://networkx.org/) to generate a graph/matrix to test different the binary relations.

You can check a detailed work through of this lab [here](https://github.com/mohammadzainabbas/DeM-Lab/blob/main/docs/preferences_as_binary_relationships.md)

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

