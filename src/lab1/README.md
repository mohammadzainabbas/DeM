## Lab 01 - Testing Binary Relations 👨🏻‍💻

### Table of contents

- [Introduction](#introduction)
  * [NetworkX](#network-x)
- [Dataset](#dataset)
  * [SNAP for Python](#snap-for-python)
  * [WormNet v3](#wormnet-v3)
- [Setup](#setup)
  * [Create new enviornment](#create-new-env)
  * [Setup `pre-commit` hooks](#setup-pre-commit)

#

<a id="introduction" />

### 1. Introduction

__`Data drives the world.`__ Nowadays, most of the data (_structured_ or _unstructured_) can be analysed as a graph. Today, many practical computing problems concern large graphs. Standard examples include the Web graph and various social networks. The scale of these graphs (_in some cases billions of vertices, trillions of edges_) poses challenges to their efficient processing.

In this lab, we will focus on some basic graph algorithms and see how we can utilise these algorithms to efficiently analyse our data. Since, there exist many similarities between graph theory and network science, you will see us using network science related packages as well. 

<a id="network-x" />

#### 1.1. NetworkX

[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

#

<a id="dataset" />

### 2. Dataset

For the purpose of this lab, we will use graph datasets. Instead of creating our own graphs (you are more then welcome if you have your own graph datasets), we will use some already existing datasets.

<a id="snap-for-python" />

#### 2.1. SNAP for Python

[Snap.py](https://snap.stanford.edu/snappy/) is a Python interface for SNAP. SNAP is a general purpose, high performance system for analysis and manipulation of large networks. SNAP is written in C++ and optimized for maximum performance and compact graph representation. It easily scales to massive networks with hundreds of millions of nodes, and billions of edges.

SNAP also provides some graph datasets which we will use in this lab. List of available datasets can be found [here](https://snap.stanford.edu/data/index.html).

<a id="wormnet-v3" />

#### 2.2. WormNet v3

[WormNet](https://www.inetbio.org/wormnet/) provides a genes data in the form of a network. It's basically a network-assisted hypothesis-generating server for `Caenorhabditis elegans`. List of available datasets can be found [here](https://www.inetbio.org/wormnet/downloadnetwork.php).

WormNet is part of [YONSEI Network Biology Lab](https://netbiolab.org/w/Welcome). You can check out other available software/data [here](https://netbiolab.org/w/Software).

#

<a id="setup" />

### 3. Setup

If you want to follow along, make sure to clone and `cd` to this lab's directory:

```bash
git clone https://github.com/mohammadzainabbas/MGMA-Lab.git
cd MGMA-Lab/src/lab1
```

<a id="create-new-env" />

#### 3.1. Create new enviornment

Before starting, you may have to create new enviornment for the lab. Kindly, checkout the [documentation](https://github.com/mohammadzainabbas/MGMA-Lab/blob/main/docs/SETUP_ENV.md) for creating an new environment.

#

Once, you have activated your new enviornment, we will install all the dependencies of this project:

```bash
pip install -r requirements.txt
```

<a id="setup-pre-commit" />

#### 3.2. Setup `pre-commit` hooks

In order to setup `pre-commit` hooks, please refer to the [documentation](https://github.com/mohammadzainabbas/MGMA-Lab/blob/main/docs/SETUP_PRE-COMMIT_HOOKS.md).

#

