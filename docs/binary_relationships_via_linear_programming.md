### Import libraries


```python
%load_ext autoreload
%autoreload

from os import getcwd
from os.path import join, abspath, pardir, relpath, exists

from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from numpy import matrixlib as npmat
import networkx as nx
from typing import Union
import pulp as p
from itertools import combinations
from typing import List, Tuple
from enum import Enum
from scipy.stats import kendalltau, spearmanr

from IPython.display import IFrame
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


### Helper methods


```python
# ------------------------ #
# Helper logging functions
# ------------------------ #
def print_log(text: str) -> None:
    """ Prints the log """
    print(f"[ log ]: {text}")

def print_error(text: str) -> None:
    """ Prints the error """
    print(f"[ error ]: {text}")
# -------------------------------------------------- #
# Helper functions
# -------------------------------------------------- #
def is_identical(list1: List, list2: List) -> bool:
    """Check if two lists are identical."""
    return sorted(list1) == sorted(list2)
```

### Documentation


```python
parent_dir = abspath(join(join(getcwd(), pardir), pardir))
data_dir = join(parent_dir, 'data')
docs_dir = join(parent_dir, 'docs')
if exists(docs_dir):
    doc_file = relpath(join(docs_dir, 'practical_works_linear_programing_v3.pdf'))
    IFrame(doc_file, width=1200, height=350)
```





<iframe
    width="1200"
    height="350"
    src="../../docs/practical_works_linear_programing_v3.pdf"
    frameborder="0"
    allowfullscreen

></iframe>




#### General settings


```python
LpSolverDefault = p.PULP_CBC_CMD
show_solver_output = False
```

#### Linear Programming via PuLP


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="The Miracle Worker", sense=p.const.LpMaximize)

# Create problem variables
x = p.LpVariable(name="Medicine_1_units", lowBound=0, upBound=None, cat=p.LpInteger)
y = p.LpVariable(name="Medicine_2_units", lowBound=0, upBound=None, cat=p.LpInteger)

# The objective function is added to "problem" first
problem += 25*x + 20*y, "Health restored; to be maximized"

# The two contraints for the herbs
problem += 3*x + 4*y <= 25, "Herb A constraint"
problem += 2*x + y <= 10, "Herb B constraint"

# The problem data is written to an .lp file
# problem.writeLP(filename=join(data_dir, "miracle_worker.lp"), writeSOS=1, mip=1, max_length=100)
problem.writeLP(filename=join(data_dir, "miracle_worker.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    /opt/homebrew/Caskroom/mambaforge/base/envs/decision_modelling/lib/python3.10/site-packages/pulp/pulp.py:1352: UserWarning: Spaces are not permitted in the name. Converted to '_'
      warnings.warn("Spaces are not permitted in the name. Converted to '_'")





    [Medicine_1_units, Medicine_2_units]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

_ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]

print_log(f"{p.value(problem.objective) = }")
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: Medicine_1_units = 3.0
    [ log ]: Medicine_2_units = 4.0
    [ log ]: p.value(problem.objective) = 155.0


#### Toy example (Linear Programming via PuLP)


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Toy Manufacturing", sense=p.const.LpMaximize)

# Create problem variables
x = p.LpVariable(name="Toy_1_units", lowBound=0, upBound=None, cat=p.LpInteger)
y = p.LpVariable(name="Toy_2_units", lowBound=0, upBound=None, cat=p.LpInteger)

# The objective function is added to "problem" first
problem += 25*x + 20*y, "Profit; to be maximized"

# The two contraints for the herbs
problem += 20*x + 12*y <= 2000, "Required units - constraint"
problem += 5*x + 5*y <= 540, "Time required - constraint"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "toy_manufacturing.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```




    [Toy_1_units, Toy_2_units]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

_ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]

print_log(f"{p.value(problem.objective) = }")
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: Toy_1_units = 88.0
    [ log ]: Toy_2_units = 20.0
    [ log ]: p.value(problem.objective) = 2600.0


### How to visit Paris ? (efficiently with low budget)


```python
@dataclass(frozen=False, order=False)
class SiteInfo:
    """
    A dataclass to hold the site information
    """
    name: str = field(default="")
    site_code: str = field(default="")
    price: float = field(default=0.0) # price in euros
    duration: float = field(default=0.0) # duration in hours
    rating: int = field(default=0) # appreciation rating
```


```python
sites_info = [
    SiteInfo(name="La Tour Eiffel", site_code="TE", duration=4.5, rating=5, price=15.50),
    SiteInfo(name="Le Musée du louvre", site_code="ML", duration=3, rating=4, price=12),
    SiteInfo(name="l’Arc de triomphe", site_code="AT", duration=1, rating=3, price=9.50),
    SiteInfo(name="le Musée d’Orsay", site_code="MO", duration=2, rating=2, price=11),
    SiteInfo(name="le Jardin des tuileries", site_code="JT", duration=1.5, rating=3, price=0),
    SiteInfo(name="les Catacombes", site_code="CA", duration=2, rating=4, price=10),
    SiteInfo(name="le Centre Pompido", site_code="CP", duration=2.5, rating=1, price=10),
    SiteInfo(name="la Cathédrale Notre Dame de Paris", site_code="CN", duration=2, rating=5, price=5),
    SiteInfo(name="la Basilique du Sacré-Coeur", site_code="BS", duration=2, rating=4, price=8),
    SiteInfo(name="la Sainte Chapelle", site_code="SC", duration=1.5, rating=1, price=8.50),
    SiteInfo(name="La Place de la Concorde", site_code="PC", duration=0.75, rating=3, price=0),
    SiteInfo(name="la Tour Montparnasse", site_code="TM", duration=2, rating=2, price=15),
    SiteInfo(name="l’Avenue des Champs-Elysées", site_code="AC", duration=1.5, rating=5, price=0),
]
```


```python
sites = [x.site_code for x in sites_info]

distance_in_kms = npmat.asmatrix(data=[
        [0, 3.8, 2.1, 2.4, 3.5, 4.2, 5.0,  4.4, 5.5, 4.2, 2.5, 3.1, 1.9],
        [0,   0, 3.8, 1.1, 1.3, 3.3, 1.3,  1.1, 3.4, 0.8, 1.7, 2.5, 2.8],
        [0,   0,   0, 3.1, 3.0, 5.8, 4.8,  4.9, 4.3, 4.6, 2.2, 4.4, 1.0],
        [0,   0,   0,   0, 0.9, 3.1, 2.5,  2.0, 3.9, 1.8, 1.0, 2.3, 2.1],
        [0,   0,   0,   0,   0, 4.2, 2.0,  2.4, 2.7, 2.0, 1.0, 3.4, 2.1],
        [0,   0,   0,   0,   0,   0, 3.5,  2.7, 6.5, 2.6, 3.8, 1.3, 4.9],
        [0,   0,   0,   0,   0,   0,   0, 0.85, 3.7, 0.9, 2.7, 3.4, 3.8],
        [0,   0,   0,   0,   0,   0,   0,    0, 4.5, 0.4, 2.8, 2.7, 3.9],
        [0,   0,   0,   0,   0,   0,   0,    0,   0, 4.2, 3.3, 5.7, 3.8],
        [0,   0,   0,   0,   0,   0,   0,    0,   0,   0, 2.5, 2.6, 3.6],
        [0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0, 3.0, 1.2],
        [0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0, 2.1],
        [0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0]
    ], dtype=float)
    
distance_df = pd.DataFrame(np.matrix(distance_in_kms.T + distance_in_kms), columns=sites, index=sites)
distance_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TE</th>
      <th>ML</th>
      <th>AT</th>
      <th>MO</th>
      <th>JT</th>
      <th>CA</th>
      <th>CP</th>
      <th>CN</th>
      <th>BS</th>
      <th>SC</th>
      <th>PC</th>
      <th>TM</th>
      <th>AC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TE</th>
      <td>0.0</td>
      <td>3.8</td>
      <td>2.1</td>
      <td>2.4</td>
      <td>3.5</td>
      <td>4.2</td>
      <td>5.00</td>
      <td>4.40</td>
      <td>5.5</td>
      <td>4.2</td>
      <td>2.5</td>
      <td>3.1</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>ML</th>
      <td>3.8</td>
      <td>0.0</td>
      <td>3.8</td>
      <td>1.1</td>
      <td>1.3</td>
      <td>3.3</td>
      <td>1.30</td>
      <td>1.10</td>
      <td>3.4</td>
      <td>0.8</td>
      <td>1.7</td>
      <td>2.5</td>
      <td>2.8</td>
    </tr>
    <tr>
      <th>AT</th>
      <td>2.1</td>
      <td>3.8</td>
      <td>0.0</td>
      <td>3.1</td>
      <td>3.0</td>
      <td>5.8</td>
      <td>4.80</td>
      <td>4.90</td>
      <td>4.3</td>
      <td>4.6</td>
      <td>2.2</td>
      <td>4.4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MO</th>
      <td>2.4</td>
      <td>1.1</td>
      <td>3.1</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>3.1</td>
      <td>2.50</td>
      <td>2.00</td>
      <td>3.9</td>
      <td>1.8</td>
      <td>1.0</td>
      <td>2.3</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>JT</th>
      <td>3.5</td>
      <td>1.3</td>
      <td>3.0</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>4.2</td>
      <td>2.00</td>
      <td>2.40</td>
      <td>2.7</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.4</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>4.2</td>
      <td>3.3</td>
      <td>5.8</td>
      <td>3.1</td>
      <td>4.2</td>
      <td>0.0</td>
      <td>3.50</td>
      <td>2.70</td>
      <td>6.5</td>
      <td>2.6</td>
      <td>3.8</td>
      <td>1.3</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>CP</th>
      <td>5.0</td>
      <td>1.3</td>
      <td>4.8</td>
      <td>2.5</td>
      <td>2.0</td>
      <td>3.5</td>
      <td>0.00</td>
      <td>0.85</td>
      <td>3.7</td>
      <td>0.9</td>
      <td>2.7</td>
      <td>3.4</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>CN</th>
      <td>4.4</td>
      <td>1.1</td>
      <td>4.9</td>
      <td>2.0</td>
      <td>2.4</td>
      <td>2.7</td>
      <td>0.85</td>
      <td>0.00</td>
      <td>4.5</td>
      <td>0.4</td>
      <td>2.8</td>
      <td>2.7</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>BS</th>
      <td>5.5</td>
      <td>3.4</td>
      <td>4.3</td>
      <td>3.9</td>
      <td>2.7</td>
      <td>6.5</td>
      <td>3.70</td>
      <td>4.50</td>
      <td>0.0</td>
      <td>4.2</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>SC</th>
      <td>4.2</td>
      <td>0.8</td>
      <td>4.6</td>
      <td>1.8</td>
      <td>2.0</td>
      <td>2.6</td>
      <td>0.90</td>
      <td>0.40</td>
      <td>4.2</td>
      <td>0.0</td>
      <td>2.5</td>
      <td>2.6</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>PC</th>
      <td>2.5</td>
      <td>1.7</td>
      <td>2.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.8</td>
      <td>2.70</td>
      <td>2.80</td>
      <td>3.3</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>TM</th>
      <td>3.1</td>
      <td>2.5</td>
      <td>4.4</td>
      <td>2.3</td>
      <td>3.4</td>
      <td>1.3</td>
      <td>3.40</td>
      <td>2.70</td>
      <td>5.7</td>
      <td>2.6</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>AC</th>
      <td>1.9</td>
      <td>2.8</td>
      <td>1.0</td>
      <td>2.1</td>
      <td>2.1</td>
      <td>4.9</td>
      <td>3.80</td>
      <td>3.90</td>
      <td>3.8</td>
      <td>3.6</td>
      <td>1.2</td>
      <td>2.1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



##### 1. It is assumed that Mr. Doe gives equal importance to each tourist site, and he wants to visit the maximum number of sites. Which list(s) of places could you recommend to him ? This solution will be called `ListVisit 1`.


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites", sense=p.const.LpMaximize)

# Create the variables
for site in sites:
    site_info = next(x for x in sites_info if x.site_code == site)
    print_log(f"Creating variable for {site = }")
    globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)

# Create the objective function
# problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)])
problem += p.lpSum([globals()[f"{site}"] * 1 for site in sites]), "Max. number of sites"

# Create the constraints
# 1. Max. duration
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"
# 2. Max. price
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "ListVisit_1.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'


    /opt/homebrew/Caskroom/mambaforge/base/envs/decision_modelling/lib/python3.10/site-packages/pulp/pulp.py:1352: UserWarning: Spaces are not permitted in the name. Converted to '_'
      warnings.warn("Spaces are not permitted in the name. Converted to '_'")





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

# _ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]
to_visit = []
for v in problem.variables():
    print_log(f"{v.name} = {v.varValue}")
    if v.varValue == 1:
        site = next(x for x in sites_info if x.site_code == v.name)
        to_visit.append(site.name)

print_log(f"{p.value(problem.objective) = }")
print_log(f"You should visit total '{p.value(problem.objective)}' places. i.e:\n\n{' --- '.join(to_visit)}")

listvisit_1 = to_visit.copy() # save the result for later use
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 1.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 7.0
    [ log ]: You should visit total '7.0' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- La Place de la Concorde --- la Sainte Chapelle


##### 2. Actually, Mr. Doe has some preferences among these tourist sites and he expresses them as follows:

- Preference 1 : If two sites are geographically very close (within a radius of 1 km of walking), he will prefer to visit these two sites instead of visiting only one.


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius)", sense=p.const.LpMaximize)

# Create the variables
for site in sites:
    site_info = next(x for x in sites_info if x.site_code == site)
    print_log(f"Creating variable for {site = }")
    globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)

# Create the objective function
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"

# Create the constraints
# 1. Max. duration
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"
# 2. Max. price
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"

# 3. Distance between sites
for i, site1 in enumerate(sites):
    for j, site2 in enumerate(sites):
        if distance_df.loc[site1, site2] <= 1 and site1 != site2: # if distance is between 0 and 1 km
            # print_log(f"{site1 = } {site2 = } -> {distance_df.loc[site1, site2]}")
            problem += globals()[f"{site1}"] - globals()[f"{site2}"] == 0, f"Distance between {site1} and {site2}"
            # problem += locals()[f"{site1}"] + locals()[f"{site2}"] <= 1 + distance_df.loc[site1, site2] / 10, f"Distance between {site1} and {site2}"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "ListVisit_2_a.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

# _ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]
to_visit = []
for v in problem.variables():
    print_log(f"{v.name} = {v.varValue}")
    if v.varValue == 1:
        site = next(x for x in sites_info if x.site_code == v.name)
        to_visit.append(site.name)

print_log(f"{p.value(problem.objective) = }")
print_log(f"You should visit total '{len(to_visit)}' places. i.e:\n\n{' --- '.join(to_visit)}")

listvisit_2a = to_visit.copy() # save the result for later use
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 1.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 24.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- le Jardin des tuileries --- le Musée d’Orsay --- La Place de la Concorde


- Preference 2 : He absolutely wants to visit the `Eiffel Tower` (TE) and `Catacombes` (CA).

Two solutions:

1. Set lower bounds for `TE` and `CA` to 1 (minimum value will be 1 i.e: must visit them)
2. Add constraints explicitly for `TE` and `CA` (for e.g: `TE` == 1)


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit effiel tower and catacombs)", sense=p.const.LpMaximize)

# Create the variables
for site in sites:
    site_info = next(x for x in sites_info if x.site_code == site)
    print_log(f"Creating variable for {site = }")
    globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)

# Create the objective function
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"

# Create the constraints
# 1. Max. duration
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"

# 2. Max. price
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"

# 3. Must visit Effiel Tower (TE) and Catacombs (CA)
must_visit = ["TE", "CA"]
for site in must_visit:
    problem += globals()[f"{site}"] == 1, f"Must visit {site}"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "ListVisit_2_b.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

# _ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]
to_visit = []
for v in problem.variables():
    print_log(f"{v.name} = {v.varValue}")
    if v.varValue == 1:
        site = next(x for x in sites_info if x.site_code == v.name)
        to_visit.append(site.name)

print_log(f"{p.value(problem.objective) = }")
print_log(f"You should visit total '{len(to_visit)}' places. i.e:\n\n{' --- '.join(to_visit)}")

listvisit_2b = to_visit.copy() # save the result for later use
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 0.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 25.0
    [ log ]: You should visit total '6' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- les Catacombes --- la Cathédrale Notre Dame de Paris --- La Place de la Concorde --- La Tour Eiffel


- Preference 3 : If he visits `Notre Dame Cathedral` (CN) then he will not visit the `Sainte Chapelle` (SC).


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (either visit Notre Dame Cathedral or Sainte Chapelle)", sense=p.const.LpMaximize)

# Create the variables
for site in sites:
    site_info = next(x for x in sites_info if x.site_code == site)
    print_log(f"Creating variable for {site = }")
    globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)

# Create the objective function
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"

# Create the constraints
# 1. Max. duration
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"

# 2. Max. price
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"

# 3. Either visit Notre Dame Cathedral or Sainte Chapelle
either_visit = ["TE", "CA"]
for _site in combinations(either_visit, 2):
    site1, site2 = _site
    problem += globals()[f"{site1}"] + globals()[f"{site2}"] == 1, f"Either visit {site1} or {site2}"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "ListVisit_2_c.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

# _ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]
to_visit = []
for v in problem.variables():
    print_log(f"{v.name} = {v.varValue}")
    if v.varValue == 1:
        site = next(x for x in sites_info if x.site_code == v.name)
        to_visit.append(site.name)

print_log(f"{p.value(problem.objective) = }")
print_log(f"You should visit total '{len(to_visit)}' places. i.e:\n\n{' --- '.join(to_visit)}")

listvisit_2c = to_visit.copy() # save the result for later use
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 27.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- le Jardin des tuileries --- La Place de la Concorde


- Preference 4 : He absolutely wants to visit `Tour Montparnasse` (TM).


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit tour montparnasse)", sense=p.const.LpMaximize)

# Create the variables
for site in sites:
    site_info = next(x for x in sites_info if x.site_code == site)
    print_log(f"Creating variable for {site = }")
    globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)

# Create the objective function
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"

# Create the constraints
# 1. Max. duration
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"

# 2. Max. price
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"

# 3. Must visit Tour Montparnasse (TM)
must_visit = ["TM"]
for site in must_visit:
    problem += globals()[f"{site}"] == 1, f"Must visit {site}"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "ListVisit_2_d.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

# _ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]
to_visit = []
for v in problem.variables():
    print_log(f"{v.name} = {v.varValue}")
    if v.varValue == 1:
        site = next(x for x in sites_info if x.site_code == v.name)
        to_visit.append(site.name)

print_log(f"{p.value(problem.objective) = }")
print_log(f"You should visit total '{len(to_visit)}' places. i.e:\n\n{' --- '.join(to_visit)}")

listvisit_2d = to_visit.copy() # save the result for later use
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 26.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- La Place de la Concorde --- la Tour Montparnasse


- Preference 5 : If he visits the `Louvre` (ML) Museum then he must visit the `Pompidou Center` (CP).


```python
# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
for site in sites:
    site_info = next(x for x in sites_info if x.site_code == site)
    print_log(f"Creating variable for {site = }")
    globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)

# Create the objective function
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"

# Create the constraints
# 1. Max. duration
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"

# 2. Max. price
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"

# 3. Must visit the Pompidou Center (CP) if he visits Louvre (ML)
problem += globals()[f"{ML}"] - globals()[f"{CP}"] <= 0, f"Must visit CP if he visits ML"

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "ListVisit_2_e.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



#### Output


```python
print_log(f"{p.LpStatus[problem.status] = }")

# _ = [print_log(f"{v.name} = {v.varValue}") for v in problem.variables()]
to_visit = []
for v in problem.variables():
    print_log(f"{v.name} = {v.varValue}")
    if v.varValue == 1:
        site = next(x for x in sites_info if x.site_code == v.name)
        to_visit.append(site.name)

print_log(f"{p.value(problem.objective) = }")
print_log(f"You should visit total '{len(to_visit)}' places. i.e:\n\n{' --- '.join(to_visit)}")

listvisit_2e = to_visit.copy() # save the result for later use
```

    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 27.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- le Jardin des tuileries --- La Place de la Concorde


#### 2.a - For each of the five preferences above, suggest to Mr. Doe, one or more lists of tourist sites to visit. Are the obtained lists different from the solution `ListVisit 1` ? To answer this last question, you can implement a python function returning `True` (respectively `False`) if two lists are identical (respectively different).


```python
for i, x in enumerate(list('abcde')):
    print_log(f"Are 'ListVisit 1' and output of 'Preference {i + 1}' same ? {is_identical(listvisit_1, globals()[f'listvisit_2{x}'])}")
```

    [ log ]: Are 'ListVisit 1' and output of 'Preference 1' same ? False
    [ log ]: Are 'ListVisit 1' and output of 'Preference 2' same ? False
    [ log ]: Are 'ListVisit 1' and output of 'Preference 3' same ? False
    [ log ]: Are 'ListVisit 1' and output of 'Preference 4' same ? False
    [ log ]: Are 'ListVisit 1' and output of 'Preference 5' same ? False



```python
#--------------------------------------------
# Helper functions for adding constraints
#--------------------------------------------
def create_variables(problem: p.LpProblem, sites: List, sites_info: List[SiteInfo]) -> p.LpProblem:
    for site in sites:
        site_info = next(x for x in sites_info if x.site_code == site)
        print_log(f"Creating variable for {site = }")
        globals()[f"{site}"] = p.LpVariable(name=f"{site}", lowBound=0, upBound=1, cat=p.LpInteger)
    return problem

def create_objective_function(problem: p.LpProblem, sites: List, sites_info: List[SiteInfo]) -> p.LpProblem:
    problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"
    return problem

def add_generic_constraints(problem: p.LpProblem, sites: List, sites_info: List[SiteInfo]) -> p.LpProblem:
    # 1. Max. duration
    problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"
    # 2. Max. price
    problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"
    return problem

class PREFERENCE(str, Enum):
    ONE = "ONE"
    TWO = "TWO"
    THREE = "THREE"
    FOUR = "FOUR"
    FIVE = "FIVE"
    def __str__(self) -> str:
        return self.value

def add_specific_constraints(problem: p.LpProblem, preference: PREFERENCE, distance_df: pd.DataFrame, sites: List, sites_info: List[SiteInfo], verbose: bool = False) -> p.LpProblem:
    if verbose: print_log(f"Adding constraints for {preference = }")
    if preference == str(PREFERENCE.ONE):
        # 3. Distance between sites (within 1 km radius)
        for i, site1 in enumerate(sites):
            for j, site2 in enumerate(sites):
                if distance_df.loc[site1, site2] <= 1 and site1 != site2: # if distance is between 0 and 1 km
                    if verbose: print_log(f"{site1 = } {site2 = } -> {distance_df.loc[site1, site2]}")
                    problem += globals()[f"{site1}"] - globals()[f"{site2}"] == 0, f"Distance between {site1} and {site2}"
    elif preference == str(PREFERENCE.TWO):
        # 3. Must visit Effiel Tower (TE) and Catacombs (CA)
        must_visit = ["TE", "CA"]
        for site in must_visit:
            problem += globals()[f"{site}"] == 1, f"Must visit {site}"
    elif preference == str(PREFERENCE.THREE):
        # 3. Either visit Notre Dame Cathedral or Sainte Chapelle
        either_visit = ["TE", "CA"]
        for _site in combinations(either_visit, 2):
            site1, site2 = _site
            problem += globals()[f"{site1}"] + globals()[f"{site2}"] == 1, f"Either visit {site1} or {site2}"
    elif preference == str(PREFERENCE.FOUR):
        # 3. Must visit Tour Montparnasse (TM)
        must_visit = ["TM"]
        for site in must_visit:
            problem += globals()[f"{site}"] == 1, f"Must visit {site}"
    elif preference == str(PREFERENCE.FIVE):
        # 3. Must visit the Pompidou Center (CP) if he visits Louvre (ML)
        problem += globals()[f"{ML}"] - globals()[f"{CP}"] <= 0, f"Must visit CP if he visits ML"
    return problem

def display_solver_output(problem: p.LpProblem, sites_info: List[SiteInfo]) -> List[str]:
    print_log(f"{p.LpStatus[problem.status] = }")
    
    to_visit = []
    for v in problem.variables():
        print_log(f"{v.name} = {v.varValue}")
        if v.varValue == 1:
            site = next(x for x in sites_info if x.site_code == v.name)
            to_visit.append(site.name)

    print_log(f"{p.value(problem.objective) = }")
    print_log(f"You should visit total '{len(to_visit)}' places. i.e:\n\n{' --- '.join(to_visit)}")
    return to_visit
```

#### 2.b - If Mr. Doe wishes, at the same time, to take into account `Preference 1` and `Preference 2`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius & must visit effiel tower and catacombs)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.ONE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2b.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2b = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 0.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 21.0
    [ log ]: You should visit total '5' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- La Tour Eiffel


#### 2.c - If Mr. Doe wishes, at the same time, to take into account `Preference 1` and `Preference 3`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius & either visit Notre Dame Cathedral or Sainte Chapelle)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.ONE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.THREE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2c.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2c = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 1.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 24.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- le Jardin des tuileries --- le Musée d’Orsay --- La Place de la Concorde


#### 2.d - If Mr. Doe wishes, at the same time, to take into account `Preference 1` and `Preference 4`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius & must visit Tour Montparnasse)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.ONE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2d.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2d = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 0.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 1.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 22.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- les Catacombes --- le Jardin des tuileries --- le Musée d’Orsay --- La Place de la Concorde --- la Tour Montparnasse


#### 2.e - If Mr. Doe wishes, at the same time, to take into account `Preference 2` and `Preference 5`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit effiel tower and catacombs & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FIVE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2e.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2e = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 0.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 25.0
    [ log ]: You should visit total '6' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- les Catacombes --- la Cathédrale Notre Dame de Paris --- La Place de la Concorde --- La Tour Eiffel


#### 2.f - If Mr. Doe wishes, at the same time, to take into account `Preference 3` and `Preference 4`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit effiel tower and catacombs & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.THREE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2f.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2f = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 0.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 26.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- le Jardin des tuileries --- La Place de la Concorde --- la Tour Montparnasse


#### 2.g - If Mr. Doe wishes, at the same time, to take into account `Preference 4` and `Preference 5`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit tour montparnasse & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FIVE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2g.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2g = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 0.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 26.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- La Place de la Concorde --- la Tour Montparnasse


#### 2.h - If Mr. Doe wishes, at the same time, to take into account `Preference 1`, `Preference 2` and `Preference 4`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius & must visit effiel tower and catacombs & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.ONE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2h.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2h = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 0.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 0.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 19.0
    [ log ]: You should visit total '5' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- les Catacombes --- La Tour Eiffel --- la Tour Montparnasse


#### 2.i - If Mr. Doe wishes, at the same time, to take into account `Preference 2`, `Preference 3` and `Preference 5`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit effiel tower and catacombs & either visit Notre Dame Cathedral or Sainte Chapelle & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.THREE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FIVE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2i.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2i = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    -1



    [ log ]: p.LpStatus[problem.status] = 'Infeasible'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 0.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = -0.41666667
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 26.33333332
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- la Cathédrale Notre Dame de Paris --- le Jardin des tuileries --- La Place de la Concorde --- La Tour Eiffel


#### 2.j - If Mr. Doe wishes, at the same time, to take into account `Preference 2`, `Preference 3`, `Preference 4` and `Preference 5`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (must visit effiel tower and catacombs & either visit Notre Dame Cathedral or Sainte Chapelle & must visit tour montparnasse & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.THREE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FIVE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2j.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2j = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    -1



    [ log ]: p.LpStatus[problem.status] = 'Infeasible'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 0.0
    [ log ]: CN = 1.0
    [ log ]: CP = 0.0
    [ log ]: JT = -1.1666667
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 23.4999999
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- la Cathédrale Notre Dame de Paris --- La Place de la Concorde --- La Tour Eiffel --- la Tour Montparnasse


#### 2.k - If Mr. Doe wishes, at the same time, to take into account `Preference 1`, `Preference 2`, `Preference 4` and `Preference 5`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius & must visit effiel tower and catacombs & must visit tour montparnasse & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.ONE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FIVE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2k.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2k = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    1



    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 0.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 0.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 19.0
    [ log ]: You should visit total '5' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- les Catacombes --- La Tour Eiffel --- la Tour Montparnasse


#### 2.l - If Mr. Doe wishes, at the same time, to take into account `Preference 1`, `Preference 2`, `Preference 3`, `Preference 4` and `Preference 5`, which list(s) would you recommend to him ?


```python
verbose = False

# Create the variable to contain the problem data
problem = p.LpProblem(name="Paris Visit - Max. number of sites (within 1 km radius & must visit effiel tower and catacombs & either visit Notre Dame Cathedral or Sainte Chapelle & must visit tour montparnasse & must visit the Pompidou Center if he visits Louvre)", sense=p.const.LpMaximize)

# Create the variables
problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

# Create the objective function
problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info)

# Create the constraints
problem = add_generic_constraints(problem=problem, sites=sites, sites_info=sites_info)

# Add specific constraints
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.ONE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.TWO), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.THREE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FOUR), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)
problem = add_specific_constraints(problem=problem, preference=str(PREFERENCE.FIVE), distance_df=distance_df, sites=sites, sites_info=sites_info, verbose=verbose)

# The problem data is written to an .lp file
problem.writeLP(filename=join(data_dir, "to_visit_2l.lp"), writeSOS=1, mip=1, max_length=100)

# The problem is solved using PuLP's choice of solver
problem.solve(solver=LpSolverDefault(msg=show_solver_output))

# Output the status of the solution
to_visit_2l = display_solver_output(problem=problem, sites_info=sites_info) # save the result for later use
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'





    [AC, AT, BS, CA, CN, CP, JT, ML, MO, PC, SC, TE, TM]






    -1



    [ log ]: p.LpStatus[problem.status] = 'Infeasible'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.5
    [ log ]: CA = 0.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 0.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 0.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 1.0
    [ log ]: p.value(problem.objective) = 21.0
    [ log ]: You should visit total '4' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- La Tour Eiffel --- la Tour Montparnasse


#### 2.m - Is the solution `ListVisit1` different to these solutions founded above (with the combination of preferences) ?


```python
for i, x in enumerate(list('bcdefijkl')):
    print_log(f"Are 'ListVisit 1' and output of '2.{x}' same ? {is_identical(listvisit_1, globals()[f'to_visit_2{x}'])}")
```

    [ log ]: Are 'ListVisit 1' and output of '2.b' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.c' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.d' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.e' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.f' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.i' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.j' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.k' same ? False
    [ log ]: Are 'ListVisit 1' and output of '2.l' same ? False


##### 3. Let be:

- Ranking of the touristic sites obtained by observing only the `Duration` criterion (see the column “Duration” of the
Table above)
- Ranking of the touristic sites obtained by observing only the `Appreciations` criterion (see the column “Appreciations”
of the Table above)
- Ranking of the touristic sites obtained by observing only the `Price` criterion (see the column “Price” of the Table
above)

Are these rankings two rankings different ? To answer this question, you can use the `Kendall` or `Spearman` rank correlation coefficient.


```python
def create_objective_function(problem: p.LpProblem, sites: List, sites_info: List[SiteInfo], use_rating: bool = False) -> p.LpProblem:
    if use_rating:
        problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]), "Max. number of sites"
    else:
        problem += p.lpSum([globals()[f"{site}"] * 1 for i, site in enumerate(sites)]), "Max. number of sites"
    return problem

def get_basic_model(title: str, sites: List, sites_info: List[SiteInfo], use_rating: bool = False) -> p.LpProblem:
    """
    Returns a basic model with no constraints
    """
    # Create the variable to contain the problem data
    problem = p.LpProblem(name=f"Paris Visit - Max. number of sites {title}", sense=p.const.LpMaximize)

    # Create the variables
    problem = create_variables(problem=problem, sites=sites, sites_info=sites_info)

    # Create the objective function
    problem = create_objective_function(problem=problem, sites=sites, sites_info=sites_info, use_rating=use_rating)

    return problem

def solve_and_write_model(problem: p.LpProblem, filename: str, show_solver_output: bool = False) -> List[str]:
    """
    Solves the model and writes the output to a file
    """
    # The problem data is written to an .lp file
    problem.writeLP(filename=filename, writeSOS=1, mip=1, max_length=100)

    # The problem is solved using PuLP's choice of solver
    problem.solve(solver=LpSolverDefault(msg=show_solver_output))

    # Output the status of the solution
    return display_solver_output(problem=problem, sites_info=sites_info)

# 1. Ranking with duration constraint
problem = get_basic_model(title="(duration)", sites=sites, sites_info=sites_info, use_rating=False)
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].duration for i, site in enumerate(sites)]) <= 12, "Max. duration"
to_visit_3a = solve_and_write_model(problem=problem, filename=join(data_dir, "to_visit_3a.lp"), show_solver_output=show_solver_output)

# 2. Ranking with appreciation constraint
problem = get_basic_model(title="(appreciation)", sites=sites, sites_info=sites_info, use_rating=True)
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].rating for i, site in enumerate(sites)]) <= 27, "Max. appreciation" # at least 4 stars
to_visit_3b = solve_and_write_model(problem=problem, filename=join(data_dir, "to_visit_3b.lp"), show_solver_output=show_solver_output)

# 3. Ranking with price constraint
problem = get_basic_model(title="(price)", sites=sites, sites_info=sites_info, use_rating=False)
problem += p.lpSum([globals()[f"{site}"] * sites_info[i].price for i, site in enumerate(sites)]) <= 65, "Max. price"
to_visit_3c = solve_and_write_model(problem=problem, filename=join(data_dir, "to_visit_3c.lp"), show_solver_output=show_solver_output)
```

    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'
    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 0.0
    [ log ]: CP = 0.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 0.0
    [ log ]: PC = 1.0
    [ log ]: SC = 1.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 7.0
    [ log ]: You should visit total '7' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- le Jardin des tuileries --- La Place de la Concorde --- la Sainte Chapelle
    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'
    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 0.0
    [ log ]: CA = 0.0
    [ log ]: CN = 1.0
    [ log ]: CP = 1.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 1.0
    [ log ]: PC = 1.0
    [ log ]: SC = 0.0
    [ log ]: TE = 1.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 27.0
    [ log ]: You should visit total '8' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Cathédrale Notre Dame de Paris --- le Centre Pompido --- le Jardin des tuileries --- le Musée d’Orsay --- La Place de la Concorde --- La Tour Eiffel
    [ log ]: Creating variable for site = 'TE'
    [ log ]: Creating variable for site = 'ML'
    [ log ]: Creating variable for site = 'AT'
    [ log ]: Creating variable for site = 'MO'
    [ log ]: Creating variable for site = 'JT'
    [ log ]: Creating variable for site = 'CA'
    [ log ]: Creating variable for site = 'CP'
    [ log ]: Creating variable for site = 'CN'
    [ log ]: Creating variable for site = 'BS'
    [ log ]: Creating variable for site = 'SC'
    [ log ]: Creating variable for site = 'PC'
    [ log ]: Creating variable for site = 'TM'
    [ log ]: Creating variable for site = 'AC'
    [ log ]: p.LpStatus[problem.status] = 'Optimal'
    [ log ]: AC = 1.0
    [ log ]: AT = 1.0
    [ log ]: BS = 1.0
    [ log ]: CA = 1.0
    [ log ]: CN = 1.0
    [ log ]: CP = 1.0
    [ log ]: JT = 1.0
    [ log ]: ML = 0.0
    [ log ]: MO = 1.0
    [ log ]: PC = 1.0
    [ log ]: SC = 1.0
    [ log ]: TE = 0.0
    [ log ]: TM = 0.0
    [ log ]: p.value(problem.objective) = 10.0
    [ log ]: You should visit total '10' places. i.e:
    
    l’Avenue des Champs-Elysées --- l’Arc de triomphe --- la Basilique du Sacré-Coeur --- les Catacombes --- la Cathédrale Notre Dame de Paris --- le Centre Pompido --- le Jardin des tuileries --- le Musée d’Orsay --- La Place de la Concorde --- la Sainte Chapelle



```python
for i, x in enumerate(list('abc')):
    print_log(f"Total recommended places by '3.{x}': {len(globals()[f'to_visit_3{x}'])}")
    print_log(f"Output of '3.{x}': {globals()[f'to_visit_3{x}']}")
```

    [ log ]: Total recommended places by '3.a': 7
    [ log ]: Output of '3.a': ['l’Avenue des Champs-Elysées', 'l’Arc de triomphe', 'la Basilique du Sacré-Coeur', 'les Catacombes', 'le Jardin des tuileries', 'La Place de la Concorde', 'la Sainte Chapelle']
    [ log ]: Total recommended places by '3.b': 8
    [ log ]: Output of '3.b': ['l’Avenue des Champs-Elysées', 'l’Arc de triomphe', 'la Cathédrale Notre Dame de Paris', 'le Centre Pompido', 'le Jardin des tuileries', 'le Musée d’Orsay', 'La Place de la Concorde', 'La Tour Eiffel']
    [ log ]: Total recommended places by '3.c': 10
    [ log ]: Output of '3.c': ['l’Avenue des Champs-Elysées', 'l’Arc de triomphe', 'la Basilique du Sacré-Coeur', 'les Catacombes', 'la Cathédrale Notre Dame de Paris', 'le Centre Pompido', 'le Jardin des tuileries', 'le Musée d’Orsay', 'La Place de la Concorde', 'la Sainte Chapelle']


##### Correlation helper functions


```python
def correlation_preprocess(x: List, y: List) -> Tuple[List, List]:
    """
    All inputs to correlation method must be of the same size, i.e: x-size == y-size
    """
    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]

def list_name(x: str) -> dict:
    return {
        'a': 'duration',
        'b': 'appreciation',
        'c': 'price',
    }[x]

def kendall_tau_distance(x: List, y: List) -> float:
    """
    Calculates the Kendall Tau distance between two lists
    """
    x, y = correlation_preprocess(x, y)
    return 1 - kendalltau(x, y)[0]

def spearman_rho_distance(x: List, y: List) -> float:
    """
    Calculates the Spearman Rho distance between two lists
    """
    x, y = correlation_preprocess(x, y)
    return 1 - spearmanr(x, y)[0]
```

#### 1. Kendall correlation


```python
for i, j in combinations(list('abc'), 2):
    print_log(f"Kendall Tau distance between '{list_name(i)}' and '{list_name(j)}': {kendall_tau_distance(globals()[f'to_visit_3{i}'], globals()[f'to_visit_3{j}'])}")
```

    [ log ]: Kendall Tau distance between 'duration' and 'appreciation': 0.5714285714285714
    [ log ]: Kendall Tau distance between 'duration' and 'price': 0.2857142857142857
    [ log ]: Kendall Tau distance between 'appreciation' and 'price': 0.5714285714285714


    /opt/homebrew/Caskroom/mambaforge/base/envs/decision_modelling/lib/python3.10/site-packages/scipy/stats/_stats_py.py:110: RuntimeWarning: The input array could not be properly checked for nan values. nan values will be ignored.
      warnings.warn("The input array could not be properly "


#### 2. Spearman correlation


```python
for i, j in combinations(list('abc'), 2):
    print_log(f"Spearman Rho distance between '{list_name(i)}' and '{list_name(j)}': {spearman_rho_distance(globals()[f'to_visit_3{i}'], globals()[f'to_visit_3{j}'])}")
```

    [ log ]: Spearman Rho distance between 'duration' and 'appreciation': 0.4285714285714285
    [ log ]: Spearman Rho distance between 'duration' and 'price': 0.1785714285714285
    [ log ]: Spearman Rho distance between 'appreciation' and 'price': 0.5238095238095237

