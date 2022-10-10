### Use case - Miracle Worker

Let’s try to formalize an use-case and carry it forward throughout the article. Suppose you are a magical healer and you goal is to
heal anyone who asks for help. The more you are able to heal someone, the better. Your secret behind the healing is 2 medicines, each of
which uses special herbs. To create one unit of medicine 1, you need 3 units of herb A and 2 units of herb B. Similarly, to create one unit
of medicine 2, you need 4 and 1 units of herb A and B respectively. Now medicine 1 can heal a person by 25 unit of health (whatever it
is) and medicine 2 by 20 units. To complicate things further, you only have 25 and 10 units of herb A and B at your disposal. Now the
question is, how many of each medicine will you create to maximize the health of the next person who walks in ?

#### Modeling the problem

First let’s try to identify the objective (what we want to do and how) and constraint (the bounding functions) of the stated problem.
As it’s clear from the problem, we want to increase the health by as many units as possible. And medicines are the only thing which
can help us with it. What we are unsure of, is the amount to each medicines to create. Going by a mathematician’s logic, lets say we
create x units of medicine 1 and y units of medicine 2. Then the total health restored can be given by,

$$
    25 * x + 20 * y = Health\ Restored
$$

where

$$
    x = Units\ of\ medicine\ 1\ created;\ 
    y = Units\ of\ medicine\ 2\ created
$$

This is the objective function, which we want to maximize. Now both the medicines are dependent on the herbs which we have in
finite quantity. Let’s understand the constraints. If we create x and y units of medicine 1 and 2,


- We use $ 3x + 4y $ units of herb A. But we only have 25 units of it, hence the constraint is, our total usage of herb A should not exceed 25, denoted by,

$$
3x + 4y ≤ 25
$$

