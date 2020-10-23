# Comparing Trust Region Policy Optimization and Natural Policy Gradient in Reinforcement Learning

The core implementation is contained in the following files:

actor.py - implements the actor class

algorithms.py - implements TRPO and NPG, as well as target algorithms (MC returns and GAE)

experiment_class - implements an experiment class and wrapper class to run experiments with different parameters more smoothly

utils.py - implements useful functions, such as sampling memory or updating pytorch model parameters.



Minimal working examples can be found in

min_example_actor.py

demo.py

experiment_demo.py



For the results, we refer to the actual report

Report.pdf
