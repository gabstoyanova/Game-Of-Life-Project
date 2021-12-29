# Using reinforcement learning to sustain high-density populations in Conway’s Game of Life

In this project we introduce a reinforcement learning agent in Conway’s Game of Life with the objective to sustain long-living, high-density populations. We experiment by proposing a subdivided variant of the environment, alongside the standard tabular method problem definition. For our purposes we use Q-Learning, a reinforcement learning algorithm that can be applied to sequential tasks.
We find that the agent creates highly correlated, static structures of living cells. We observe fast learning rate for small grid sizes. 
We also approach the task with a deep Q-network and observe similar results.

# Approaches

We use the following three approaches for our task:
 -  Q-learning on a small grid
 -  Q-learning on a large subdivided grid
 -  Deep Convolutional Q-Learning
 
 A more detailed description of the methodology and analysis of the obtained results can be found in our [paper](project_paper.pdf).
