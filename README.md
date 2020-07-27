# Game Theoretic Analysis of Defence Algorithms Against Data Poisoning

These code replicates the experiments from the following paper:

> Game Theoretic Analysis of Defence Algorithms Against Data Poisoning
>
> Yifan Ou, Reza Samavi

Each .py file corresponds to one of the experiment from the paper:

> l2_online.py: replicates optimal attack with L2 defense in online learning
>
> LBGameExperiment.py: replicates optimal attack with LBDefense defense
>
> PCAGame.py: replicates PCA Game Analysis: PCA-Aware attack with PCA defense in online learning
>
> MultidayCurie.py: replicates multiday optimal sneaky attack with Curie defense
>
> mixDefense.py: replicates the algorithm to approximate for the NE defense strategy


> In the python scripts, specify the dataset (spamabse or mnist17) and the output directory for results.

Datasets used:

> Spambase Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
>
> MNIST Dataset: http://yann.lecun.com/exdb/mnist/ (loaded from tensorflow keras in the experiment code)

Dependencies:
- Numpy/Scipy/Scikit-learn/Pandas
- Tensorflow (tested on v1.1.0)
- Keras (tested on v2.0.4)
- Spacy (tested on v1.8.2)
- h5py (tested on v2.7.0)
- cvxpy (tested on 0.4.9)
- MATLAB/Gurobi
- Matplotlib/Seaborn (for visualizations)

If you have questions, please contact Yifan Ou (<ouyf@mcmaster.ca>) or Reza Samavi (<samavir@mcmaster.ca>).