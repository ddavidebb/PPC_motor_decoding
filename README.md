# Motor decoding from the posterior parietal cortex using deep neural networks

The codes and trained models of this repository are associated to the journal paper _Motor decoding from the posterior parietal cortex using deep neural networks_ by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso (submitted to Neurocomputing, 2022). 
In that study, we compared different deep neural networks (fully-connected neural networks, convolutional neural networks, and recurrent neural networks) while decoding motor states from the neural activity of the posterior parietal cortex of macaques. 

Three different motor decoding tasks were addressed: reaching and reach-to-grasping decoding (this one with two different illumination conditions), involving the classification of different reaching end-points or different grip shapes. 
These motor tasks are referred as: _reaching_, _reach-to-grasping light_ (i.e., performed with good illumination conditions) and _reach-to-grasping dark_ (i.e., performed in darkness).
Four different monkeys performed the motor tasks (m1-m4).

The used neural networks are defined in the 'models.py' script. 
The optimal neural network designs were searched using Bayesian optimization, separately for the different neural networks and decoding problems. 
The optimal designs are contained in the pickle file 'optimal_hparams.pkl'.  

The decoding performance was investigated under different training conditions, also by artificially reducing the datasets, reflecting different recording scenarios. 
For brevity, the trained models contained in this repository ('trained_models' subfolder) refer only to the experiments performed while not artificially reducing datasets; 
furthermore, the uploaded trained models are only associated to the first cross-validation fold ('..._fold00.pth') or to the average across folds ('..._avg_folds.pth'). The latter was not used in the original publication, and it is reported here as a representative trained model (without uploading all fold-specific models). 
The 'main.py' script is a sample script showing how to use the trained networks with the optimal designs (as resulted from Bayesian optimization).


Please cite our manuscript if you use our code or results for your research.
