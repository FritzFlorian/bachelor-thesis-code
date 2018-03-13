# Batch Norm

Next run after 'normalize_probs'.
Main change is that it uses batch norm + elu instead of tanh.
It uses the same 5 maps as 'multiple_maps' and copies most other settings from this experiment.
The output cleaning from 'normalize_probs' is also used.


This will help to only compare the change in the neural network + output coversion (so the parts that predict
move probs and game outcome) and their effect on the training. To further help with analyzing
this run will also log both test and training error, to get an better idea about what the neural network is doing.


The experiment is still running.


