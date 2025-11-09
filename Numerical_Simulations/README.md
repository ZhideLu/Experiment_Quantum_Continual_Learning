# Description for this folder:

## functions:
We define some functions to be used in the subsequent simulation.

## dataset: 
For binary classification tasks, we provide three datasets in this folder, including the MNIST dataset (handwritten digits "0" and "9"), the Fashion MNIST dataset ("T-shirts" and "ankle boot"), and the medical dataset ("Hand" and "Breast").

For trinary classification tasks, we provide two datasets in this folder, including the MNIST dataset (handwritten digits "1", "4" and "8") and the Fashion MNIST dataset ("trouser", "sandal" and "bag").

## Quantum continual learning for binary classification tasks:

### demo_for_catastrophic_forgetting&quantum_continual_learning:
We provide numerical simulations to demonstrate catastrophic forgetting in quantum learning. To overcome this problem, we exploit the elastic weight consolidation strategy and show that quantum continual learning can be realized. The numerical results are exhibited in Supplementary Sec.IB (Supplementary Fig.S4, Supplementary Fig.S5, Supplementary Fig.S6).

### demo_for_FFNN/CNN_learning_engineered_quantum&classical_tasks:
We provide a numerical simulation of continually learning the quantum engineered task and the classical task using classical feedforward neural networks.
The numerical results is exhibited in Fig.4(d) in the main text.

## Quantum continual learning for trinary classification tasks:
We perform numerical simulations involving sequential learning of two trinary classification tasks,  involves classifying clothing images labeled as "trouser'', "sandal''and "bag'' from the Fashion-MNIST dataset, and classifying handwritten digits "1'', "4'' and "8'' from the MNIST dataset. The numerical results is exhibited in Supplementary Fig.S7 and Supplementary Fig.S8.

## Additonal numerical results:
We provide numerical results consisting of ten independent repetitions of the three-task continual learning experiment, each initialized with randomly chosen parameters. 
The results are shown in Supplementary Fig.S14. We also conduct an additional numerical simulation of  the three-task continual learning experiment using the different task order. 
The results shown in Supplementary Fig.S15. 