## On Variational Generalization Bounds for Unsupervised Visual Recognition
This repository contains the code for instance discrimination experiments accompanying my project. Code is implemented in PyTorch with 
the basic framework borrowed from its original implementation [(arxiv)](https://arxiv.org/pdf/1805.01978.pdf).

## Usage
The implementation is easy to run and consists of all the divregence metrics provided in the manuscript. To train a ResNet-34 with Jensen-Shannon Divergence on the CIFAR10 dataset, execute the following-  
  `python cifar.py --nce-k 0 --nce-t 0.1 --lr 0.03 --phi JSD`

  - `--nce-k`- controls the number of negative samples. If nce-k sets to 0, the code also supports full softmax learning.
  - `--nce-t`- controls temperature of the distribution. 0.07-0.1 works well in practice.
  - `--nce-m`- stabilizes the learning process. A value of 0.5 works well in practice.
  - `--lr`- learning rate is initialized to 0.03, a bit smaller than standard supervised learning.
  - `--low-dim`- the embedding size is controlled by the parameter low-dim.
  - `--phi`- selects the divergence loss to be used during training (choose from JSD, KL, RKL, DV and InfoNCE).

Dataset instances are stored in the `cifar.py` and `mnist.py` files in `datasets` folder. These can be used to execute training on different datasets. During training, we monitor training loss, validation loss, top-1 and top-5 accuracies. Results are stored as a `log.pkl` file. 

## Citation
If you find my implementation useful then please cite the following-
```
@misc{1504project,
  title={On Variational Generalization Bounds for Unsupervised Visual Recognition},
  author={Suri, Karush and Haghifam, Mahdi and Khisti, Ashish},
  booktitle={Preprint},
  year={2020}
}
```

## Contact
For any questions, please feel free to reach 
```
Karush Suri: karush.suri@mail.utoronto.ca
```
