# Online Adaptation to Label Distribution Shift

## 1. Dependencies and Data Preparation
### 1.1 Dependencies
```derivatives, tqdm, torch```
### 1.2 Data Preparation for Simulated Shift
The processed data are
```
./data_preparation/val_cifar10_resnet18_cal.pt
./data_preparation/test_cifar10_resnet18_cal.pt
```
We also provide the code to generate them from scratch: first download the resnet18 pretrained on CIFAR10 from `https://drive.google.com/file/d/1gkPX4HKhwyXl820yUF6LEjXnhNzWTh99/view?usp=sharing`. Then run the following script:
```
cd data_preparation
python cifar10_prepare.py
cd -
```

### 1.3 Data Preparation for Shift in Arxiv
The processed data are
```
./data_preparation/val_arxiv_linear_cal.pt
./data_preparation/test_arxiv_linear_cal.pt
```
We also provide the code to generate them from scratch: download `arxiv-metadata-oai-snapshot.json` at `data_preparation/data/` from `https://www.kaggle.com/Cornell-University/arxiv`. Then run the following two jupyternotebooks sequentially:
```
Arxiv Data Process.ipynb
Arxiv Training.ipynb
```

## 2. Experiment with Simulated Shift
```
python simulation_main.py --shift_process ${shift_process} --algo {algo} --seed {seed}
```
- `shift_process` is `constant_shift`, `monotone_shift`, `period_shift_{T_p}` or `exp_period_shift_{k}` if we'd like to run the simulation with constant shift, monotone shift, periodic shift with period `T_p`, exponential periodic shift with exponential parameter `k`.
- `algo` is `const`, `opt_const`, `fth`, `ftfwh_{w}`, `ogd` if we choose the method as `base classifier`, `opt fixed classifier`, `Follow The HIstory`, `Follow The Fixed Window History` with window size `w`, `Online Gradient Descent`.
- When we choose online gradient descent, run the above command with `--conf_type zero_one` if using finite difference to approximate the gradient or run it with `--conf_type prob`


## 3. Experiment with Shift in Arxiv
```
python arxiv_main.py --algo {algo}
```
- `algo` is `const`, `opt_const`, `fth`, `ftfwh_{w}`, `ogd` if we choose the method as `base classifier`, `opt fixed classifier`, `Follow The HIstory`, `Follow The Fixed Window History` with window size `w`, `Online Gradient Descent`.
- When we choose online gradient descent, run the above command with `--conf_type zero_one` if using finite difference to approximate the gradient or run it with `--conf_type prob`


# Reference

This code corresponds to the following paper:

Ruihan Wu, Chuan Guo, Yi Su, and Kilian Q. Weinberger. **[Online Adaption to Label Distribution Shift](https://proceedings.neurips.cc/paper/2021/file/5e6bd7a6970cd4325e587f02667f7f73-Paper.pdf)**. NeurIPS 2021.

```
@inproceedings{
wu2021online,
title={Online Adaptation to Label Distribution Shift},
author={Ruihan Wu and Chuan Guo and Yi Su and Kilian Q Weinberger},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021}
}
```
