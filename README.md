# Online Adaption to Label Distribution Shift

## 1. Dependencies and Data Preparation
### 1.1 Dependencies
```derivatives, tqdm, torch```
### 1.2 Data Preparation
The processed data are
```
./data_preparation/val_cifar10_resnet18.pt
./data_preparation/test_cifar10_resnet18.pt
```
We also provide the code to generate them from scratch:
```
cd data_preparation
python cifar10_prepare.py
cd -
```

## 2. Experiment with Simulated Shift
```
python simulation_main.py --shift_process ${shift_process} --algo {algo} --seed {seed}
```
- `shift_process` is `constant_shift`, `monotone_shift`, `period_shift_{T_p}` or `exp_period_shift_{k}` if we'd like to run the simulation with constant shift, monotone shift, periodic shift with period `T_p`, exponential periodic shift with exponential parameter `k`.
- `algo` is `const`, `opt_const`, `fth`, `ftfwh_{w}`, `ogd` if we choose the method as `base classifier`, `opt fixed classifier`, `Follow The HIstory`, `Follow The Fixed Window History` with window size `w`, `Online Gradient Descent`.
- When we choose online gradient descent, run the above command with `--conf_type zero_one` if using finite difference to approximate the gradient or run it with `--conf_type prob`