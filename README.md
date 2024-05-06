# Annealed adaptive importance sampling method in PINNs for solving high dimensional partial differential equations 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)]([https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/))
[![Pytorch 1.13](https://img.shields.io/badge/pytorch-1.13-blue.svg)](https://pytorch.org/)

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2209.14977-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2209.14977&#41;)

## Summary
An adaptive sampling method called Annealed Adaptive Importance Sampling(AAIS) method is used in PINNs for solving high dimensional partial differential equations. The method could be used in many PDE works but here we may focus on the singular Poisson problems with multi peaks.


## Training
Training model:

`--pde` includes various type of PDEs whose name is in `configs.yml` and `libs`. If you want to test other PDEs please refer to any PDE files in `libs` like `Poisson.py`, added PDEs should follow the structure of our PDE Class Object structures.

`--strategy` includes four type adaptive resampling strategy: `Uni_resample`, `RAD_resample`, `AAIS_g_resample`, `AAIS_t_resample`

other args could be seen in `main.py` or refer to `--help` in command.

All different models' settings can be found in `configs.yml`, you could change some settings there.

For example if you want to test 2D Poisson problems with only one peak, just go with:
```bash
python main.py --cuda_dev 0 --pde Poisson2D1Peak --strategy AAIS_t_resample --num_sample 1500 500 500 --lr 1e-4 3e-1 --epoch 500 2000 500 2000 --num_search 60000 --dirname 'Your_File_Name' --max_iter 20
```

the code could create a directory `/Project_Path/results/Poisson2D1Peak/Your_File_Name`, the loss and sampling results can be seen in `/img`, solutions and absolute errors are plotted in '/test'. Running logs and settings are saved in `/logger.log` and `/inform.txt`.

By The Way, if you want to rerun the high dimensional Poisson problems `PoissonNDpeaks`, firstly the dimensions could be changed in `configs.yml`, just change `input_size` of `PoissonNDpeaks`. If you want to add more centers or change the scaling parameters, please go to '/libs/Poisson.py' and at line 670-675. It is recommended to change the coordinates of centers in the first two dimensions since the plot would project the solution into $x_1x_2$-plane.

If you have some problems, please report.

