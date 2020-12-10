# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

---

This is the origin pytorch implementation of Informer in the following paper: 
[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting]()

<p align="center">
<img src=".\img\informer.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> Informer.
</p>

## Requeirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.17.3
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.2.0

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

The ETDataset used in the paper can be download in [ETDataset](https://github.com/zhouhaoyi/ETDataset)
The data files should be put into `data/ETT/` folder. A demo of the ETT data is illustrated as the following figure.

<p align="center">
<img src="./img/data.png" height = "168" alt="" align=center />
<br><br>
<b>Figure 2.</b> A demo of the ETT data.
</p>

In the experiments, the input of each dataset is zero-mean normalized.

## Usage
Commands for training and testing the model with ProbSparse self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main_informer.py --model informer --data ETTh1 --attn prob

# ETTh2
python -u main_informer.py --model informer --data ETTh2 --attn prob

# ETTm1
python -u main_informer.py --model informer --data ETTm1 --attn prob
```

More parameter information can be founded in `main_informer.py`.


## Results

<p align="center">
<img src="./img/result_univariate.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 3.</b> Univariate forecasting results.
</p>

<p align="center">
<img src="./img/result_multivariate.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 4.</b> Multivariate forecasting results.
</p>


## Citation

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  pages     = {online},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```
