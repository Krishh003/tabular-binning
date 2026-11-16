# Binning as a Pretext Task

Run from the repository root using Python 3.10 and PowerShell.

```powershell
python -m pip install -r requirements.txt
python tabular_binning.py --dataset california
python -m unittest test_tabular_binning
```

Kyungeun Lee and colleagues, ICML 2024. [Original paper](https://proceedings.mlr.press/v235/lee24v.html) · [Authors' code](https://github.com/kyungeun-lee/tabularbinning)

`--dataset churn` loads the authors' released BinRecon checkpoint and fits a frozen-encoder logistic head. The local asset files are from upstream commit `b036294963b5170a39d628483e856659cb12d867` (`dataset/Churn_Modelling.csv` and `results/CH/model_encoder_ep1000.pt`). The checkpoint uses the bank churn features and preprocessing.

`--dataset california` compares ValueRecon, BinXent, and BinRecon, including the masking variants from the slides. Quantile edges and scalers use training rows only. A validation split selects ridge regularization, and results are reported as test RMSE in units of $100,000. The fast run uses eight pretraining epochs and ten bins. Use `--epochs` and `--bins` to change those settings. The results are a single split and a frozen linear probe.

### Pretrained churn example

Download the [input CSV](https://github.com/kyungeun-lee/tabularbinning/blob/b036294963b5170a39d628483e856659cb12d867/dataset/Churn_Modelling.csv) and [checkpoint](https://github.com/kyungeun-lee/tabularbinning/blob/b036294963b5170a39d628483e856659cb12d867/results/CH/model_encoder_ep1000.pt) into `assets/tabularbinning`, then run `python tabular_binning.py --dataset churn`.
