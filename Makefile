.PHONY: data figures

data: data/example_traj_s.csv data/example_traj_x.csv experiments/pws experiments/mlpws experiments/infonce experiments/doe

data/example_traj_s.csv data/example_traj_x.csv:
	python scripts/01-ar_dataset.py

experiments/pws:
	python scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 -o experiments/pws --estimator PWS

experiments/mlpws:
	python scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 -o experiments/mlpws --estimator ML-PWS

experiments/infonce:
	python scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 --backward_epochs 100 -o experiments/infonce --estimator InfoNCE

experiments/doe:
	python scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 --backward_epochs 100 -o experiments/doe --estimator DoE


figures: reports/figures/ml_pws_fig_1.png

reports/figures/ml_pws_fig_1.png: data
	python scripts/figure1.py
