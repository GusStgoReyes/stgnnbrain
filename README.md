# STGNNBrain: Spatial-Temporal GNN for Brain data
### Predicting one person's brain state while caffeinated vs non-caffeinated

#### Exploring the notebook
1. Refer to `src/STGNNBrain/example_load_run_model.ipynb` for a walkthrough on the creation of the spatio-temporal dataset and running an example model. 

2. Refer to `src/STGNNBrain/models.py` and `src/STGNNBrain/train.py` to see some baseline spatial-temporal models and the code to train them. Currently under development. 

### How to setup?
We are using `uv` as version control. To setup the environment, run:

```bash
bash setup.sh
```

Furthermore, add the base path to the data in `config.json` and change the current user ID tag so that everything is running using the variables you defined. 

For the `src/STGNNBrain/example_load_run_model.ipynb` file, we recommend running it in google colab (as it has not been tested directly using the packaging in this repo). 