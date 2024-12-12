# STGNNBrain: Spatial-Temporal GNN for Brain data
### Predicting Brain-State Using Spatio-Temporal GNNs
We use the MyConnectome dataset, composed of thirty-two 10-minute fMRI scans while a participant was caffeinated and fourty 10-minute fMRI scans while the same participant was fasted. Our task is to leverage Spatiotemporal patterns of brain activity using Graph Neural Networks to classify the state of the participant. 

#### Graph definition
We define two types of graphs: (1) The brain ROIs are the nodes and (2) The time points are the nodes. 
![Overview of the graphs created](figures/graphs.png)

#### Architectures explored

![Overview of the model architectures](figures/model_architecture.png)

### How to setup?

Not working currently....
We are using `uv` as version control. To setup the environment, run:

```bash
bash setup.sh
```

Furthermore, add the base path to the data in `config.json` and change the current user ID tag so that everything is running using the variables you defined. 

For the `src/STGNNBrain/example_load_run_model.ipynb` file, we recommend running it in google colab (as it has not been tested directly using the packaging in this repo). 