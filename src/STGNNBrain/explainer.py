from torch_geometric.explain import Explainer, GNNExplainer
import torch
from models import STGCN, STChebNet, STGAT, STSGConv, MLP

models = {
        # 'Baseline LSTM': SimpleTimeSeriesLSTM(in_channels, hidden_channels, out_channels),
        'MLP': MLP(in_channels, [hidden_channels], out_channels),
        # 'STGCN': STGCN(in_channels, hidden_channels, out_channels),
        # 'STChebNet': STChebNet(in_channels, hidden_channels, out_channels),
    }

def load_model(model_name, fold, in_channels, hidden_channels, out_channels, device):
    model_path = f'./output/model/{model_name}_fold{fold}.pth'

    # Assuming 'MLP' is the model class and in_channels, hidden_channels, out_channels were defined earlier
    model = MLP(in_channels, [hidden_channels], out_channels).to(device) # Create an instance of your model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"Model '{model_name}' (fold {fold}) loaded successfully from {model_path}")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    ),
)

# This is how to do it in 
explanation = explainer(dataset[0].x, dataset[0].edge_index)
print(explanation.edge_mask)