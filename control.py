"""Matched vanilla-M3 control for the Perturb-seq experiment.""" 
 
import torch.nn as nn 
 
 
def build_control_model( 
    base_model, 
    transformed_dataset, 
    train_indices, 
    validation_indices, 
    n_cell_lines, 
): 
    """Keep vanilla M3 unchanged except for matched RNA mean dropout.""" 
    dropout = base_model.encoder.encoders_mean[0][3] 
    if not isinstance(dropout, nn.Dropout): 
        raise TypeError("expected RNA mean encoder dropout at encoders_mean[0][3]") 
 
    dropout.p = 0.5 
    return base_model 
