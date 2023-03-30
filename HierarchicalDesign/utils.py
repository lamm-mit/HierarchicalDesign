#########################################################
# Utility functions
#########################################################

def count_parameters (imagen):

    pytorch_total_params = sum(p.numel() for p in imagen.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in imagen.parameters() if p.requires_grad)

    print ("----------------------------------------------------------------------------------------------------")
    print ("Total parameters: ", pytorch_total_params," trainable parameters: ", pytorch_total_params_trainable)
    print ("----------------------------------------------------------------------------------------------------")
    return

