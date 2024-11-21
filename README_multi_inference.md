This readme is just for our internal purposes. The changes made to incorporate fractions to final layer prediction to MACE are done to only two model classes (MACE and ScaleShiftMACE) for now. Easily expandable to other models.
The similarity function takes the final layer embedding from the trained model to predict the fractions and instead of a delta function that the MACE currently uses for [1,0,0] for head1 and so on we predict a fraction set for each head and only final readout 
layer is affected. 
For now the training sets to compare the similarity function should come from the trained mace final layer descriptor. It can be changed easily to use SOAP as well. But changes must be done in forward function and not in MACEcalculator as 
they make its requires_grad true inside forward and not in batch preparation. Hence, we need the data that has gradients to get continuous flow through the fractions.
Also I assume u implement the similarity function outside MACE ( so as to has as little changes as we could make to the current code).
SO the implementation would be:
import torch
import torch.nn as nn
import torch.nn.functional as F
from mace.modules.multi_head_fraction import Attentionpoolingfraction
Similarity_fn = Attentionpoolingfraction(X1,X2,pool_type='avg',temperature=0.3) # X1,X2,.... are traning data descriptors
model.similarity_fn = Similarity_fn
#Adding this two lines to the calculation will add the delta or fractions to the prediction as we seek.
model.similarity_fn = None # This again disable the delta calculations. IT should be called similarity_fn always as it is hard coded now. We can make it like a similarity_fn_key but it is unnecessary and too much additions in the codes
