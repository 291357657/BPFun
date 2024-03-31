import numpy as np
import torch
from multiprocessing import cpu_count
torch.set_num_threads(cpu_count())



model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t48_15B_UR50D")

def infer(seq):
    seq = seq[0:32]
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("tmp", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()
    # print(token_representations.shape)
    token_representations = token_representations[0][1:-1,:]
    # (7, 1280)
    res = np.zeros((32,1280),dtype='float32')
    for i,x in enumerate(token_representations):
        res[i]= x
    return res

def enc(seqs):
    num = 1
    data = []
    for i in seqs:
        print(num)
        tmp = infer(i)
        data.append(tmp)
        num = num + 1
    return data

# seqs = ['GIPCGESCVFIPCITGAIGCSCKSKVCYRN','YSPFSSFPR']
# for i in seqs:
#     tmp = infer(i)