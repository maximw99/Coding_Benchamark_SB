import numpy as np
import torch
import numpy as np
import json
import sklearn.metrics as metrics


def get_pass(data):
    passes = []
    for i in data:
        passes.append(i["passed"])
    return passes


def read_samples(loc: str):
    samples = []
    with open(loc) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def compare_results(x, y):
    compared = []
    for i in range(0, 164):
        res = (x[i], y[i])
        compared.append(res)
    return compared


def get_entropies(embeddings):
    entropies = []
    for x in embeddings:
        entropy = kernel_entropy(x, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None))
        entropies.append(entropy)
    return entropies


def split_into_blocks(samples, block_size):
    return [samples[i:i+block_size] for i in range(0, len(samples), block_size)]


def get_embedding_bert(model, tokenizer, code: str):

    code_tokens = tokenizer.tokenize(code, max_length=510)
    tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.eos_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]

    return context_embeddings[0]


def kernel_entropy(Y, kernel):
    # author = Sebastian G. Gruber

    """
    Kernel entropy, estimator of -||P||_k^2 with Y_1, ..., Y_n ~ P via -1/n(n-1) \sum_{j=\=i} k( Y_i, Y_j )
    Rows of Y are instances, columns are features
    """
    with torch.no_grad():
      YY = kernel(Y, Y)
      # length of Y
      n = YY.shape[0]
      # print(YY)
      return (YY.diagonal().sum() - YY.sum())/(n*(n-1))
    
    def get_block(data, start, end):
        block = []
        for i in range(start, end):
            block.append(data[i])
        return block


def form_emb(blocks, model, tokenizer):
    l = 0
    embeddings = []
    for block in blocks:
        emb_block = []
        for code in block:
            try:
                embedding = get_embedding_bert(model, tokenizer, code["completion"])
            except:
                continue
            emb_block.append(embedding)
        embeddings_mean = [i.mean(0).detach().numpy() for i in emb_block]
        embeddings_mean = np.array(embeddings_mean)
        embeddings.append(embeddings_mean)
        print(l)
        l = l+1
            
    return embeddings