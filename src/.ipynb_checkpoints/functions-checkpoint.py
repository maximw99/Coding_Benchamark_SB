import torch
import sklearn.metrics as metrics
import jsonlines
import numpy as np


def get_embedding(model, tokenizer, code: str):

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


def form_emb(data, model, tokenizer):
    embeddings = []
    for line in data:
        print(line["task_id"])
        try:
            embedding = get_embedding(model, tokenizer, line["completion"])
            fix = embedding.detach()
            embeddings.append(fix)
        except:
            continue
            

    embeddings_mean = [i.mean(0).detach().numpy() for i in embeddings]
    embeddings_mean = np.array(embeddings_mean)
            
    return embeddings_mean


def embeddings_fails(model, tokenizer):

    passed = 0
    killed = 0
    dead_man = []
    with jsonlines.open("/home/maxim/Code/Coding_Benchamark_SB/src/CLR10.jsonl") as reader:
        for obj in reader:
            print(obj["task_id"])
            try:
                get_embedding(model, tokenizer, obj["completion"])
                passed += 1
            except:
                killed += 1
        print("pass ", passed, "killed: ", killed)


def get_length(test: int):

    if test == 1:

        total = 0
        over_512 = []
        under_512 = []
        over_1000 = []
        longest = {"completion": "ja"}

        with jsonlines.open("/home/maxim/Code/Coding_Benchamark_SB/src/CLR10.jsonl") as reader:

            for obj in reader:
                # print("id: ", obj["task_id"], "length: ", len(obj["completion"]))
                total += 1
                if len(obj["completion"]) > len(longest["completion"]):
                    longest = obj
                if len(obj["completion"]) > 512:
                    over_512.append(obj)
                    if len(obj["completion"]) > 1000:
                        over_1000.append(obj)
                else:
                    under_512.append(obj)
                        

            print(total, len(under_512), len(over_512))

            for item in over_1000:
                print(item["task_id"], len(item["completion"]))

            print("longest: ", longest["task_id"])


    else:

        total = 0
        over_512 = 0
        under_512 = 0
        over_512_tot = 0
        under_512_tot = 0
        counter = 0

        with jsonlines.open("src/CLR1.jsonl:Zone.Identifier") as reader:

            for obj in reader:
                if counter == 10:
                    counter = 0
                    if under_512 > over_512:
                        under_512_tot += 1
                    else:
                        over_512_tot += 1
                    over_512 = 0
                    under_512 = 0
                # print(len(obj["completion"]))
                total += 1
                if len(obj["completion"]) > 512:
                    over_512 += 1
                else:
                    under_512 += 1
                counter += 1
                        

            print(total, under_512_tot, over_512_tot)


