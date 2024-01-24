import helper
import jsonlines
import numpy as np


def embeddings_fails(model, tokenizer):

    passed = 0
    killed = 0
    dead_man = []
    with jsonlines.open("/home/maxim/Code/Coding_Benchamark_SB/src/CLR10.jsonl") as reader:
        for obj in reader:
            print(obj["task_id"])
            try:
                helper.get_embedding(model, tokenizer, obj["completion"])
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


