
import torch


def count_model_weights(model):
    count = 0
    for param in model.parameters():
        count += torch.prod(torch.tensor(param.shape))

    print(f"Number of weights {count:.3g}")


def update_parts_of_model(parent_model_state_dict, child_model, rank):
    child_model_dict = child_model.state_dict()

    # 1. filter out unnecessary keys
    # partial_parent_model_dict = {k: v for k, v in parent_model_dict.items() if k in child_model_dict}
    partial_parent_model_dict = {}
    for k, v in parent_model_state_dict.items():
        if k in child_model_dict:
            if rank == 0:
                print(f"{k} updated")
            v_child = child_model_dict[k]
            if v.shape == v_child.shape:
                partial_parent_model_dict[k] = v
            else:
                if rank == 0:
                    print(f"!!!! {k} param shows size mismatch between parent and child models!!!!")

        else:
            if rank == 0:
                print(f"!!!! {k} model param. is not presented in child model!!!!")

    # omitted_key = [k for k, v in parent_model_dict.items() if k not in child_model_dict]
    # print(f"Omittted keys: {omitted_key}")

    # 2. overwrite entries in the existing state dict
    child_model_dict.update(partial_parent_model_dict)
    child_model.load_state_dict(child_model_dict)

    return child_model


