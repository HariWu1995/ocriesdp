from copy import deepcopy
from collections import Counter, OrderedDict

import re

import numpy as np
import torch
import torch.nn.functional as F

from nltk.metrics.distance import edit_distance


def refine_ind_text(key, val, idx):
    if "price" in key:
        val = ("-" if val.startswith("-") and idx == 0 else "") + re.sub(r"[^0-9]+", "", val)

    elif "cnt" in key:
        val = ("-" if val.startswith("-") and idx == 0 else "") + re.sub(r"[^0-9^\.]+", "", val)

    return val


def refine_parse_receipt(parse_orig):
    parse = parse_orig["parse"]
    parse_refined = deepcopy(parse_orig)
    if "prob" in parse_orig:
        prob = parse_orig["prob"]
        flag_prob_exist = True
    else:
        flag_prob_exist = False
        prob = None

    if "lang" not in parse_orig or parse_orig["lang"] == "ind":
        refine_text = refine_ind_text
    else:
        raise NotImplementedError

    for i, group_parse in enumerate(parse):
        # To handle the edge case: (menu.unitprice without menu.price & menu.itemsubtotal)
        keys = group_parse.keys()
        if ("menu.unitprice"        in keys
        and "menu.itemsubtotal" not in keys
        and "menu.price"        not in keys):
            parse[i]["menu.price"] = parse[i].pop("menu.unitprice")
            if flag_prob_exist:
                if prob[i]:
                    prob[i]["menu.price"] = prob[i].pop("menu.unitprice")

        for key in group_parse:
            if "unitprice" in key and len(group_parse[key]) > 1:
                group_parse[key] = ["".join(group_parse[key])]

            for j, val in enumerate(group_parse[key]):
                val = refine_text(key, val, j)
                parse[i][key][j] = val

    # Return
    parse_refined["parse"] = parse
    parse_refined["prob"] = prob
    return parse_refined


def group_counter(group):
    txt = []
    for key in group:
        txt = txt + group[key]
    return Counter(sorted("".join(txt).replace(" ", "")))


def get_group_compare_score(gr1, gr2):
    score = 0
    # if gr1.keys() == gr2.keys():
    #    score += 100

    for key in list(set(list(gr1.keys()) + list(gr2.keys()))):
        if key in gr1 and key in gr2:
            if gr1[key] == gr2[key]:
                score += 50
            elif gr1[key] in gr2[key] or gr2[key] in gr1[key]:
                score += 30
            else:
                score += sum((Counter("".join(gr1[key])) & Counter("".join(gr2[key]))).values())

    score += sum((group_counter(gr1) & group_counter(gr2)).values())
    return score


def get_int_number(string):
    num = re.sub(r"[^0-9]", "", string)
    return int(num) if len(num) > 0 else 0


def get_price_from_parse(parse):
    price = 0
    total = 0
    for gr in parse:
        for key in gr.keys():
            val = get_int_number(gr[key][0].split(" ")[0])
            if "menu.price" == key or "menu.sub_price" == key:
                price += val
            elif "menu.discountprice" == key:
                price -= val
            elif "total.total_price" == key:
                total += val

    return price, total


def get_init_stats_receipt():
    return {
        "label_stats": dict(),
        "group_stats": [0, 0, 0],
        "receipt_cnt": 0,
        "price_count_cnt": 0,
        "prices_cnt": 0,
        "receipt_total": 0,
    }


def get_statistics_receipt(gt, pr, stats, receipt_refine=False,
                                          receipt_edit_distance=False, return_refined_parses=False,):

    gt = refine_parse_receipt(gt)["parse"] if receipt_refine else gt["parse"]
    pr = refine_parse_receipt(pr)["parse"] if receipt_refine else pr["parse"]
    label_stats = stats["label_stats"]
    group_stats = stats["group_stats"]

    mat = np.zeros((len(gt), len(pr)), dtype=np.int)
    for i, gr1 in enumerate(gt):
        for j, gr2 in enumerate(pr):
            mat[i][j] = get_group_compare_score(gr1, gr2)

    pairs = []
    gt_paired = []
    pr_paired = []
    for _ in range(min(len(gt), len(pr))):
        if np.max(mat) == 0:
            break

        x = np.argmax(mat)
        y = int(x / len(pr))
        x = int(x % len(pr))
        mat[y, :] = 0
        mat[:, x] = 0
        pairs.append((y, x))
        gt_paired.append(y)
        pr_paired.append(x)

    for i in range(len(gt)):
        stat = dict()
        for key in gt[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][1] += stat[key]

    for i in range(len(pr)):
        stat = dict()
        for key in pr[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][2] += stat[key]

    group_stat = [0, len(gt), len(pr)]
    price_count_check = True
    for i, j in pairs:
        # For each group,
        stat = dict()
        for key in set(list(gt[i].keys()) + list(pr[j].keys())):
            if key not in stat:
                stat[key] = 0

        cnt = 0
        for key in gt[i]:
            pr_val = (
                [norm_receipt(val, key) for val in pr[j][key]] if key in pr[j] else []
            )
            gt_val = [norm_receipt(val, key) for val in gt[i][key]]

            # if key in pr[j] and pr[j][key] == gt[i][key]:
            #    stat[key] += 1
            #    cnt += 1
            if pr_val == gt_val:
                stat[key] += 1
                cnt += 1

            elif (
                "nm" in key
                and receipt_edit_distance
                and len(pr_val) > 0
                and (
                    edit_distance(pr_val[0], gt_val[0]) <= 2
                    or edit_distance(pr_val[0], gt_val[0]) / len(pr_val[0]) <= 0.4
                )
            ):
                stat[key] += 1
                cnt += 1

            elif "price" in key or "cnt" in key:
                price_count_check = False

        if cnt == len(gt[i]):
            group_stat[0] += 1

        # Stat Update
        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][0] += stat[key]

    for i, gr in enumerate(gt):
        if i not in gt_paired:
            for key in gr:
                if "price" in key or "cnt" in key:
                    price_count_check = False
    for i, gr in enumerate(pr):
        if i not in pr_paired:
            for key in gr:
                if "price" in key or "cnt" in key:
                    price_count_check = False

    stats["price_count_cnt"] += price_count_check
    for k in range(3):
        group_stats[k] += group_stat[k]

    gt_prices = get_price_from_parse(gt)
    pr_prices = get_price_from_parse(pr)  # return price, total
    if gt_prices == pr_prices and gt_prices[1] != 0:  # total_price != 0
        stats["prices_cnt"] += 1

    item_correct = group_stat[0] == group_stat[1] and group_stat[1] == group_stat[2]
    stats["receipt_cnt"] += item_correct
    stats["receipt_total"] += 1

    label_stats["total"] = [0, 0, 0]
    for key in sorted(label_stats):
        if key not in ["total"]:
            for i in range(3):
                label_stats["total"][i] += label_stats[key][i]

    stats["label_stats"] = label_stats
    stats["group_stats"] = group_stats
    if return_refined_parses:
        return stats, item_correct, gt, pr
    else:
        return stats, item_correct


def norm_receipt(val, key):
    val = val.replace(" ", "")
    return val


def get_scores(tp, fp, fn):
    pr = tp / (tp + fp) if (tp + fp) != 0 else 0
    re = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * pr * re) / (pr + re) if (pr + re) != 0 else 0
    return pr, re, f1


def summary_receipt(path, stats, print_screen=False):
    st = stats["label_stats"]
    st["Group_accuracy"] = stats["group_stats"]

    s = dict()
    for key in st:
        tp = st[key][0]
        fp = st[key][2] - tp
        fn = st[key][1] - tp
        s[key] = (tp, fp, fn) + get_scores(tp, fp, fn)

    c = {
          "main_key": "receipt",
            "prices": stats["prices_cnt"],
         "price/cnt": stats["price_count_cnt"],
           "receipt": stats["receipt_cnt"],
             "total": stats["receipt_total"],
    }

    if print_screen:
        other_fields = ["total", "Group_accuracy"]
        header = ("field", "tp", "fp", "fn", "prec", "rec", "f1")
        print("%25s\t%6s\t%6s\t%6s\t%6s\t%6s\t%6s" % header)
        print("------------------------------------------------------------------------------")
        for key in sorted(s):
            if key not in other_fields:
                print("%-25s\t%6d\t%6d\t%6d\t%6.3f\t%6.3f\t%6.3f" % (key, s[key][0], s[key][1], s[key][2],
                                                                          s[key][3], s[key][4], s[key][5],))
        print("------------------------------------------------------------------------------")
        for key in other_fields:
            print("%-25s\t%6d\t%6d\t%6d\t%6.3f\t%6.3f\t%6.3f" % (key, s[key][0], s[key][1], s[key][2],
                                                                      s[key][3], s[key][4], s[key][5],))

        for key in c:
            if key not in ["total", "main_key"]:
                print(" - %10s accuracy :  %.4f (%d/%d)" % (key, c[key] / c["total"], c[key], c["total"]))

    return s, c


"""
Graph Decoder
"""

def pred_label(task, score, method, n_fields, l_units, return_tensor=False):
    bS, n_relation_doubled, max_nr, max_nc = score.shape
    n_relation = n_relation_doubled // 2

    # 1. Construct probability
    pr_label_tensors = []
    pr_label_tensor = None
    for i_rel in range(n_relation):
        pr_label_tensor = gen_pr_label_tensor(
            task, i_rel, n_fields, method[i_rel], score, pr_label_tensor
        )
        pr_label_tensors.append(pr_label_tensor)

    # 2. Concant rel-s, rel-g tensors.
    pr_label_tensors = torch.cat(pr_label_tensors, dim=1)

    if return_tensor:
        return pr_label_tensor
    else:
        pr_labels = pr_label_tensors_to_list(pr_label_tensors, n_fields, l_units)
        return pr_labels


def select_single_most_probable_node(pr_label_tensor, prob):
    # 1. Find probability of each edges selected from the previous step.
    edge_existence_prob = prob[:, 1:2, :, :]  #
    pr_label_arc_pval_tensor = edge_existence_prob * pr_label_tensor.type(torch.float)

    # 2. Found best edge ids at each row
    best_idx = pr_label_arc_pval_tensor.argmax(dim=3, keepdim=True)  # [batch, nr]

    # 3. Trim other edges
    pr_label_tensor = torch.zeros_like(pr_label_tensor).scatter(3, best_idx, 1).type(torch.long)

    # 4. Trim spurious edges.
    pr_label_tensor = pr_label_tensor * pr_label_tensor

    return pr_label_tensor


def gen_pr_label_tensor(task, i_rel, n_fields, method, score, pr_label_rel_s=None):

    # 1. Predict the existence of edge without any constraint.
    # This is equivalnt to use 50% threshold.
    st = 2 * i_rel  # 2 since the relation is either 0 or 1.
    ed = st + 2
    pr_label_tensor0 = score[:, st:ed, :, :].argmax(dim=1, keepdim=True)
    prob = F.softmax(score[:, st:ed, :, :], dim=1)

    if method == "no_constraint":
        # 2.1 No constraint.
        pr_label_tensor = pr_label_tensor0

    elif method == "zero_fields":
        # 2.2 Remove any edge start from "field-nodes".
        pr_label_tensor = pr_label_tensor0
        pr_label_tensor[:, :, 0:n_fields, :] = 0

    elif method == "force_single_tail_node":
        # 2.3 Each head node has only a single tail node (which is not true on namecard data).
        pr_label_tensor = select_single_most_probable_node(pr_label_tensor0, prob)

    elif method == "force_single_tail_node_but_allow_multiple_seeds":
        # 2.4 Each head node has only a single tail node (which is not true on namecard data).
        # But, field-node can have multiple tail nodes (multiple seeds).
        pr_label_tensor = select_single_most_probable_node(pr_label_tensor0, prob)
        pr_label_tensor[:, :, 0:n_fields, :] = pr_label_tensor0[:, :, 0:n_fields, :]

    elif method == "avoid_collapse":
        pr_label_tensor = select_single_most_probable_node(pr_label_tensor0, prob)
        pr_label_tensor[:, :, 0:n_fields, :] = pr_label_tensor0[:, :, 0:n_fields, :]

        pr_label_sub = pr_label_tensor[:, :, n_fields:, :]
        prob_sub = prob[:, :, n_fields:, :]

        max_iter = 20
        pr_label_sub = avoid_tail_collision(max_iter, pr_label_sub, prob_sub)
        pr_label_tensor[:, :, n_fields:, :] = pr_label_sub

    elif method == "tca_rel_s":
        assert i_rel == 0
        # 2.5.1. Start from the result of "force_single_tail_node_but_allow_multiple_seeds" method.
        pr_label_tensor = select_single_most_probable_node(pr_label_tensor0, prob)
        pr_label_tensor[:, :, 0:n_fields, :] = pr_label_tensor0[:, :, 0:n_fields, :]

        # 2.5.2 Apply TCA algorithm to non-field nodes.
        pr_label_sub = pr_label_tensor[:, :, n_fields:, :]
        prob_sub = prob[:, :, n_fields:, :]

        max_iter = 20
        pr_label_sub = avoid_tail_collision(max_iter, pr_label_sub, prob_sub, allow_multiple_outgoing=False)
        pr_label_tensor[:, :, n_fields:, :] = pr_label_sub

        # between field-node and start node
        # 2.5.3 Apply TCA algorithm to field nodes.
        # Apply TCA only a sinlge time.
        _st = 0
        _ed = n_fields

        pr_label_sub2 = pr_label_tensor[:, :, _st:_ed, :]
        prob_sub2 = prob[:, :, _st:_ed, :]

        max_iter = 1  # Only trimming without generating new tail nodes.
        pr_label_sub2 = avoid_tail_collision(max_iter, pr_label_sub2, prob_sub2, allow_multiple_outgoing=True)
        pr_label_tensor[:, :, _st:_ed, :] = pr_label_sub2

    elif method == "tca_rel_g":
        assert pr_label_rel_s is not None
        assert i_rel == 1
        # 2.6.1 Start from the result of "no_constraint".
        pr_label_tensor = pr_label_tensor0
        pr_label_tensor[:, :, 0:n_fields, :] = 0

        assert pr_label_rel_s.shape == pr_label_tensor.shape

        # 2.6.2 Apply TCA to field-nodes but only a single time.
        pr_label_sub2 = pr_label_tensor[:, :, n_fields:, :]
        prob_sub2 = prob[:, :, n_fields:, :]

        max_iter = 1  # Only trimming without generating new tail nodes.
        pr_label_sub2 = avoid_tail_collision(max_iter, pr_label_sub2, prob_sub2, allow_multiple_outgoing=True)
        pr_label_tensor[:, :, n_fields:, :] = pr_label_sub2

        pr_label_tensor = trim_non_seed_nodes(n_fields, pr_label_rel_s, pr_label_tensor)

    else:
        raise NotImplementedError

    return pr_label_tensor


def pr_label_tensors_to_list(pr_label_tensors, n_fields, l_units):
    pr_label_arrs = pr_label_tensors.cpu().numpy()
    new_pr_labels = []
    for b, l_unit in enumerate(l_units):
        nr1 = n_fields + l_unit
        nc1 = l_unit
        new_pr_label = []
        for pr_label_arr1 in pr_label_arrs[b]:
            # relation level
            new_pr_label.append(pr_label_arr1[:nr1, :nc1].tolist())
        new_pr_labels.append(new_pr_label)
    return new_pr_labels


def avoid_tail_collision(max_iter, pr_label_sub, prob_sub, allow_multiple_outgoing=False):
    for i in range(max_iter):
        pr_label_sub, _ids = remove_multiple_incoming(pr_label_sub, prob_sub, allow_multiple_outgoing)
        if not _ids.sum():
            break
    if i == max_iter - 1 and max_iter > 1:
        print(f"Max iteration {max_iter} reached in TCA algorithm.")

    return pr_label_sub


def trim_non_seed_nodes(n_fields, pr_label_rels, pr_label_tensor):
    # collect col_ids of start nodes
    mask_rels = torch.zeros_like(pr_label_rels)
    ids = pr_label_rels[:, :n_fields, :, :].sum(dim=2, keepdim=True).nonzero()
    for id in ids:
        b, _, _ir, ic = id
        mask_rels[b, :, n_fields + ic, :] = 1

    return mask_rels * pr_label_tensor


def remove_multiple_incoming(pr_label_sub, prob_sub, allow_multiple_outgoing=False):

    # 1. Find tail node ids with multiple incoming edges.
    ids = get_tail_node_ids_having_multiple_incoming(pr_label_sub)
    for id in ids:
        b, _, _ir, ic = id
    
        # 2. Find head node ides connected to target tail id.
        row_ids = pr_label_sub[b, 0, :, ic].nonzero()
        assert len(row_ids) > 1

        # 3. Find the best head node id.
        _p = -99  # probability
        best_i_row = -1
        for i_row, row_id in enumerate(row_ids):
            current_p = prob_sub[b, 1, row_id[0], ic]
            if _p <= prob_sub[b, 1, row_id[0], ic]:
                best_i_row = i_row
                _p = current_p

        for i_row, row_id in enumerate(row_ids):
            if i_row == best_i_row:
                continue

            # 4. Trim other head nodes.
            pr_label_sub[b, 0, row_id[0], ic] = 0

            # 5. Generate new tail node for the tail-trimmed-head node.
            if not allow_multiple_outgoing:

                # 5.1. Find candidate col-ids
                _ps, _col_ids = prob_sub[b, 1, row_id[0], :].topk(4)
                if ic not in _col_ids:
                    # Extend topk range to include ic
                    _ps, _col_ids = prob_sub[b, 1, row_id[0], :].topk(max(_col_ids))
                current_rank = (_col_ids == ic).nonzero()[0][0]
                assert _ps[current_rank] >= 0.5
                if current_rank < 3:  # top k above
                    if _ps[current_rank + 1] >= 0.5:
                        assert pr_label_sub[b, 0, row_id[0], _col_ids[current_rank + 1]] == 0

                        # Connect the new tail node.
                        pr_label_sub[b, 0, row_id[0], _col_ids[current_rank + 1]] == 1
            else:
                # don't do anything.
                pass

    return pr_label_sub, ids


def get_tail_node_ids_having_multiple_incoming(pr_label_sub):
    # in: [B, 1, nr, nc], nr = nc
    # out: [ [b, 0, 0, ic] ]
    ids = (pr_label_sub.sum(dim=2, keepdim=True) > 1).nonzero()
    return ids


def get_key_from_single_key_dict(f_parse1):
    target_field_list = list(f_parse1.keys())
    assert len(target_field_list) == 1
    field_of_target = target_field_list[0]

    return field_of_target


def gen_text_field_labelsl(text_units, f_parse_box_ids):
    text_field_labels = []
    for b, (text_unit, f_parse_box_id) in enumerate(zip(text_units, f_parse_box_ids)):
        l_text_units = len(text_unit)
        text_field_label = ["O"] * l_text_units
        for f_parse_box_id1 in f_parse_box_id:
            field = get_key_from_single_key_dict(f_parse_box_id1)
            ids = f_parse_box_id1[field]
            for id in ids:
                text_field_label[id] = field

        text_field_labels.append(text_field_label)

    return text_field_labels


def gen_parses(task, fields, field_rs, text_units, label_units,
                header_toks, l_max_gen, max_info_depth, strict, token_lv_boxing, backbone_name,):
    """
    f_parses: a list of serialized fields
    g_parses: a list of grouped serialized fields
    """
    f_label_idx = 0
    g_label_idx = 1

    # 1. Generate serialized individual fields
    label_fs = [label[f_label_idx] for label in label_units]
    f_parses, f_parse_box_ids, f_parse_head_ids = gen_f_parses(task, fields, text_units, label_fs, l_max_gen)

    # have to return f_parse_box_ids
    # 1.1 Tag each text_unit as its field type
    text_unit_field_labels = gen_text_field_labelsl(text_units, f_parse_box_ids)

    # 1.2 Detokenization
    if token_lv_boxing:
        f_parses = detokenize_f_parse(f_parses, f_parse_box_ids, header_toks, backbone_name)

    # 2. Inter-field grouping
    if task in ["receipt_v1", "funsd"]:
        assert field_rs is not None
        label_gs = [label[g_label_idx] for label in label_units]
        parses, grouped_col_ids = gen_fg_parses(fields, field_rs, label_gs, max_info_depth,
                                                f_parses, f_parse_head_ids, strict,)
            
    else:
        raise NotImplementedError

    return parses, f_parses, text_unit_field_labels, f_parse_box_ids


def gen_root_fg_parses_new_rel(field_roots, new_rel_mats, grouped_col_ids, parses):
    parses_root = []
    for b, parse in enumerate(parses):
        representer_col_id = list(grouped_col_ids[b].keys())
        parse_root = []
        for i_root, field_root in enumerate(field_roots):
            new_parse = []
            root_f_parse_head_id = np.nonzero(new_rel_mats[b][i_root])[0]
            for root_f_parse_head_id1 in root_f_parse_head_id:
                if root_f_parse_head_id1 in representer_col_id:
                    idx = representer_col_id.index(root_f_parse_head_id1)
                    new_parse.append(parse[idx])
            parse_root.append(new_parse)

        parses_root.append(parse_root)

    return parses_root


def extract_root_f_parses(f_parses, f_parse_head_ids, field_roots):
    new_f_parses = []
    new_f_parse_head_ids = []

    root_f_parses = []
    root_f_parse_head_ids = []

    for f_parse, f_parse_head_id in zip(f_parses, f_parse_head_ids):
        new_f_parse = []
        new_f_parse_head_id = []

        root_f_parse = []
        root_f_parse_head_id = []

        for kv, f_parse_head_id1 in zip(f_parse, f_parse_head_id):
            k = get_key_from_single_key_dict(kv)
            if k in field_roots:
                root_f_parse.append(kv)
                root_f_parse_head_id.append(f_parse_head_id1)
            else:
                new_f_parse.append(kv)
                new_f_parse_head_id.append(f_parse_head_id1)

        new_f_parses.append(new_f_parse)
        new_f_parse_head_ids.append(new_f_parse_head_id)

        root_f_parses.append(root_f_parse)
        root_f_parse_head_ids.append(root_f_parse_head_id)

    return new_f_parses, new_f_parse_head_ids, root_f_parses, root_f_parse_head_ids


def gen_root_fg_parses(field_roots, root_f_parses, root_f_parse_head_ids, grouped_col_ids, parses):
    parses_root = []
    for b, parse in enumerate(parses):
        representer_col_id = list(grouped_col_ids[b].keys())
        parse_root = []
        for field_root in field_roots:
            new_parse = []
            for root_f_parse1, root_f_parse_head_id1 in zip(root_f_parses[b], root_f_parse_head_ids[b]):
                k = get_key_from_single_key_dict(root_f_parse1)
                if k == field_root:
                    if root_f_parse_head_id1 in representer_col_id:
                        idx = representer_col_id.index(root_f_parse_head_id1)
                        new_parse.append(parse[idx])

            parse_root.append(new_parse)
        parses_root.append(parse_root)

    return parses_root


def gen_f_parses(task, fields, text_units, label_fs, l_max_gen):
    parses = []
    parse_box_ids = []
    parse_head_ids = []

    target_relation = 1  # Currently only single type in rel-s. 0 for no relation.
    row_offset = len(fields)

    for b, (text_unit, label_f) in enumerate(zip(text_units, label_fs)):
        # for each batch
        parse = []
        parse_box_id = []
        parse_head_id = []
        rel_mat = (label_f.cpu() if isinstance(label_f, torch.Tensor) else label_f)  # edge

        for i, field in enumerate(fields):
            row = np.array(rel_mat[i])
            # 1. Seeding: Find seed nodes for each field type.
            idx_nz = np.where(row == target_relation)[0]  # non-zero
            for i_nz in idx_nz:
                # 2. Serialization: from seed nodes, generate boxes recursively.
                boxes = [text_unit[i_nz]]
                box_ids = [i_nz]
                boxes, box_ids = gen_boxes_single_path(boxes, box_ids, i_nz, text_unit, rel_mat, row_offset, target_relation, l_max_gen,)

                parse.append({field: boxes})
                box_ids = np.array(box_ids).tolist()
                parse_box_id.append({field: box_ids})
                parse_head_id.append(i_nz)

        parses.append(refine_parse(task, parse))
        parse_box_ids.append(parse_box_id)
        parse_head_ids.append(parse_head_id)

    return parses, parse_box_ids, parse_head_ids


def refine_parse(task, parse: list):
    if task == "namecard" or "receipt_v1":
        new_parse = []
        for parse1 in parse:
            assert len(parse1) == 1
            for k, v in parse1.items():
                # new_parse1 = {k: ''.join(v)}
                new_parse1 = {k: " ".join(v)}

            new_parse.append(new_parse1)
    else:
        raise NotImplementedError
    return new_parse


def gen_boxes_single_path(boxes, box_ids, col_idx, text_unit, rel_mat, row_offset, target_relation, l_max_gen):
    row = np.array(rel_mat[col_idx + row_offset])

    next_col_idxs = np.where(row == target_relation)[0]
    if next_col_idxs.size > 0 and len(boxes) < l_max_gen:
        assert next_col_idxs.size == 1
        next_col_idx = next_col_idxs[0]
        boxes += [text_unit[next_col_idx]]
        box_ids += [next_col_idx]
        return gen_boxes_single_path(boxes, box_ids, next_col_idx, text_unit, rel_mat, row_offset, target_relation, l_max_gen,)

    else:
        return boxes, box_ids


def detokenize_f_parse(f_parses, f_parse_box_ids, header_toks, backbone_name):
    if backbone_name in ["bert-base-multilingual-cased"]:
        chars_for_detok = "##"
    else:
        raise NotImplementedError
    new_f_parses = []
    for f_parse, f_parse_box_id, header_tok in zip(f_parses, f_parse_box_ids, header_toks):
        new_f_parse = []
        for i_parse, (f_parse1, f_parse_box_id1) in enumerate(zip(f_parse, f_parse_box_id)):
            key = get_key_from_single_key_dict(f_parse1)
            val = list(f_parse1.values())[0]

            id_val = list(f_parse_box_id1.values())[0]
            toks = val.split()

            # Clean first tok
            words = toks[0].replace(chars_for_detok[0], "")
            for i in range(len(toks) - 1):
                tok_pre = toks[i]
                tok = toks[i + 1]
                box_id1 = id_val[i + 1]
                is_not_head = not header_tok[box_id1]

                if tok.startswith(chars_for_detok):
                    tok_trimed = tok[2:]
                else:
                    if (tok in [".", ","]
                    or (len(words) > 0 and words[-1] in [".", ","] and tok.isdigit())
                    or (tok in [")", "%", "/"])
                    or (tok_pre in ["@", "(", "/"]) 
                    or is_not_head):
                        tok_trimed =       tok
                    else:
                        tok_trimed = " " + tok

                words += tok_trimed
            new_f_parse.append({key: words})
        new_f_parses.append(new_f_parse)

    return new_f_parses


def gen_fg_parses(fields, field_rs, label_gs, max_info_depth, f_parses, f_parse_head_ids, strict,):
    row_offset = len(fields)
    representer_col_ids, _ = find_representer_col_ids(field_rs, f_parses, f_parse_head_ids)
    groups = gen_groups_col_id(representer_col_ids, label_gs, row_offset, max_info_depth=max_info_depth)
    parses, remained_f_parse_head_ids = gen_grouped_parses(groups, f_parses, f_parse_head_ids, strict, max_info_depth)

    return parses, groups


def find_representer_col_ids(field_rs, f_parses, f_parse_head_ids):
    representer_col_ids = []
    representer_fields = []
    for f_parse, f_parse_head_id in zip(f_parses, f_parse_head_ids):
        representer_col_id = []
        representer_field = []
        for f_parse1, f_parse_head_id1 in zip(f_parse, f_parse_head_id):
            field_of_target = get_key_from_single_key_dict(f_parse1)
            if field_of_target in field_rs:
                representer_col_id.append(f_parse_head_id1)
                representer_field.append(field_of_target)
        representer_col_ids.append(representer_col_id)
        representer_fields.append(representer_field)
    return representer_col_ids, representer_fields


def gen_groups_col_id(representer_col_ids, label_gs, row_offset, max_info_depth):
    """
    Retruns
    groups: [ {rep_col_id: [id, id, id]}, ... ]
    """
    target_relation = 1
    groups = []
    for b, representer_col_id in enumerate(representer_col_ids):
        rel_mat = (label_gs[b].cpu() if isinstance(label_gs[b], torch.Tensor) else label_gs[b])  # edge
        group = OrderedDict()
        for i_rep, representer_col_id1 in enumerate(representer_col_id):
            group[representer_col_id1] = gen_group_col_id1(representer_col_id1, rel_mat, row_offset, target_relation, 
                                                            max_info_depth, lv=0,)
        groups.append(group)

    return groups


def gen_group_col_id1(representer_col_id1, rel_mat, row_offset, target_relation, max_info_depth, lv=0):
    row = np.array(rel_mat[representer_col_id1 + row_offset])
    ids_nz = np.where(row == target_relation)[0]
    if lv == max_info_depth - 1:
        return ids_nz.tolist()
    else:
        group = OrderedDict()
        for i_nz in ids_nz:
            group[i_nz] = gen_group_col_id1(i_nz, rel_mat, row_offset, target_relation, max_info_depth, lv+1)
        return group


def gen_grouped_parses(groups, f_parses, f_parse_head_ids, strict, max_info_depth):

    parses = []
    remained_f_parse_head_ids = deepcopy(f_parse_head_ids)
    for b, group in enumerate(groups):
        parse = gen_grouped_parse1(group, f_parse_head_ids[b], f_parses[b],
                                 remained_f_parse_head_ids[b], strict,)
        # pprint(parse)
        parses.append(parse)
    return parses, remained_f_parse_head_ids


def gen_grouped_parse1(group, f_parse_head_id, f_parse, remained_f_parse_head_id, strict):

    # group: OrderedDict. Recursive.
    parse = []

    for i_grp, id_rep in enumerate(group):
        sub_group = group[id_rep]
        if not sub_group or isinstance(sub_group, list):
            id_members = [id_rep]
            if isinstance(sub_group, list):
                id_members = [id_rep] + sub_group
            sub_parse = []
            for id_member in id_members:
                f_parse1 = imp_get_f_parse_from_id_member(strict, f_parse, f_parse_head_id,
                                                                  remained_f_parse_head_id, id_member,)
                if f_parse1 is not None:
                    sub_parse.append(f_parse1)

        else:
            f_parse1 = imp_get_f_parse_from_id_member(strict, f_parse, f_parse_head_id, remained_f_parse_head_id, id_rep)
            sub_parse_value = gen_grouped_parse1(sub_group, f_parse_head_id, f_parse, remained_f_parse_head_id, strict)
            sub_parse = {f"{f_parse1}": sub_parse_value}
        parse.append(sub_parse)
    return parse


def imp_get_f_parse_from_id_member(strict, f_parse, f_parse_head_id, remained_f_parse_head_id, id_member):
    if strict:
        idx = f_parse_head_id.index(id_member)
    else:
        if id_member in f_parse_head_id:
            idx = f_parse_head_id.index(id_member)
        else:
            idx = None
    if idx is not None:
        remained_f_parse_head_id[idx] = -1
        return f_parse[idx]
    else:
        return None
    return idx

