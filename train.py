import numpy as np
import pickle
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd
import geopandas as gpd
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
import time
from haversine import haversine
import networkx as nx
import multiprocessing as mp
from args import *
from my_constants import *
from scipy import stats
import statistics
from traffic import *
from utils import *
import torch_geometric
from termcolor import cprint, colored
import sys
from collections import OrderedDict
import datetime
import collections
from tqdm import tqdm
from termcolor import cprint
from models_general import GNN, MLP
from Model_ORL_OPP import Model, PrioritizedReplayBuffer

args = make_args()
print(args)

unimplemented = False

if unimplemented:
    print("pending functionality")
    raise SystemExit

TRIPS_TO_EVALUATE = 100000
MAX_ITERS = 300
BATCH = args.batch_size
PRINT_FREQ = 1000

JUMP = 10000

if args.check_script:
    TRIPS_TO_EVALUATE = 10000
    PRINT_FREQ = 50


def trip_length(path):
    global graph, backward
    return sum(
        [graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path])


def intersections_and_unions(path1, path2):
    global graph, backward
    path1, path2 = set(path1), set(path2)
    intersection = sum(
        [graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in
         path1.intersection(path2)])
    union = sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in
                 path1.intersection(path2)])

    return intersection, union


def shorten_path(path, true_dest):
    global map_edge_id_to_u_v, backward, map_node_osm_to_coords
    dest_node = map_edge_id_to_u_v[backward[true_dest]][0]
    _, index = min([(haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[edge]][1]],
                               map_node_osm_to_coords[dest_node]), i) for i, edge in enumerate(path)])
    return path[:index + 1]


def gen_paths_no_hierarchy(all_paths):
    global JUMP
    ans = []
    for i in tqdm(list(range(0, len(all_paths), JUMP)), desc="batch_eval", dynamic_ncols=True):
        temp = all_paths[i:i + JUMP]
        ans.append(gen_paths_no_hierarchy_helper(temp))
    return [t for sublist in ans for t in sublist]


def gen_paths_no_hierarchy_helper(all_paths):
    global model_node_embeddings, node_nbrs, max_nbrs, edge_to_node_mapping
    global forward_interval_map
    if args.traffic:
        intervals = [forward_interval_map[(s)] for _, _, (s, _) in all_paths]
    true_paths = [p for _, p, _ in all_paths]

    model_node_embeddings.eval()
    pending = OrderedDict({i: None for i in range(len(all_paths))})

    results = []
    with torch.no_grad():
        for idx, (_, traj, time) in enumerate(tqdm(all_paths, desc="generating paths")):
            start = traj[0]
            end = traj[-1]
            t = 0
            t_time = 0
            path = [start]
            current = start
            visited = set()

            while current != end and current not in visited and t <= 300:
                t += 1
                visited.add(current)
                neighbors = node_nbrs[current]
                valid_neighbors = [n for n in neighbors if n != -1]

                if not valid_neighbors:
                    break

                q_values = []
                for next_node in valid_neighbors:
                    q_val = model_node_embeddings(int(next_node), return_vec=True)
                    q_values.append(q_val[0].mean().item())

                best_next, best_q = max(zip(valid_neighbors, q_values), key=lambda x: x[1])

                t_time += speed_dict[best_next]
                path.append(best_next)
                current = best_next

            origin_time = (time[-1] - time[0]) / 3600
            ratio = (origin_time - t_time) / origin_time if origin_time > 0 else 0

            if current != end:
                results.append((None, float('inf'), idx))
            else:
                results.append((t, path, ratio))

    model_node_embeddings.train()
    return results


def evaluate_no_hierarchy(data, num=1000, with_correction=False, without_correction=True, with_dijkstra=False):
    global map_node_osm_to_coords, map_edge_id_to_u_v, backward
    to_do = ["precision", "recall", "reachability", "avg_reachability", "acc", "nll", "generated"]
    results = {s: None for s in to_do}
    cprint("Evaluating {} number of trips".format(num), "magenta")
    partial = random.sample(data, num)
    t1 = time.time()
    if with_dijkstra:
        gens = [dijkstra(t) for t in tqdm(partial, desc="Dijkstra for generation", unit="trip", dynamic_ncols=True)]
    else:
        gens = gen_paths_no_hierarchy(partial)
    # elapsed = time.time() -t1
    # results["time"] = elapsed
    # jaccs = []
    # preserved_with_stamps = partial.copy()
    # partial = [p for _,p,_ in partial]
    # print("Without correction (everything is weighed according to the edge lengths)")
    # generated = list(zip(partial, gens))
    # generated = [(t,g) for t,g in generated if len(t)>1]
    # lengths = [(trip_length(t), trip_length(g)) for (t,g) in generated]
    # inter_union = [intersections_and_unions(t, g) for (t,g) in generated]
    # m = len(generated)
    # inters = [inter for inter,union in inter_union]
    # unions = [union for inter,union in inter_union]
    # lengths_gen = [l_g for l_t,l_g in lengths]
    # lengths_true = [l_t for l_t,l_g in lengths]
    # precs = [i/l if l >0 else 0 for i,l in zip(inters, lengths_gen) ]
    # precision1 = round(100*sum(precs)/len(precs), 2)
    # recs = [i/l if l >0 else 0 for i,l in zip(inters, lengths_true) ]
    # recall1 = round(100*sum(recs)/len(recs), 2)
    # deepst_accs = [i/max(l1,l2) for i,l1,l2 in zip(inters, lengths_true, lengths_gen) if max(l1,l2)>0]
    # deepst = round(100*sum(deepst_accs)/len(deepst_accs), 2)
    #
    # num_reached = len([None for t,g in generated if t[-1] == g[-1]])
    # lefts = [haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[g[-1]]][0]], map_node_osm_to_coords[map_edge_id_to_u_v[backward[t[-1]]][0]]) for t,g in generated]
    # rights = [haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[g[-1]]][1]], map_node_osm_to_coords[map_edge_id_to_u_v[backward[t[-1]]][1]]) for t,g in generated]
    # reachability = [(l+r)/2 for (l,r) in zip(lefts,rights)]
    # all_reach = np.mean(reachability)
    # all_reach = round(1000*all_reach,2)
    #
    # if len(reachability) != num_reached:
    # 	reach_reach = sum(reachability)/(len(reachability)-num_reached)
    # else:
    # 	reach_reach = 0
    #
    # reach_reach = round(1000*reach_reach,2)
    #
    # percent_reached = round(100*(num_reached/len(reachability)), 2)
    # print()
    # cprint("Precision is                            {}%".format(precision1), "green")
    # cprint("Recall is                               {}%".format(recall1), "green")
    # print()
    # cprint("%age of trips reached is                {}%".format(percent_reached), "green")
    # cprint("Avg Reachability(across all trips) is   {}m".format(all_reach), "green")
    # cprint("Avg Reach(across trips not reached) is  {}m".format(reach_reach), "green")
    # print()
    # cprint("Deepst's Accuracy metric is             {}%".format(deepst), "green", attrs = ["dark"])
    # print()
    # results["precision"] = precision1
    # results["reachability"] = percent_reached
    # results["avg_reachability"] = (all_reach, reach_reach)
    # results["recall"] = recall1
    # results["generated"] = list(zip(preserved_with_stamps, gens))
    return gens


def lipschitz_node_embeddings(nodes_forward, G, k):
    nodes = list(nodes_forward.keys())
    G_temp = G.reverse(copy=True)
    anchor_nodes = random.sample(nodes, k)
    print('Starting Dijkstra')
    num_workers = 32
    cutoff = None
    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range, \
                                args=(
                                G_temp, anchor_nodes[int(k / num_workers * i):int(k / num_workers * (i + 1))], cutoff))
               for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    print('Dijkstra done')
    embeddings = np.zeros((len(nodes), k))
    for i, node_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(nodes):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                embeddings[nodes_forward[node_j], i] = 1 / (dist + 1)
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    return embeddings


def local_nodes_to_global_edges(trip):
    global backward
    osm_trip = [backward[e] for e in trip]
    return osm_trip


def dijkstra(true_trip):
    global args, transformed_graph, max_nbrs
    _, (src, *_, dest), (s, _) = true_trip
    g = transformed_graph
    model.eval()
    with torch.no_grad():
        current_temp = [c for c in g.nodes()]
        current = [c for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else [])]
        pot_next = [nbr for c in current_temp for nbr in (node_nbrs[c] if c in node_nbrs else [])]
        dests = [dest for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else [])]
        traffic = None
        if args.traffic:
            traffic = [forward_interval_map[s] for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else [])]

        unnormalized_confidence = model(current, dests, pot_next, traffic)
        unnormalized_confidence = -1 * torch.nn.functional.log_softmax(unnormalized_confidence.reshape(-1, max_nbrs),
                                                                       dim=1)
        transition_nll = unnormalized_confidence.detach().cpu().tolist()
    model.train()
    torch.cuda.empty_cache()
    count = 0
    for u in g.nodes():
        for i, nbr in enumerate(node_nbrs[u]):
            if nbr == -1:
                break
            g[u][nbr]["nll"] = transition_nll[count][i]
        count += 1
    path = nx.dijkstra_path(g, src, dest, weight="nll")
    path = [x for x in path]
    return path


def compare_with_dijkstra(generated, other_time=None):
    t1 = time.time()
    dijkstra_output = [dijkstra(t) for (t, _) in
                       tqdm(generated, desc="Dijkstra for generation", unit="trip", dynamic_ncols=True)]
    elapsed = time.time() - t1
    with_greedy = [(t[1], g) for t, g in generated if len(t) > 1]
    with_dijkstra = [(t[1], g) for (t, _), g in zip(generated, dijkstra_output) if len(t) > 1]
    reached_with_greedy = [(t, g) for t, g in with_greedy if t[-1] == g[-1]]
    reached_with_dijkstra = [a for (a, (t, g)) in zip(with_dijkstra, with_greedy) if t[-1] == g[-1]]
    percent_reached = round(100 * (len(reached_with_greedy) / len(generated)), 2)
    comparisons = {"all": "all queries", "reached": "only those queries where greedy reached"}
    cols = ["cyan", "cyan"]
    descriptions = ["", "({}% trips)".format(percent_reached)]
    to_compare = [(with_greedy, with_dijkstra), (reached_with_greedy, reached_with_dijkstra)]
    t_greedy = round(other_time, 2)
    s_greedy = round(len(generated) / other_time, 2)
    t_dijkstra = round(elapsed, 2)
    s_dijkstra = round(len(generated) / elapsed, 2)
    cprint("\nGreedy vs Dijktsra", "yellow")
    cprint("Comparing generation times - {}s and {}s for greedy and dijkstra to generate {} trips"
           .format(t_greedy, t_dijkstra, len(generated)), "cyan")
    cprint("Comparing generation speeds - {} trips/s and {} trips/s for greedy and dijkstra"
           .format(s_greedy, s_dijkstra), "cyan")
    results = {}
    for comparison_type, col, desc, generation in zip(comparisons, cols, descriptions, to_compare):
        p = []
        r = []
        dst = []
        print()
        if comparison_type == "reached" and len(reached_with_greedy) == 0:
            cprint("No trip reached with greedy, so cannot run that comparison")
            return
        for gen in generation:
            lengths = [(trip_length(t), trip_length(g)) for (t, g) in gen]
            inter_union = [intersections_and_unions(t, g) for (t, g) in gen]
            m = len(generated)
            inters = [inter for inter, union in inter_union]
            unions = [union for inter, union in inter_union]
            lengths_gen = [l_g for l_t, l_g in lengths]
            lengths_true = [l_t for l_t, l_g in lengths]
            precs = [i / l if l > 0 else 0 for i, l in zip(inters, lengths_gen)]
            precision1 = round(100 * sum(precs) / len(precs), 2)
            p.append(precision1)
            recs = [i / l if l > 0 else 0 for i, l in zip(inters, lengths_true)]
            recall1 = round(100 * sum(recs) / len(recs), 2)
            r.append(recall1)
            deepst_accs = [i / max(l1, l2) for i, l1, l2 in zip(inters, lengths_true, lengths_gen) if max(l1, l2) > 0]
            deepst = round(100 * sum(deepst_accs) / len(deepst_accs), 2)
            dst.append(deepst)
        if comparison_type == "reached":
            results["precision_reached"] = p[1]
            results["recall_reached"] = r[1]
        else:
            results["precision_all"] = p[1]
            results["recall_all"] = r[1]
        cprint("Comparing {} {}".format(comparisons[comparison_type], desc), "yellow", attrs=["bold"])
        cprint("precision : Greedy & Dijkstra are {}% and {}%".format(*p), col)
        cprint("recall    : Greedy & Dijkstra are {}% and {}%\n".format(*r), col)
        cprint("deepST acc: Greedy & Dijkstra are {}% and {}%".format(*dst), col, attrs=["dark"])
    print()
    return results


def plot_performance_against_trip_lengths(
        generated,
        bins=[0, 2, 5, 10, 15, 20, 30, 100],
        filename="results/performance_against_trip_length.png"
):
    generated = [(t, g) for ((_, t, _), g) in generated]
    lengths = [(trip_length(t), trip_length(g)) for (t, g) in generated]
    inter_union = [intersections_and_unions(t, g) for (t, g) in generated]
    m = len(generated)
    inters = [inter for inter, union in inter_union]
    unions = [union for inter, union in inter_union]
    lengths_gen = [l_g for l_t, l_g in lengths]
    lengths_true = [l_t for l_t, l_g in lengths]
    precs = [100 * i / l if l > 0 else 0 for i, l in zip(inters, lengths_gen)]
    recs = [100 * i / l if l > 0 else 0 for i, l in zip(inters, lengths_true)]
    distance_from_dest = [100 if g[-1] == t[-1] else 0 for t, g in generated]
    trip_lengths = [trip_length(t) for t, g in generated]
    bs_reachability = stats.binned_statistic(trip_lengths, distance_from_dest, 'mean', bins=bins)
    bs_reachability_count = stats.binned_statistic(trip_lengths, distance_from_dest, 'count', bins=bins)
    bs_prec = stats.binned_statistic(trip_lengths, precs, 'mean', bins=bins)
    bs_rec = stats.binned_statistic(trip_lengths, recs, 'mean', bins=bins)
    counts = stats.binned_statistic(trip_lengths, [1] * len(trip_lengths), 'sum', bins=bins)
    counts = counts.statistic.tolist()
    x_axis = ["{} and {} km".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    plt.clf()
    w = 0.2
    x = np.arange(len(bins) - 1)
    plt.bar(x, bs_prec.statistic, width=w, label="precision", color="red", align="center")
    plt.bar(x + w, bs_rec.statistic, width=w, label="recall", color="blue", align="center")
    plt.bar(x - w, bs_reachability.statistic, width=-w, label="reachability", color="green", align="center")
    plt.ylim(0, 100)
    plt.ylabel("Performance metric %age")
    plt.xlabel("Trips whose length was between")
    plt.xticks(x, x_axis, rotation=45)
    plt.title("Perfomance variation with trip length")
    plt.legend(loc="best")
    plt.savefig(filename, bbox_inches="tight")
    results = {}
    s = sum(counts)
    percentage_counts = [round(100 * x / s, 2) for x in counts]
    results["precision"] = bs_prec.statistic.tolist()
    results["recall"] = bs_rec.statistic.tolist()
    results["reachability"] = bs_reachability.statistic.tolist()
    results["bins"] = bins
    results["counts"] = counts
    results["percentage_counts"] = percentage_counts
    return results


def save_model(path_model=MODEL_SAVE_PATH, path_extras=MODEL_SUPPORT_PATH):
    global map_node_osm_to_coords, map_edge_id_to_u_v, forward, model
    torch.save(model, path_model)
    f = open(path_extras, 'wb')
    pickle.dump((forward, map_node_osm_to_coords, map_edge_id_to_u_v), f)
    f.close()


destination_nodes = set()

def time_mean(all_paths):
    edge_times = defaultdict(list)
    for _, traj, (s, e) in all_paths:
        t_total = (e - s) / 3600
        l_total = sum([haversines[ej] for ej in traj])
        if l_total == 0:
            continue
        speed = l_total / t_total

        for i in traj:
            dist = haversines[i]
            time = dist / speed
            edge_times[i].append(time)

    edge_avg_time = {edge: sum(times) / len(times) for edge, times in edge_times.items()}
    return edge_avg_time


def merge_dicts_average(*dicts):
    merged = defaultdict(list)

    for d in dicts:
        for k, v in d.items():
            merged[k].append(v)


    result = {k: sum(v_list) / len(v_list) for k, v_list in merged.items()}
    return result



def extract_data_points(partial):
    data_points = []
    for _, traj, (s, e) in partial:
        for i in range(len(traj) - 1):
            s = int(traj[i])
            a = int(traj[i + 1])
            s_prime = a
            time_penalty = speed_dict[i]

            dist_to_goal = haversines[traj[-1]] - haversines[a] if traj[-1] in haversines and a in haversines else 0

            r = -time_penalty - 0.1 * dist_to_goal
            data_points.append((s, a, r, s_prime))

        destination_nodes.add(traj[-1])
    return data_points


if __name__ == "__main__":
    time_start = datetime.datetime.now()
    cprint('This file was run on {}'.format(time_start), 'cyan')

    f = open("./preprocessed_data/" + "{}/".format(args.dataset) + "map/graph_with_haversine.pkl", 'rb')
    graph = pickle.load(f)

    for e in graph.edges(data=True):
        e[2]['length'] = e[2]['length'] / 1000
    f.close()


    train_data, forward = load_data(args=args, fname=TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)

    test_data = load_test_data(args, forward, fname=TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
    val_data = load_test_data(args, forward, fname=VAL_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)


    backward = {forward[k]: k for k in forward}

    node_nbrs = create_node_nbrs(forward)  # 每个边的领接边

    nbrs_sanity_check(node_nbrs, train_data)
    nbrs_sanity_check(node_nbrs, test_data)
    nbrs_sanity_check(node_nbrs, val_data)


    transformed_graph = nx.DiGraph()
    for e1 in node_nbrs:
        for e2 in node_nbrs[e1]:
            if e2 != -1:
                transformed_graph.add_edge(e1, e2)

        device = torch.device(
            'cuda:{}'.format(args.gpu_index) if ((not args.force_cpu) and torch.cuda.is_available()) else 'cpu')
        print("running this on {}".format(device))


        traffic_matrix = None
        if args.traffic:
            find_interval = find_interval_2 if args.ignore_day else find_interval_1
            traffic_feature_store = fetch_traffic_features_stored(device=device, find_interval=find_interval)
            all_intervals = list(traffic_feature_store.keys())
            all_intervals.sort()
            forward_interval_map = {interval: index for index, interval in enumerate(all_intervals)}
            backward_interval_map = all_intervals
            traffic_matrix = torch.empty((len(all_intervals), 10))
            for i, interval in enumerate(all_intervals):
                traffic_matrix[i] = traffic_feature_store[interval]
            traffic_matrix = traffic_matrix.float().to(device)
            train_data = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in
                          tqdm(train_data, desc="marking intervals")]
            val_data = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in val_data]
            test_data = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in test_data]
            test_data_fixed = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in test_data_fixed]

        loss_function_cross_entropy = nn.MSELoss(reduction="sum")
        sigmoid_function = nn.Sigmoid()

        nodes_used = set()

        for e in forward:
            u, v = map_edge_id_to_u_v[e]
            nodes_used.add(u)
            nodes_used.add(v)
        nodes_used = list(nodes_used)
        nodes_forward = {node: i for i, node in enumerate(nodes_used)}

        edge_to_node_mapping = {
            forward[e]: (nodes_forward[map_edge_id_to_u_v[e][0]], nodes_forward[map_edge_id_to_u_v[e][1]]) for e in
            forward}
        edge_to_node_mapping[-1] = (-1, -1)

        speed_dict = dict(zip(speed_df['key'], speed_df['value']))

        embeddings = None
        if (args.initialise_embeddings_lipschitz):
            embeddings = lipschitz_node_embeddings(nodes_forward, graph, args.embedding_size)
            map_node_zero_indexed_to_coords = {nodes_forward[n]: map_node_osm_to_coords[n] for n in nodes_forward}

        if (args.gnn is not None):
            node_embeddings = torch.from_numpy(embeddings).float() if embeddings is not None else None
            node_feats = node_embeddings
            edge_index = []
            for u, v in map_edge_id_to_u_v:
                if u in nodes_forward and v in nodes_forward:
                    u, v = nodes_forward[u], nodes_forward[v]
                    edge_index.append((u, v))
            edge_index = torch.LongTensor(edge_index).T
            torch_graph = torch_geometric.data.Data(x=node_feats, edge_index=edge_index)
            torch_graph = torch_graph.to(device)
            model_node_embeddings = Model(num_nodes=len(nodes_forward),
                                          graph=torch_graph,
                                          device=device,
                                          args=args,
                                          embeddings=node_embeddings,
                                          mapping=edge_to_node_mapping,
                                          traffic_matrix=traffic_matrix).to(device)
        else:
            model_node_embeddings = Model(num_nodes=len(nodes_forward),
                                          device=device,
                                          args=args,
                                          embeddings=(None if embeddings is None else torch.from_numpy(embeddings)),
                                          mapping=edge_to_node_mapping,
                                          traffic_matrix=traffic_matrix).to(device)

        optimiser = torch.optim.Adam(model_node_embeddings.parameters(), lr=args.lr, amsgrad=True)

        max_nbrs = max(len(nbr_array) for nbr_array in node_nbrs.values())
        num_nodes = len(forward)

        for u in range(num_nodes):
            if u in node_nbrs:
                node_nbrs[u].extend([-1] * (max_nbrs - len(node_nbrs[u])))
            else:
                node_nbrs[u] = [-1] * max_nbrs

        loss_curve = []
        train_acc_curve = []
        test_acc_curve = []
        max_len = 1 + max(len(t) for _, t, _ in train_data)

        total_loss = 0
        total_trajs = 0
        preds = 0
        correct = 0
        prob_sum = 0

        level = 0
        val_evals_till_now_reachability = []
        val_evals_till_now_precision = []
        val_evals_till_now_recall = []

        if args.initial_eval:
            tqdm.write(colored("\nInitial Eval on Validation set", "blue", attrs=["bold", "underline"]))
            val_results = evaluate_no_hierarchy(data=val_data, num=len(val_data), with_correction=False,
                                                with_dijkstra=False)
            val_evals_till_now_reachability.append(val_results["reachability"])
            val_evals_till_now_precision.append(val_results["precision"])
            val_evals_till_now_recall.append(val_results["recall"])


        per_buffer = PrioritizedReplayBuffer(capacity=100000)

        for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epochs", dynamic_ncols=True):
            random.shuffle(train_data)
            model_node_embeddings.train()

            num_iterations = 1
            for batch_num, k in tqdm(list(enumerate((range(0, len(train_data), BATCH)))),
                                     desc="Batch", unit="steps", leave=True, dynamic_ncols=True):

                partial = random.sample(train_data, BATCH)
                valid_trajs = len(partial)

                adj_list = {}
                for _, traj, _ in partial:
                    for i in range(len(traj) - 1):
                        s, a = int(traj[i]), int(traj[i + 1]),
                        if s not in adj_list:
                            adj_list[s] = set()
                        adj_list[s].add(a)

                for s in list(adj_list.keys()):
                    for a in adj_list[s]:
                        if a not in adj_list:
                            adj_list[a] = set()
                        adj_list[a].add(s)


                for _, traj, _ in partial:
                    initial_data_points = extract_data_points(train_data)
                for s, a, r, s_prime in initial_data_points:
                    td_error = abs(r)
                    per_buffer.add((s, a, r, s_prime), td_error)

                for iteration in range(num_iterations):
                    samples, indices = per_buffer.sample(batch_size=64)
                    targets = []
                    inputs = []
                    new_td_errors = []

                    for s, a, r, s_prime in samples:
                        if s_prime in destination_nodes:
                            target = r
                        else:
                            q_values = []
                            for a_next in adj_list.get(s_prime, []):
                                input_vec = torch.cat((model_node_embeddings(int(s_prime), return_vec=True),
                                                       model_node_embeddings(int(a_next), return_vec=True)), dim=0)
                                q_val = model_node_embeddings(input_vec.mean(dim=0), return_vec=False).item()
                                q_values.append(q_val)
                            target = r + (max(q_values) if q_values else 0)

                        input_vec = torch.cat((model_node_embeddings(int(s), return_vec=True),
                                               model_node_embeddings(int(a), return_vec=True)), dim=0)
                        inputs.append(input_vec)
                        targets.append(target)

                    inputs = torch.stack(inputs)
                    current_q = model_node_embeddings(inputs, return_vec=False).squeeze(-1).mean(dim=1)
                    targets = torch.tensor(targets, dtype=torch.float).to(inputs.device)
                    td_errors = targets - current_q.detach()
                    loss = loss_function_cross_entropy(current_q, targets)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    per_buffer.update_priorities(indices, td_errors.cpu().numpy())


            if (epoch + 1) % args.eval_frequency == 0:
                # save_model()
                # cprint('Model saved', 'yellow', attrs=['underline'])
                tqdm.write(colored("\nDoing a partial evaluation on train set", "blue", attrs=["bold", "underline"]))
                tqdm.write(colored("\nStandard", "cyan", attrs=["bold", "reverse", "blink"]))
                train_results = evaluate_no_hierarchy(data=train_data,
                                                         num=min(TRIPS_TO_EVALUATE, len(train_data)),
                                                         with_correction=False,
                                                         without_correction=True,
                                                         with_dijkstra=False)
                tqdm.write(colored("\nEvaluation on the validation set (size = {})".format(len(val_data)), "blue",
                                   attrs=["bold", "underline"]))
                tqdm.write(colored("\nStandard", "cyan", attrs=["bold", "reverse", "blink"]))
                val_results = evaluate_no_hierarchy(data=val_data,
                                                       num=len(val_data),
                                                       with_correction=False,
                                                       without_correction=True,
                                                       with_dijkstra=False)
                if (args.with_dijkstra):
                    tqdm.write(colored("\nEvaluation on the validation set (size = {})".format(len(val_data)), "blue",
                                       attrs=["bold", "underline"]))
                    tqdm.write(colored("\nDIJKSTRA", "cyan", attrs=["bold", "reverse", "blink"]))
                    val_results, tvt = evaluate_no_hierarchy(data=val_data,
                                                           num=len(val_data),
                                                           with_correction=False,
                                                           without_correction=True,
                                                           with_dijkstra=True)

        tqdm.write(colored("\nAfter training for {} epochs, ".format(epoch + 1), "yellow"))
        tqdm.write(colored("FINAL EVALUATION ON TEST\n", "blue", attrs=["bold", "underline"]))
        tqdm.write(colored("\nStandard", "cyan", attrs=["bold", "reverse", "blink"]))
        test_result = evaluate_no_hierarchy(data=test_data,
                                                num=len(test_data),
                                                with_correction=True,
                                                with_dijkstra=False)
