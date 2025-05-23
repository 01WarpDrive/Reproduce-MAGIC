import argparse
import json
import os
import re
from tqdm import tqdm
import networkx as nx
import pickle as pkl


SRC_DIR = 'F:/datasets/LANL/unzip/' # Directory of flows.txt, auth.txt
RED = f'{SRC_DIR}redteam.txt' # Location of redteam.txt
SRC = f'{SRC_DIR}auth.txt' # Location of auth.txt
DATE_OF_EVIL_LANL = 150885 # train data before time
END_TEST_TIME = 1270000

node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0


pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')


def read_single_graph(path):
    global node_type_cnt, edge_type_cnt
    g = nx.DiGraph()
    print('converting {} ...'.format(path))
    f = open(path, 'r')
    lines = []

    node_type_map = {}
    edge_type_map = {}
    for l in f.readlines():
        src, src_type, dst, dst_type, edge_type, ts = l.split('\t')
        src_u, dst_u, s_or_f = edge_type.split('_')

        if src not in node_type_map:
            node_type_map[src] = [False, False, False]
        if dst not in node_type_map:
            node_type_map[dst] = [False, False, False]
        
        if src_u[0] == 'C':
            idx = 0
        elif src_u[0] == 'U':
            idx = 1
        else:
            idx = 2
        node_type_map[src][idx] = True

        if dst_u[0] == 'C':
            idx = 0
        elif dst_u[0] == 'U':
            idx = 1
        else:
            idx = 2
        node_type_map[dst][idx] = True
        
        edge = (src, dst)
        if edge not in edge_type_map:
            edge_type_map[edge] = [False, False]

        if s_or_f == 'Success':
            idx = 0
        else:
            idx = 1
        edge_type_map[edge][idx] = True
    
    for k in node_type_map:
        node_type_map[k] = tuple(node_type_map[k])
    for k in edge_type_map:
        edge_type_map[k] = tuple(edge_type_map[k])
    
    # node type encoding
    for t in node_type_map.values():
        if t not in node_type_dict:
            node_type_dict[t] = node_type_cnt
            node_type_cnt += 1
    
    # edge type encoding
    for edge, edge_type in edge_type_map.items():
        if edge_type not in edge_type_dict:
            edge_type_dict[edge_type] = edge_type_cnt
            edge_type_cnt += 1
        lines.append([edge[0], edge[1], node_type_map[edge[0]], node_type_map[edge[1]], edge_type])

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []
    for l in lines:
        src, dst, src_type, dst_type, edge_type = l
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        # node encoding, add nodes
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        # add edges
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)

    return node_map, g, node_list


def map_or_add(m, k, v):
    if k in m:
        m[k].add(v)
    else:
        m[k] = {v}


def preprocess_dataset():
    """get the mapping relation between id and node type, id and node name; extract node/edge information

    Args:
        dataset (_type_): _description_
    """
    id_nodetype_map = {}
    id_nodename_map = {}

    with open(SRC, 'r') as f:
        for line in tqdm(f):
            # Some filtering for better FPR/less Kerb noise
            if 'NTLM' not in line.upper():
                continue

            #0: ts, 1: src_u, 2: dest_u, 3: src_c, 4: dest_c, 5:auth_type, 6: logon_type, 7: auth_orientation, 8: success/failure
            tokens = line.strip().split(',')
            src_com, dst_com = tokens[3], tokens[4]
            # src_user, dst_user = tokens[1], tokens[2]
            # TODO: same node type
            id_nodetype_map[src_com] = 'com'
            id_nodetype_map[dst_com] = 'com'
            id_nodename_map[src_com] = src_com
            id_nodename_map[dst_com] = dst_com
    
    train_f = open('../data/lanl/train.txt', 'w')
    test_f = open('../data/lanl/test.txt', 'w')
    with open(SRC, 'r') as f:
        for line in tqdm(f):
            if 'NTLM' not in line.upper():
                continue

            tokens = line.strip().split(',')
            # TODO: edge type
            edgeType = '_'.join([tokens[1], tokens[2], tokens[8]])
            timestamp = int(tokens[0])
            srcId = tokens[3]
            srcType = id_nodetype_map[srcId]
            dstId = tokens[4]
            dstType = id_nodetype_map[dstId]
            this_edge = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId) + '\t' + str(
                            dstType) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'

            if timestamp < DATE_OF_EVIL_LANL:
                train_f.write(this_edge)
            elif timestamp < END_TEST_TIME:
                test_f.write(this_edge)
            else:
                break

    train_f.close()
    test_f.close()

    if len(id_nodename_map) != 0:
        fw = open('../data/lanl/names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open('../data/lanl/types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)


def load_malicious_entities():
    """load malicious enities

    Returns:
        _type_: {src_com_ts, dst_comm_ts}
    """
    with open(RED, 'r') as f:
        red_events = f.read().split()
    # red_events = red_events[1:]

    malicious_entities = set()
    for event in red_events:
        tokens = event.strip().split(',') # time,user@domain, source computer,destination computer
        src_com = tokens[2]
        dst_com = tokens[3]
        malicious_entities.add(src_com)
        malicious_entities.add(dst_com)

    return malicious_entities


def load_malicous_edges():
    with open(RED, 'r') as f:
        red_events = f.read().split()
    malicious_edges = set()
    for event in red_events:
        tokens = event.strip().split(',')
        malicious_edges.add((tokens[2], tokens[3]))
    return malicious_edges


def read_graphs(dataset):
    # load malicious enities
    # TODO
    print('load malicous')
    malicious_entities = load_malicious_entities()
    malicious_edges = load_malicous_edges()

    # get mapping relationships, node/edge information
    print('preprocess')
    preprocess_dataset()

    # get graphs
    print('construct graph')

    train_gs = []
    train_path = '../data/lanl/train.txt'
    _, train_g, _ = read_single_graph(train_path)
    train_gs.append(train_g)

    test_gs = []
    test_path = '../data/lanl/test.txt'
    test_node_map, test_g, _ = read_single_graph(test_path)
    test_gs.append(test_g)

    print(len(node_type_dict), len(edge_type_dict))

    if os.path.exists('../data/{}/names.json'.format(dataset)) and os.path.exists('../data/{}/types.json'.format(dataset)):
        with open('../data/{}/names.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)
        with open('../data/{}/types.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []

        # get the final malicious entities in test data
        for e in malicious_entities:
            if e in test_node_map:
                final_malicious_entities.append(test_node_map[e])
                malicious_names.append(id_nodename_map[e])
                f.write('{}\t{}\n'.format(e, id_nodename_map[e]))

        # TODO
        final_malicious_edges = []
        for edge in malicious_edges:
            src, dst = edge[0], edge[1]
            if src in test_node_map and dst in test_node_map:
                final_malicious_edges.append((test_node_map[src], test_node_map[dst]))
        with open('../data/lanl/malicious_edges.pkl', 'wb') as f:
            pkl.dump(final_malicious_edges, f)

    pkl.dump((final_malicious_entities, malicious_names), open('../data/{}/malicious.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(train_g) for train_g in train_gs], open('../data/{}/train.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], open('../data/{}/test.pkl'.format(dataset), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="lanl")
    args = parser.parse_args()
    if args.dataset not in ['lanl']:
        raise NotImplementedError
    read_graphs(args.dataset)
