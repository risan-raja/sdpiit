from pyvis.network import Network
from collections import defaultdict


def gen_adj_list(threshold, data=data):
    adj_list = defaultdict(list)
    corr_mat = data.corr()
    for c in data.columns:
        # adj_list[c].append()
        e = list(corr_mat[corr_mat[c] >= threshold][c].index)
        e.remove(c)
        adj_list[c] = e
        # print(e)
        # break
    return adj_list


adj_list = gen_adj_list(0.9, data=data)


# Adding Nodes
def addfnetnodes(nets, data):
    for c, f in enumerate(data.columns):
        colr = None
        if "nom" in f:
            colr = "#008000"
        elif "ord" in f:
            colr = "#800080"
        elif "bin" in f:
            colr = "#FFEC00"
        elif "label" in f:
            colr = "#D800FF"
        else:
            colr = "#FF0083"
        nets.add_node(c, label=f, shape="circle", color=colr)
    return nets


def addfnetedges(nt: Network, nnli):
    global adj_list
    corr_mat = data.corr()
    for nd in adj_list:
        if len(adj_list[nd]) > 0:
            for conn in adj_list[nd]:
                nt.add_edge(nnli[nd], nnli[conn], weight=corr_mat.loc[nd, conn])
    return nt


def draw_feature_network():
    global data, adj_list
    fnet = Network(
        notebook=False,
        heading="Clustering of Features",
        height="1080px",
        width="2000px",
        font_color="#46ff00",
        bgcolor="#9ed6c8",
    )
    fnet = addfnetnodes(fnet, data)
    nnli = {f["label"]: f["id"] for f in fnet.nodes}  #  net_node_label_ids
    fnet = addfnetedges(fnet, nnli)
    fnet.toggle_physics(True)
    fnet.set_options(
        """
        const options = {
        "edges": {
        "color": {
          "inherit": true
        },
        "selfReferenceSize": null,
        "selfReference": {
          "angle": 0.7853981633974483,
          "renderBehindTheNode": false
        },
        "smooth": {
          "type": "vertical",
          "forceDirection": "vertical"
        }
        },
        "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": true
        },
        "manipulation": {
        "enabled": true,
        "initiallyActive": true
        },
        "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0,
          "springLength": 440,
          "springConstant": 0,
          "nodeDistance": 90,
          "damping": 1,
          "avoidOverlap": 0.46
        },
        "maxVelocity": 1,
        "minVelocity": 0.01,
        "solver": "hierarchicalRepulsion",
        "timestep": 0.54,
        "wind": {
          "x": 0
        }
        }
        }
        """
    )
    return fnet


fnet = draw_feature_network()
# fnet.show_buttons()

fnet.show("../reports/feature_network.html")
