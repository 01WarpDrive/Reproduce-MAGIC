import tqdm

import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


from utils.utils import create_activation, create_norm


class GAT(nn.Module):
    def __init__(self,
                 n_dim,
                 e_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_heads_out = nhead_out
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(GATConv(
                n_dim, e_dim, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                n_dim, e_dim, num_hidden, nhead,
                feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # hidden layers

            for _ in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * nhead, e_dim, num_hidden, nhead,
                    feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))

            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * nhead, e_dim, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))
        self.head = nn.Identity()
        
    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def inference(self, g, x, batch_size, device, emb=False):
        """
        TODO
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        num_heads = self.num_heads
        num_heads_out = self.num_heads_out
        for l, layer in enumerate(self.gat_layers):
            if l < self.num_layers - 1:
                y = torch.zeros(g.num_nodes(), self.num_hidden * num_heads if l != len(self.gat_layers) - 1 else self.num_classes)
            else:
                if emb == False:
                    y = torch.zeros(g.num_nodes(), self.num_hidden if l != len(self.gat_layers) - 1 else self.num_classes)
                else:
                    y = torch.zeros(g.num_nodes(), self.out_dim * num_heads_out)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                    g,
                    torch.arange(g.num_nodes()),
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=8)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                h = x[input_nodes].to(device)
                if l < self.num_layers - 1:
                    h = layer(block, h)
                else:
                    h = layer(block, h)

                if l == len(self.gat_layers) - 1 and (emb == False):
                    h = self.head(h)
                y[output_nodes] = h.cpu()
            x = y
        return y

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.is_pretraining = False
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)



class GATConv(nn.Module):
    def __init__(self,
                 in_dim,
                 e_dim,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_dim)
        self.edge_feat = e_dim
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._concat_out = concat_out

        if isinstance(in_dim, tuple):
            self.fc_node_embedding = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc_node_embedding = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.edge_fc = nn.Linear(self.edge_feat, out_feats * num_heads, bias=False)
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = None
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
    
        self.norm = norm
        if norm is not None:
            self.norm = create_norm(norm)(num_heads * out_feats)
        
        # self.set_allow_zero_in_degree(False)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        edge_feature = graph.edata['attr']
        with graph.local_scope():
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                # h_dst = self.feat_drop(feat[1])
                h_dst = feat[1]

                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            edge_prefix_shape = edge_feature.shape[:-1]
            eh = (feat_src * self.attn_h).sum(-1).unsqueeze(-1)
            et = (feat_dst * self.attn_t).sum(-1).unsqueeze(-1)

            graph.srcdata.update({'hs': feat_src, 'eh': eh})
            graph.dstdata.update({'et': et})

            feat_edge = self.edge_fc(edge_feature).view(
                *edge_prefix_shape, self._num_heads, self._out_feats)
            ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)

            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))
            graph.apply_edges(fn.e_add_v('ee', 'et', 'e'))

            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('hs', 'a', 'm'),
                             fn.sum('m', 'hs'))
            
            rst = graph.dstdata['hs'].view(-1, self._num_heads, self._out_feats)

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            if self._concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
