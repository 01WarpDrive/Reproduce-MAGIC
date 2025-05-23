# gmae2
from .gat import GAT
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random


def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='batchnorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    return model


class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        self._remask_method = "random"
        self._num_remasking = 3
        self._remask_rate = 0.5
        self._momentum = 0.996

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        # build encoder
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=n_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=True
        )

        # build decoder for attribute prediction
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            num_hidden=dec_num_hidden,
            out_dim=n_dim,
            num_layers=1,
            nhead=n_heads,
            nhead_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)
        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        # TODO
        dimension = 256
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, dimension),
            nn.PReLU(),
            nn.Linear(dimension, hidden_dim),
        )
        self.projector_ema = nn.Sequential(
            nn.Linear(hidden_dim, dimension),
            nn.PReLU(),
            nn.Linear(dimension, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
        self.encoder_ema = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=n_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=True
        )
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()
        
        # self.print_num_parameters()


    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if  p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if  p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        # TODO
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)

        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)

        # ---- Latent representation prediction ----
        with torch.no_grad():
            latent_target = self.encoder_ema(g, pre_use_x,)
            latent_target = self.projector_ema(latent_target[keep_nodes])

        latent_pred = self.projector(enc_rep[keep_nodes])
        latent_pred = self.predictor(latent_pred)
        loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        # Feature Reconstruction
        enc_rep = torch.cat(all_hidden, dim=1)
        origin_rep = self.encoder_to_decoder(enc_rep)

        if self._remask_method == "random":
            loss_rec_all = 0
            for _ in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, self._remask_rate)
                recon = self.decoder(pre_use_g, rep)
                x_init = pre_use_x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(g, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = pre_use_x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        # Structural Reconstruction
        threshold = min(10000, g.num_nodes())

        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
            g.device)
        
        loss = loss_latent + loss_rec + self.recon_loss(y_pred, y)

        self.ema_update()

        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def embed(self, g):
        # TODO
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    
    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict
    
    def random_remask(self,g,rep,remask_rate=0.5):
        
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep
