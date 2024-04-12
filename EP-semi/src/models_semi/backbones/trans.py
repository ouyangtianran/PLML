import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        num_layers,
        nhead,
        d_model,
        dim_feedforward,
        device,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
    ):
        super().__init__()
        self.ways = ways
        self.shot = shot

        self.cls_type = cls_type
        self.pos_type = pos_type
        self.agg_method = agg_method

        if self.cls_type == "cls_learn":
            self.cls_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            )
        elif self.cls_type == "rand_const":
            self.cls_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            ).requires_grad_(False)

        if self.pos_type == "pos_learn":
            self.pos_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            )
        elif self.pos_type == "rand_const":
            self.pos_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            ).requires_grad_(False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.device = device

    def forward(self, x):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        n_arng = torch.arange(ways, device=self.device)

        # Concatenate cls tokens with support embeddings
        if self.cls_type in ["cls_learn", "rand_const"]:
            cls_tokens = self.cls_embeddings(n_arng)  # (ways, dim)
        # elif self.cls_type == "proto":
        #     cls_tokens = gen_prototypes(x, ways, shot, self.agg_method)  # (ways, dim)
        else:
            raise NotImplementedError

        cls_sup_embeds = torch.cat((cls_tokens, x), dim=0)  # (ways*(shot+1), dim)
        cls_sup_embeds = torch.unsqueeze(
            cls_sup_embeds, dim=1
        )  # (ways*(shot+1), BS, dim)

        # Position embeddings based on class ID
        pos_idx = torch.cat((n_arng, torch.repeat_interleave(n_arng, shot)))
        pos_tokens = torch.unsqueeze(
            self.pos_embeddings(pos_idx), dim=1
        )  # (ways*(shot+1), BS, dim)

        # Inputs combined with position encoding
        transformer_input = cls_sup_embeds + pos_tokens

        return self.encoder(transformer_input)