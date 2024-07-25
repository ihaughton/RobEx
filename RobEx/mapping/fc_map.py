import torch

def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if type(m) == torch.nn.Linear:
        init_fn(m.weight)


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )


class OccupancyMap(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        out_scale,
        hidden_size=256,
        do_color=False,
        do_semantics=False,
        n_classes=None,
        hidden_layers_block=1,
    ):
        super(OccupancyMap, self).__init__()
        self.do_color = do_color
        self.do_semantics = do_semantics
        self.out_scale = out_scale

        self.in_layer = fc_block(embedding_size, hidden_size)

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)

        self.cat_layer = fc_block(
            hidden_size + embedding_size, hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        if self.do_color:
            self.out_color = torch.nn.Linear(hidden_size, 3)
        if self.do_semantics:
            self.out_sem = torch.nn.Linear(hidden_size, n_classes)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x,
                noise_std=None,
                do_alpha=True,
                do_color=False,
                do_sem=False,
                scale=50,
                return_features=False):
        fc1 = self.in_layer(x)
        fc2 = self.mid1(fc1)
        fc2_x = torch.cat((fc2, x), dim=-1)
        fc3 = self.cat_layer(fc2_x)
        fc4 = self.mid2(fc3)

        alpha = None
        if do_alpha:
            raw = self.out_alpha(fc4)
            if noise_std is not None:
                noise = torch.randn(raw.shape, device=x.device) * noise_std
                raw = raw + noise
            alpha = self.relu(raw) * self.out_scale

        color = None
        if self.do_color and do_color:
            raw_color = self.out_color(fc4)
            ## HYPERPARAM
            color = self.sigmoid(raw_color) #*0.1)

        sem = None
        if self.do_semantics and do_sem:
            sem = self.out_sem(fc4)

        if return_features:
            return alpha, color, sem, fc4

        return alpha, color, sem

    def forward_features_only(self, x):
        fc1 = self.in_layer(x)
        fc2 = self.mid1(fc1)
        fc2_x = torch.cat((fc2, x), dim=-1)
        fc3 = self.cat_layer(fc2_x)
        fc4 = self.mid2(fc3)

        return fc4


class OutputDecoder(torch.nn.Module):
    def __init__(
        self,
        feature_size,
        do_color=False,
    ):
        super(OutputDecoder, self).__init__()
        self.do_color = do_color

        out_size1 = feature_size // 2
        out_size2 = out_size1 // 2

        self.layer1 = fc_block(feature_size, out_size1)
        self.layer2 = fc_block(out_size1, out_size2)
        self.out_layer = torch.nn.Linear(out_size2, 1)
        if do_color:
            self.out_layer2 = torch.nn.Linear(out_size2, 3)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self,
                feature,
                noise_std=None,
                scale=10,
                do_alpha=True,
                do_color=False
                ):
        fc1 = self.layer1(feature)
        fc2 = self.layer2(fc1)

        alpha = None
        if do_alpha:
            raw = self.out_layer(fc2)

            if noise_std is not None:
                noise = torch.randn(
                    raw.shape, device=feature.device) * noise_std
                raw = raw + noise

            # Scaling to behave better with network init.
            alpha = self.relu(raw) * scale

        color = None
        if self.do_color and do_color:
            raw_color = self.out_layer2(fc2)
            color = self.sigmoid(raw_color)

        return alpha, color


class FeatureDecoder(torch.nn.Module):
    def __init__(
        self,
        in_f,
        hidden_size,
        out_f,
        n_hidden=1
    ):
        super(FeatureDecoder, self).__init__()

        self.in_layer = fc_block(in_f, hidden_size)
        hidden = [fc_block(hidden_size, hidden_size)
                  for _ in range(n_hidden)]
        self.hidden_layers = torch.nn.Sequential(*hidden)
        self.out_layer = fc_block(hidden_size, out_f)

    def forward(self, x, feature=None):
        if feature is not None:
            f = torch.cat((feature, x), dim=-1)
        else:
            f = x
        fc1 = self.in_layer(f)
        fc2 = self.hidden_layers(fc1)
        fc3 = self.out_layer(fc2)

        return fc3


class OccupancyMapTwoStage(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size=256,
        do_color=False
    ):
        super(OccupancyMapTwoStage, self).__init__()

        out_size1 = hidden_size // 2
        out_size2 = out_size1 // 2
        n_hidden1 = 1
        n_hidden2 = 2

        self.f_dec1 = FeatureDecoder(
            in_f=embedding_size,
            hidden_size=hidden_size,
            out_f=out_size1,
            n_hidden=n_hidden1)

        self.f_dec2 = FeatureDecoder(
            in_f=out_size1 + embedding_size,
            hidden_size=out_size1,
            out_f=out_size2,
            n_hidden=n_hidden2)

        self.o_dec = OutputDecoder(out_size2, do_color=do_color)

    def forward(self, x, noise_std=None, do_alpha=True, do_color=False, do_sem=False):
        f1 = self.f_dec1(x)
        f2 = self.f_dec2(x, f1)

        alpha, color = self.o_dec(
            f2, noise_std, do_alpha=do_alpha, do_color=do_color)

        return alpha, color, None
