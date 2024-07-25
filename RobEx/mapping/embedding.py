import torch
import numpy as np


def positional_encoding(
    tensor,
    B_layer=None,
    num_encoding_functions=6,
    scale=10.,
    uniform_dirs=False
):
    if B_layer is not None:
        embedding = B_layer(tensor / scale)

        if uniform_dirs:
            embedding = embedding[..., None, :].repeat(1, 1, 9, 1)
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                9 - 1,
                9,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            embedding =  frequency_bands[None, None, :, None] * embedding
            embedding = embedding.view(
                embedding.shape[0], embedding.shape[1], 30 * 9)
            embedding = torch.sin(embedding)

        embedding = torch.sin(embedding)
    else:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        n_repeat = num_encoding_functions * 2 + 1
        embedding = tensor[..., None, :].repeat(1, 1, n_repeat, 1) / scale
        even_idx = np.arange(1, num_encoding_functions + 1) * 2
        odd_idx = even_idx - 1

        frequency_bands = frequency_bands[None, None, :, None]

        embedding[:, :, even_idx, :] = torch.cos(
            frequency_bands * embedding[:, :, even_idx, :])
        embedding[:, :, odd_idx, :] = torch.sin(
            frequency_bands * embedding[:, :, odd_idx, :])

        n_dim = tensor.shape[-1]
        embedding = embedding.view(
            embedding.shape[0], embedding.shape[1], n_repeat * n_dim)

    return embedding
