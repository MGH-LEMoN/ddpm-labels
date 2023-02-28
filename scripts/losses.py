import beta_schedule
import torch
import torch.nn.functional as F


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, closed_form_results, noise=None, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    sqrt_alphas_cumprod = closed_form_results["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = closed_form_results["sqrt_one_minus_alphas_cumprod"]

    if noise is None:
        noise = torch.randn_like(x_0, dtype=torch.float32)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def p_losses(
    denoise_model, x_start, t, closed_form_results, noise=None, loss_type="l1"
):
    if noise is None:
        noise = torch.randn_like(x_start, dtype=torch.float32)

    x_noisy, _ = forward_diffusion_sample(
        x_0=x_start,
        t=t,
        closed_form_results=closed_form_results,
        noise=noise,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
