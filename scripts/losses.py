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


@torch.no_grad()
def reverse_diffusion_sample(model, x, t, i, closed_form_results):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas = closed_form_results["betas"]
    sqrt_one_minus_alphas_cumprod = closed_form_results["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas = closed_form_results["sqrt_recip_alphas"]
    posterior_variance = closed_form_results["posterior_variance"]

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    # Call model (current image - noise prediction)
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if i == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(config, model, shape, cf_calculations):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, config.T)):
        img = reverse_diffusion_sample(
            model,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
            cf_calculations,
        )
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(config, model, image_size, cf_calculations, batch_size=16, channels=3):
    return p_sample_loop(
        config,
        model,
        shape=(batch_size, channels, *image_size),
        cf_calculations=cf_calculations,
    )
