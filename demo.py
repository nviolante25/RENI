import torch
from torchvision.transforms import ToPILImage

from src.models.RENI import RENIVariationalAutoDecoder
from src.utils.utils import get_directions
from src.utils.custom_transforms import UnMinMaxNormlise
from src.utils.utils import get_directions, sRGB


if __name__ == "__main__":
    device = "cuda"
    chkpt_path = './models/RENI_Pretrained_Weights/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_decoder_epoch=1579.ckpt'
    chkpt = torch.load(chkpt_path, map_location=device)
    config = chkpt['hyper_parameters']['config']

    num_samples = 8
    model = RENIVariationalAutoDecoder(
        num_samples,
        config.RENI.LATENT_DIMENSION,
        config.RENI.EQUIVARIANCE,
        config.RENI.HIDDEN_FEATURES,
        config.RENI.HIDDEN_LAYERS,
        config.RENI.OUT_FEATURES,
        config.RENI.LAST_LAYER_LINEAR,
        config.RENI.OUTPUT_ACTIVATION,
        config.RENI.FIRST_OMEGA_0,
        config.RENI.HIDDEN_OMEGA_0,
        fixed_decoder=True,
    )
    model.load_state_dict(chkpt['state_dict'])
    model.to(device)
    model.eval()
    H, W = 256, 512
    model.init_latent_codes(num_samples, config.RENI.LATENT_DIMENSION, False) # TODO" maybe there's a more elegant way to do this

    Z = model.mu.cuda()
    directions = get_directions(W)
    D = directions.repeat(Z.shape[0], 1, 1).cuda()
    output = model(Z, D).view(-1, H, W, 3)
    
    transforms = config.DATASET[config.DATASET.NAME].TRANSFORMS
    minmax = transforms[0][1]
    unnormalise = UnMinMaxNormlise(minmax)

    img = sRGB(unnormalise(output))
    ToPILImage()(torch.cat([a for a in img], 0).permute(2, 0, 1)).show()