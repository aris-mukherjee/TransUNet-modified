from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch

def get_params():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    net = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    net.load_state_dict(torch.load('/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/TU_NO_QKV_best_val_loss_seed1234.pth'))

    return sum(p.numel() for p in net.parameters())




if __name__ == "__main__":
    size = get_params()
    print(f'Number of parameters: {size}')