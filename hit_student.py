if __name__ == "__main__":
    import ipynbname
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from torch import nn

from src.models.HIT.backbone import build_backbone
from src.models.HIT.head import build_box_head
from src.models.HIT.neck import build_neck
from src.utils.utilities import imshow, _center_crop, _imread_rgb, _prep_image, imshow_np
import cv2
import numpy as np

class VT(nn.Module):
    def __init__(self, cfg):
        super(VT, self).__init__()
        head_type=cfg.MODEL.HEAD_TYPE
        neck_type=cfg.MODEL.NECK.TYPE
        self.backbone = build_backbone(cfg)
        #self.box_head = build_box_head(cfg)
        self.bottleneck = build_neck(cfg, self.backbone.num_channels, self.backbone.body.num_patches_search, self.backbone.body.embed_dim_list)
        #self.backbone = backbone
        self.num_patch_x = self.backbone.body.num_patches_search
        self.num_patch_z = self.backbone.body.num_patches_template
        self.neck_type = neck_type
        if neck_type in ['UPSAMPLE', 'FB', 'MAXF', 'MAXMINF', 'MAXMIDF', 'MINMIDF', 'MIDF', 'MINF']:
            self.num_patch_x = self.backbone.body.num_patches_search * ((self.bottleneck.stride_total) ** 2)
        self.side_fx = int(self.num_patch_x ** 0.5)
        self.side_fz = int(self.num_patch_z ** 0.5)
        #self.bottleneck = bottleneck
        # self.box_head = box_head
        channel = 256
        groupchannel = 32
        self.box_head = nn.Sequential(  # TODO: try add CBAM modules
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
        )
        self.cls2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s = int(20)
            self.feat_len_s = int(20 ** 2)

    def forward(self, template, search):
        # run the backbone
        img_list = [search, template]
        #print(search.shape, template.shape)
        xz = self.backbone(img_list)  # BxCxHxW
        # for el in xz:
        #     print(el.shape)
        if self.neck_type in ['FB', 'MAXF', 'MAXMINF', 'MAXMIDF', 'MINMIDF', 'MIDF', 'MINF']:
            xz_mem = self.bottleneck(xz)
        else:
            xz_mem = xz[-1].permute(1, 0, 2)
            xz_mem = self.bottleneck(xz_mem)

        output_embed = xz_mem[0:1, :, :].unsqueeze(-2)
        # print(xz_mem.shape)
        x_mem = xz_mem[1:1 + self.num_patch_x]
        # print(x_mem.shape)
        # adjust shape
        enc_opt = x_mem[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # print(enc_opt.shape)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # print(dec_opt.shape)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        # print(att.shape)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        return self.cls2(self.box_head(opt_feat)).view(-1, 400)

    def forward_sat(self, search):
        #print(self.backbone)
        return self.backbone.body.patch_embed_sat(search)



def build_model(cfg):
    model = VT(cfg)
    checkpoint_name = '/home/ilya/Code/VisNav/visnav_alg/MCVisLoc/ckpts/student_0916.pth'
    state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
    #print(state_dict.keys())
    model_state_dict = model.state_dict()
    filtered_state_dict = {k.replace('net.', ''): v for k, v in state_dict.items() if
                           k.replace('net.', '') in model_state_dict and v.size() == model_state_dict[
                               k.replace('net.', '')].size()}
    print(len(filtered_state_dict.keys()))
    model.load_state_dict(filtered_state_dict, strict=True)
    return model


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    import ipynbname
    import rootutils
    import os
    print(__file__)
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    config_dir = '/home/ilya/Code/VisNav/visnav_alg/MCVisLoc/configs'
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="hit_student.yaml")
    cfg = cfg.net.cfg

    model = build_model(cfg)
    model.eval()

    res = model(torch.rand(4, 3, 348, 348), model.forward_sat(torch.rand(4, 3, 640, 640)))
    print(res.shape)
    #
    # import timeit
    #
    # start = timeit.default_timer()
    # model.cuda()
    # for i in range(100):
    #     res = model(torch.rand(4, 3, 348, 348).cuda(), torch.rand(4, 3, 640, 640).cuda())
    # print(timeit.default_timer() - start)


def get_encoder(model):
    def encode(path):
        with torch.no_grad():
            rgb = _imread_rgb(path)
            img_chw = _prep_image(
                    rgb,
                    out_size=640,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    center_crop=None,
                    do_normalize=True,
                )
            #imshow(torch.from_numpy(img_chw))
            return model.forward_sat(torch.from_numpy(img_chw).cuda().unsqueeze(0)).cpu().numpy()
    return encode


def get_model_forward(model, tau=1, **kwargs):
    def hit_forward(uav_meta, sat_meta):
        model_res = model(torch.from_numpy(uav_meta.image).to(torch.float32).cuda().unsqueeze(0),
                          torch.from_numpy(sat_meta.data).to(torch.float32).cuda().unsqueeze(0))
        return torch.nn.functional.softmax(tau * model_res, dim=1).view(20, 20).detach().cpu().numpy()

    return hit_forward

# def get_model_forward(model, **kwargs):
#     def hit_forward(uav_meta, sat_meta):
#         model_res = model(torch.from_numpy(uav_meta.image).to(torch.float32).cuda().unsqueeze(0),
#                           torch.from_numpy(sat_meta.data).to(torch.float32).cuda().unsqueeze(0))
#         return model_res.view(20, 20).detach().cpu().numpy()
#
#     return hit_forward

#
# def gauss_mix_kernel(k=7, *, sigma_peak=0.7, sigma_tail=1.0, alpha_peak=0.45):
#     ax = np.arange(-(k//2), k//2 + 1, dtype=np.float32)
#     X, Y = np.meshgrid(ax, ax)
#     R2 = X**2 + Y**2
#     g1 = np.exp(-R2 / (2 * sigma_peak**2))          # узкий пик
#     g2 = np.exp(-R2 / (2 * sigma_tail**2))          # широкие хвосты
#     ker = alpha_peak * g1 + (1.0 - alpha_peak) * g2
#     s = ker.sum()
#     return (ker / s) if s > 0 else ker
#
# def get_model_forward(model, *,
#                       blur: str | None = None,   # 'gaussian' | 'box' | 'gauss_mix' | None
#                       sigma: float = 1.0,        # базовый σ в «клетках»
#                       ksize: int | None = None,  # нечётный размер ядра; если None — по sigma
#                       border: str = "reflect",   # 'reflect' | 'replicate' | 'constant' | 'wrap'
#                       normalize: bool = True,
#                       tau: float = 5.0,
#                       prune_q=0.05):
#     _BORDER = {
#         "reflect":  cv2.BORDER_REFLECT_101,
#         "replicate": cv2.BORDER_REPLICATE,
#         "constant": cv2.BORDER_CONSTANT,
#         "wrap":     cv2.BORDER_WRAP,
#     }
#     def _auto_ksize(sig: float) -> int:
#         k = int(2 * round(3 * sig) + 1)   # ~±3σ и нечётный
#         return 19#max(3, min(k, 19))         # для 20×20 нет смысла >19
#
#     def hit_forward(uav_meta, sat_meta):
#         with torch.no_grad():
#             logits = model(
#                 torch.from_numpy(uav_meta.image).to(torch.float32).cuda().unsqueeze(0),
#                 torch.from_numpy(sat_meta.data).to(torch.float32).cuda().unsqueeze(0)
#             )
#         heat = torch.softmax(tau * logits, dim=1)\
#                  .view(20, 20).detach().cpu().numpy().astype(np.float32)
#
#         if blur:
#             k = (ksize if ksize is not None else _auto_ksize(sigma))
#             if k % 2 == 0: k += 1
#             bt = _BORDER.get(border, cv2.BORDER_REFLECT_101)
#
#             if blur == "gaussian":
#                 heat = cv2.GaussianBlur(heat, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=bt)
#             elif blur == "box":
#                 heat = cv2.blur(heat, (k, k), borderType=bt)
#             elif blur == "gauss_mix":
#                 ker = gauss_mix_kernel(
#                     k=k,
#                     sigma_peak=max(1e-3, 0.45 * float(sigma)),
#                     sigma_tail=max(1e-3, 3.0 * float(sigma)),
#                     alpha_peak=0.9
#                 )
#                 heat = cv2.filter2D(heat, -1, ker, borderType=bt)
#             else:
#                 return heat
#
#
#         if prune_q is not None and 0.0 < prune_q < 1.0:
#             flat = heat.ravel()
#             n = flat.size
#             kq = int(np.floor(prune_q * (n - 1)))
#             thr = np.partition(flat, kq)[kq]  # q-квантиль без полной сортировки
#             np.maximum(heat, thr, out=heat)
#
#
#         if normalize:
#             heat = np.clip(heat, 0, None)
#             s = float(heat.sum())
#             if s > 0: heat /= s
#
#         return heat
#
#     return hit_forward