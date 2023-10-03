
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.arch_util as arch_util
try:
    from models.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation

# import functools
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import arch_util as arch_util
# try:
#     from dcn.deform_conv import ModulatedDeformConvPack as DCN
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')
# from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2 
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1([fea1[2], L3_offset]))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1([fea1[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1([fea1[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2([fea2[2], L3_offset]))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2([fea2[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2([fea2[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        
        y = torch.cat(y, dim=1)
        return y


class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        try:
            L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        except RuntimeError:
            L3_fea = L3_fea.view(B, N, -1, L3_fea.shape[2], L3_fea.shape[3])

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(), L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(), L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea) # [B, N, C, H, W]
        return fusion_fea

class ProgressiveFusion3F(nn.Module): #############################################################################################
    def __init__(self, nf):
        super(ProgressiveFusion3F, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*3, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        arch_util.initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        x0_in ,x1_in, x2_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:]
        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x_fusion = self.fusion(torch.cat([x0,x1,x2],1))
        x0 = self.conv_decoder(torch.cat([x0,x_fusion],1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1)],1)
        return x_out

class PatchCorrespondenceAggregation(nn.Module):
    def __init__(self, nf=64, nbr=4, n_group=8, kernels=3, patches=11, cor_ksize=3):
        super(PatchCorrespondenceAggregation, self).__init__()
        self.nbr = nbr
        self.cas_k = 3
        self.k = kernels
        self.g = n_group

        self.conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mask = nn.Conv2d(nf, self.g * self.k ** 2, self.k, 1, (self.k-1)//2)
        self.nn_conv = nn.Conv2d(nf * self.nbr, nf, 3, 1, 1, bias=True)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.patch_size = patches
        self.cor_k = cor_ksize
        self.padding = (self.cor_k - 1) // 2
        self.pad_size = self.padding + (11 - 1) // 2
        self.add_num = 2 * self.pad_size - self.cor_k + 1
        self.corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size,
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)
        self.conv_agg = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        B, C, H, W = nbr_fea_l.size()
        w = torch.cat([nbr_fea_l, ref_fea_l], dim=1)
        w = self.conv2(self.conv1(w))
        mask = self.mask(w).view(B, self.g, 1, self.k ** 2, H, W)
        # corr: B, (2 * dis + 1) ** 2, H, W
        norm_ref_fea = F.normalize(ref_fea_l, dim=1)
        norm_nbr_fea = F.normalize(nbr_fea_l, dim=1)
        corr = self.corr(norm_ref_fea, norm_nbr_fea).view(B, -1, H, W)
        # corr_ind: B, H, W
        _, corr_ind = torch.topk(corr, self.nbr, dim=1)
        corr_ind = corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        ind_row_add = corr_ind // self.patch_size * (W + self.add_num)
        ind_col_add = corr_ind % self.patch_size
        corr_ind = ind_row_add + ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        lt_ind = y * (W + self.add_num) + x
        lt_ind = lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        corr_ind = (corr_ind + lt_ind).view(-1)
        # nbr: B, 64 * k * k, (H + 2 * pad - k + 1) * (W + 2 * pad -k + 1)
        nbr = F.unfold(nbr_fea_l, self.cor_k, dilation=1, padding=self.pad_size, stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        # L: B * H * W * nbr, 64 * k * k
        L = nbr[ind_B, :, corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        L = self.nn_conv(L)
        L = L.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L = L.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L = self.relu((L * mask).sum(dim=3).view(B, C, H, W))
        L = self.conv_agg(torch.cat([L, ref_fea_l], dim=1))
        return L


class FeatureSelectionReduction(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionReduction, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=True)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class prealign(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        super(prealign, self).__init__()
        self.offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack_2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, angular, center):
        offset = torch.cat([angular, center], dim=1)
        offset = self.lrelu(self.offset_conv1_2(offset))
        offset = self.lrelu(self.offset_conv2_2(offset))
        fea = self.lrelu(self.dcnpack_2([angular, offset]))
        return fea

class LFSTVSR(nn.Module):
    def __init__(self):
        #  nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
        #          predeblur=False, HR_in=False, w_TSA=True
        super(LFSTVSR, self).__init__()
        self.nf = 64
        self.groups = 8
        self.front_RBs = 5
        self.back_RBs = 40
        self.upscale_factor = 4
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=self.nf)
        ProgressiveFusion3FModule = functools.partial(ProgressiveFusion3F, nf=self.nf)
        ###################################################################################################
        # feature extractor for center view and one/two/three views
        self.conv_first_center = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv_first_one = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv_first_two = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv_first_three = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)

        self.feature_extraction_center = arch_util.make_layer(ResidualBlock_noBN_f, self.front_RBs)
        self.feature_extraction_one = arch_util.make_layer(ResidualBlock_noBN_f, self.front_RBs)
        self.feature_extraction_two = arch_util.make_layer(ResidualBlock_noBN_f, self.front_RBs)
        self.feature_extraction_three = arch_util.make_layer(ResidualBlock_noBN_f, self.front_RBs)


        self.concat1 = nn.Conv2d(self.nf*25, self.nf, 1, 1, 0, bias=True)
        self.concat3 = nn.Conv2d(self.nf*25, self.nf, 1, 1, 0, bias=True)

        ##################################################################################################
        # center view feature pyramid generation
        self.fea_L2_conv1 = nn.Conv2d(self.nf, self.nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(self.nf, self.nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        ##################################################################################################
        # feature selection and reduction
        self.FeatureSelectionReduction_one1 = FeatureSelectionReduction(self.nf*8, self.nf)
        self.FeatureSelectionReduction_one3 = FeatureSelectionReduction(self.nf*8, self.nf)
        self.FeatureSelectionReduction_two1 = FeatureSelectionReduction(self.nf*8, self.nf)
        self.FeatureSelectionReduction_two3 = FeatureSelectionReduction(self.nf*8, self.nf)
        self.FeatureSelectionReduction_three1 = FeatureSelectionReduction(self.nf*8, self.nf)
        self.FeatureSelectionReduction_three3 = FeatureSelectionReduction(self.nf*8, self.nf)

        self.FeatureSelectionReduction1 = FeatureSelectionReduction(self.nf*2, self.nf)
        self.FeatureSelectionReduction2 = FeatureSelectionReduction(self.nf*2, self.nf)
        self.FeatureSelectionReduction3 = FeatureSelectionReduction(self.nf*2, self.nf)
        self.FeatureSelectionReduction4 = FeatureSelectionReduction(self.nf*2, self.nf)
        self.FeatureSelectionReduction5 = FeatureSelectionReduction(self.nf*2, self.nf)
        self.FeatureSelectionReduction6 = FeatureSelectionReduction(self.nf*2, self.nf)

        ##################################################################################################
        # feature correspondence aggregation
        self.PatchCorrespondenceAggregation_one1 = PatchCorrespondenceAggregation()
        self.PatchCorrespondenceAggregation_one3 = PatchCorrespondenceAggregation()
        self.PatchCorrespondenceAggregation_two1 = PatchCorrespondenceAggregation()
        self.PatchCorrespondenceAggregation_two3 = PatchCorrespondenceAggregation()
        self.PatchCorrespondenceAggregation_three1 = PatchCorrespondenceAggregation()
        self.PatchCorrespondenceAggregation_three3 = PatchCorrespondenceAggregation()
        ##################################################################################################
        self.one1_prealign1 = prealign()
        self.one1_prealign2 = prealign()
        self.one1_prealign3 = prealign()
        self.one1_prealign4 = prealign()
        self.one1_prealign5 = prealign()
        self.one1_prealign6 = prealign()
        self.one1_prealign7 = prealign()
        self.one1_prealign8 = prealign()
        self.two1_prealign1 = prealign()
        self.two1_prealign2 = prealign()
        self.two1_prealign3 = prealign()
        self.two1_prealign4 = prealign()
        self.two1_prealign5 = prealign()
        self.two1_prealign6 = prealign()
        self.two1_prealign7 = prealign()
        self.two1_prealign8 = prealign()
        self.three1_prealign1 = prealign()
        self.three1_prealign2 = prealign()
        self.three1_prealign3 = prealign()
        self.three1_prealign4 = prealign()
        self.three1_prealign5 = prealign()
        self.three1_prealign6 = prealign()
        self.three1_prealign7 = prealign()
        self.three1_prealign8 = prealign()
        self.one3_prealign1 = prealign()
        self.one3_prealign2 = prealign()
        self.one3_prealign3 = prealign()
        self.one3_prealign4 = prealign()
        self.one3_prealign5 = prealign()
        self.one3_prealign6 = prealign()
        self.one3_prealign7 = prealign()
        self.one3_prealign8 = prealign()
        self.two3_prealign1 = prealign()
        self.two3_prealign2 = prealign()
        self.two3_prealign3 = prealign()
        self.two3_prealign4 = prealign()
        self.two3_prealign5 = prealign()
        self.two3_prealign6 = prealign()
        self.two3_prealign7 = prealign()
        self.two3_prealign8 = prealign()
        self.three3_prealign1 = prealign()
        self.three3_prealign2 = prealign()
        self.three3_prealign3 = prealign()
        self.three3_prealign4 = prealign()
        self.three3_prealign5 = prealign()
        self.three3_prealign6 = prealign()
        self.three3_prealign7 = prealign()
        self.three3_prealign8 = prealign()



        self.concat1 = nn.Conv2d(self.nf*25, self.nf, 1, 1, 0, bias=True)
        self.concat3 = nn.Conv2d(self.nf*25, self.nf, 1, 1, 0, bias=True)
        ##################################################################################################
        # fusion
        self.fusion_one1 = FeatureSelectionReduction(self.nf*4, self.nf)
        self.fusion_one3 = FeatureSelectionReduction(self.nf*4, self.nf)

        ##################################################################################################
        # interpolation
        self.interpolation = Easy_PCD()

        ##################################################################################################
        # interpolation
        self.featureFusion = arch_util.make_layer(ProgressiveFusion3FModule, self.front_RBs)


        #### reconstruction
        self.recon_trunk1 = arch_util.make_layer(ResidualBlock_noBN_f, 10)
        self.recon_trunk2 = arch_util.make_layer(ResidualBlock_noBN_f, 10)
        self.recon_trunk3 = arch_util.make_layer(ResidualBlock_noBN_f, 10)
        self.recon_trunk4 = arch_util.make_layer(ResidualBlock_noBN_f, 10)
        #### upsampling
        self.upconv1 = nn.Conv2d(self.nf, self.nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.nf, 64 * 4, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(self.nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, centerView1, one1, two1, three1, centerView3, one3, two3, three3):
        B_one, N_one, C_one, H_one, W_one = one1.size()
        B_two, N_two, C_two, H_two, W_two = two1.size()
        B_three, N_three, C_three, H_three, W_three = three1.size()
        
        two1 = F.interpolate(two1.view(-1, C_one, H_one, W_one), scale_factor=0.5, mode="bilinear")
        three1 = F.interpolate(three1.view(-1, C_three, H_three, W_three), scale_factor=0.25, mode="bilinear")
        two3 = F.interpolate(two3.view(-1, C_one, H_one, W_one), scale_factor=0.5, mode="bilinear")
        three3 = F.interpolate(three3.view(-1, C_three, H_three, W_three), scale_factor=0.25, mode="bilinear")
        
        centerView1_L1_fea = self.feature_extraction_center(self.lrelu(self.conv_first_center(centerView1)))
        centerView3_L1_fea = self.feature_extraction_center(self.lrelu(self.conv_first_center(centerView3)))    # torch.Size([8, 64, 256, 104])

        # L2 and L3
        centerView1_L2_fea = self.lrelu(self.fea_L2_conv1(centerView1_L1_fea))
        centerView1_L2_fea = self.lrelu(self.fea_L2_conv2(centerView1_L2_fea))
        centerView1_L3_fea = self.lrelu(self.fea_L3_conv1(centerView1_L2_fea))
        centerView1_L3_fea = self.lrelu(self.fea_L3_conv2(centerView1_L3_fea))
        centerView3_L2_fea = self.lrelu(self.fea_L2_conv1(centerView3_L1_fea))
        centerView3_L2_fea = self.lrelu(self.fea_L2_conv2(centerView3_L2_fea))
        centerView3_L3_fea = self.lrelu(self.fea_L3_conv1(centerView3_L2_fea))
        centerView3_L3_fea = self.lrelu(self.fea_L3_conv2(centerView3_L3_fea))

        one1_fea = self.lrelu(self.conv_first_one(one1.view(-1, C_one, H_one, W_one)))
        one1_fea = self.feature_extraction_one(one1_fea)
        one3_fea = self.lrelu(self.conv_first_one(one3.view(-1, C_one, H_one, W_one)))
        one3_fea = self.feature_extraction_one(one3_fea)  # torch.Size([8, 64, 256, 104])

        two1_fea = self.lrelu(self.conv_first_two(two1.view(-1, C_one, H_one//2, W_one//2)))
        two1_fea = self.feature_extraction_two(two1_fea)
        two3_fea = self.lrelu(self.conv_first_two(two3.view(-1, C_one, H_one//2, W_one//2)))
        two3_fea = self.feature_extraction_two(two3_fea)  # torch.Size([8, 64, 128, 52])

        three1_fea = self.lrelu(self.conv_first_three(three1.view(-1, C_one, H_one//4, W_one//4)))
        three1_fea = self.feature_extraction_three(three1_fea)
        three3_fea = self.lrelu(self.conv_first_three(three3.view(-1, C_one, H_one//4, W_one//4)))
        three3_fea = self.feature_extraction_three(three3_fea)    # torch.Size([8, 64, 64, 26])

        one1_fea = one1_fea.view(B_one, 8, -1, H_one, W_one)
        one3_fea = one3_fea.view(B_one, 8, -1, H_one, W_one)
        two1_fea = two1_fea.view(B_two, 8, -1, H_one//2, W_one//2)
        two3_fea = two3_fea.view(B_two, 8, -1, H_one//2, W_one//2)
        three1_fea = three1_fea.view(B_three, 8, -1, H_one//4, W_one//4)
        three3_fea = three3_fea.view(B_three, 8, -1, H_one//4, W_one//4)

        #############################################################################
        one1_1 = self.one1_prealign1(one1_fea[:,0,:,:,:].contiguous(),centerView1_L1_fea)
        one1_2 = self.one1_prealign2(one1_fea[:,1,:,:,:].contiguous(),centerView1_L1_fea)
        one1_3 = self.one1_prealign3(one1_fea[:,2,:,:,:].contiguous(),centerView1_L1_fea)
        one1_4 = self.one1_prealign4(one1_fea[:,3,:,:,:].contiguous(),centerView1_L1_fea)
        one1_5 = self.one1_prealign5(one1_fea[:,4,:,:,:].contiguous(),centerView1_L1_fea)
        one1_6 = self.one1_prealign6(one1_fea[:,5,:,:,:].contiguous(),centerView1_L1_fea)
        one1_7 = self.one1_prealign7(one1_fea[:,6,:,:,:].contiguous(),centerView1_L1_fea)
        one1_8 = self.one1_prealign8(one1_fea[:,7,:,:,:].contiguous(),centerView1_L1_fea)
        two1_1 = self.two1_prealign1(two1_fea[:,0,:,:,:].contiguous(),centerView1_L2_fea)
        two1_2 = self.two1_prealign2(two1_fea[:,1,:,:,:].contiguous(),centerView1_L2_fea)
        two1_3 = self.two1_prealign3(two1_fea[:,2,:,:,:].contiguous(),centerView1_L2_fea)
        two1_4 = self.two1_prealign4(two1_fea[:,3,:,:,:].contiguous(),centerView1_L2_fea)
        two1_5 = self.two1_prealign5(two1_fea[:,4,:,:,:].contiguous(),centerView1_L2_fea)
        two1_6 = self.two1_prealign6(two1_fea[:,5,:,:,:].contiguous(),centerView1_L2_fea)
        two1_7 = self.two1_prealign7(two1_fea[:,6,:,:,:].contiguous(),centerView1_L2_fea)
        two1_8 = self.two1_prealign8(two1_fea[:,7,:,:,:].contiguous(),centerView1_L2_fea)
        three1_1 = self.three1_prealign1(three1_fea[:,0,:,:,:].contiguous(),centerView1_L3_fea)
        three1_2 = self.three1_prealign2(three1_fea[:,1,:,:,:].contiguous(),centerView1_L3_fea)
        three1_3 = self.three1_prealign3(three1_fea[:,2,:,:,:].contiguous(),centerView1_L3_fea)
        three1_4 = self.three1_prealign4(three1_fea[:,3,:,:,:].contiguous(),centerView1_L3_fea)
        three1_5 = self.three1_prealign5(three1_fea[:,4,:,:,:].contiguous(),centerView1_L3_fea)
        three1_6 = self.three1_prealign6(three1_fea[:,5,:,:,:].contiguous(),centerView1_L3_fea)
        three1_7 = self.three1_prealign7(three1_fea[:,6,:,:,:].contiguous(),centerView1_L3_fea)
        three1_8 = self.three1_prealign8(three1_fea[:,7,:,:,:].contiguous(),centerView1_L3_fea)

        # one3_fea = self.feature_extraction_otherview(self.lrelu(self.conv_first_otherview(one3.view(-1, C, H, W)))).view(B, 8, -1, H, W)
        # two3_fea = self.feature_extraction_otherview(self.lrelu(self.conv_first_otherview(two3.view(-1, C, H, W)))).view(B, 8, -1, H, W)
        # three3_fea = self.feature_extraction_otherview(self.lrelu(self.conv_first_otherview(three3.view(-1, C, H, W)))).view(B, 8, -1, H, W)
        
        one3_1 = self.one3_prealign1(one3_fea[:,0,:,:,:].contiguous(),centerView3_L1_fea)
        one3_2 = self.one3_prealign2(one3_fea[:,1,:,:,:].contiguous(),centerView3_L1_fea)
        one3_3 = self.one3_prealign3(one3_fea[:,2,:,:,:].contiguous(),centerView3_L1_fea)
        one3_4 = self.one3_prealign4(one3_fea[:,3,:,:,:].contiguous(),centerView3_L1_fea)
        one3_5 = self.one3_prealign5(one3_fea[:,4,:,:,:].contiguous(),centerView3_L1_fea)
        one3_6 = self.one3_prealign6(one3_fea[:,5,:,:,:].contiguous(),centerView3_L1_fea)
        one3_7 = self.one3_prealign7(one3_fea[:,6,:,:,:].contiguous(),centerView3_L1_fea)
        one3_8 = self.one3_prealign8(one3_fea[:,7,:,:,:].contiguous(),centerView3_L1_fea)
        two3_1 = self.two3_prealign1(two3_fea[:,0,:,:,:].contiguous(),centerView3_L2_fea)
        two3_2 = self.two3_prealign2(two3_fea[:,1,:,:,:].contiguous(),centerView3_L2_fea)
        two3_3 = self.two3_prealign3(two3_fea[:,2,:,:,:].contiguous(),centerView3_L2_fea)
        two3_4 = self.two3_prealign4(two3_fea[:,3,:,:,:].contiguous(),centerView3_L2_fea)
        two3_5 = self.two3_prealign5(two3_fea[:,4,:,:,:].contiguous(),centerView3_L2_fea)
        two3_6 = self.two3_prealign6(two3_fea[:,5,:,:,:].contiguous(),centerView3_L2_fea)
        two3_7 = self.two3_prealign7(two3_fea[:,6,:,:,:].contiguous(),centerView3_L2_fea)
        two3_8 = self.two3_prealign8(two3_fea[:,7,:,:,:].contiguous(),centerView3_L2_fea)
        three3_1 = self.three3_prealign1(three3_fea[:,0,:,:,:].contiguous(),centerView3_L3_fea)
        three3_2 = self.three3_prealign2(three3_fea[:,1,:,:,:].contiguous(),centerView3_L3_fea)
        three3_3 = self.three3_prealign3(three3_fea[:,2,:,:,:].contiguous(),centerView3_L3_fea)
        three3_4 = self.three3_prealign4(three3_fea[:,3,:,:,:].contiguous(),centerView3_L3_fea)
        three3_5 = self.three3_prealign5(three3_fea[:,4,:,:,:].contiguous(),centerView3_L3_fea)
        three3_6 = self.three3_prealign6(three3_fea[:,5,:,:,:].contiguous(),centerView3_L3_fea)
        three3_7 = self.three3_prealign7(three3_fea[:,6,:,:,:].contiguous(),centerView3_L3_fea)
        three3_8 = self.three3_prealign8(three3_fea[:,7,:,:,:].contiguous(),centerView3_L3_fea)

        one1_fea = torch.cat((one1_1,one1_2,one1_3,one1_4,one1_5,one1_6,one1_7,one1_8),dim=1)
        two1_fea = F.interpolate(torch.cat((two1_1,two1_2,two1_3,two1_4,two1_5,two1_6,two1_7,two1_8),dim=1), scale_factor=2, mode="bilinear")
        three1_fea = F.interpolate(torch.cat((three1_1,three1_2,three1_3,three1_4,three1_5,three1_6,three1_7,three1_8),dim=1), scale_factor=4, mode="bilinear")
        
        one3_fea = torch.cat((one3_1,one3_2,one3_3,one3_4,one3_5,one3_6,one3_7,one3_8),dim=1)
        two3_fea = F.interpolate(torch.cat((two3_1,two3_2,two3_3,two3_4,two3_5,two3_6,two3_7,two3_8),dim=1), scale_factor=2, mode="bilinear")
        three3_fea = F.interpolate(torch.cat((three3_1,three3_2,three3_3,three3_4,three3_5,three3_6,three3_7,three3_8),dim=1), scale_factor=4, mode="bilinear")


        left_fea = self.concat1(torch.cat((centerView1_L1_fea,one1_fea,two1_fea,three1_fea),dim=1))
        right_fea = self.concat3(torch.cat((centerView3_L1_fea,one3_fea,two3_fea,three3_fea),dim=1))

        del one1_fea,two1_fea,three1_fea,centerView1_L1_fea,one1_1,one1_2,one1_3,one1_4,one1_5,one1_6,one1_7,one1_8,\
                                            two1_1,two1_2,two1_3,two1_4,two1_5,two1_6,two1_7,two1_8,\
                                            three1_1,three1_2,three1_3,three1_4,three1_5,three1_6,three1_7,three1_8
        del one3_fea,two3_fea,three3_fea,centerView3_L1_fea,one3_1,one3_2,one3_3,one3_4,one3_5,one3_6,one3_7,one3_8,\
                                            two3_1,two3_2,two3_3,two3_4,two3_5,two3_6,two3_7,two3_8,\
                                            three3_1,three3_2,three3_3,three3_4,three3_5,three3_6,three3_7,three3_8
        torch.cuda.empty_cache()

        # one1_fea = self.FeatureSelectionReduction_one1(one1_fea)
        # one3_fea = self.FeatureSelectionReduction_one3(one3_fea)
        # two1_fea = self.FeatureSelectionReduction_two1(two1_fea)
        # two3_fea = self.FeatureSelectionReduction_two3(two3_fea)
        # three1_fea = self.FeatureSelectionReduction_three1(three1_fea)
        # three3_fea = self.FeatureSelectionReduction_three3(three3_fea)

        # fused_correspondence_one1_1 = self.PatchCorrespondenceAggregation_one1(one1_fea, centerView1_L1_fea)
        # fused_correspondence_one1_3 = self.PatchCorrespondenceAggregation_one1(one1_fea, centerView3_L1_fea)
        # fused_correspondence_one3_1 = self.PatchCorrespondenceAggregation_one3(one3_fea, centerView1_L1_fea)
        # fused_correspondence_one3_3 = self.PatchCorrespondenceAggregation_one3(one3_fea, centerView3_L1_fea)

        # fused_correspondence_two1_1 = self.PatchCorrespondenceAggregation_two1(two1_fea, centerView1_L2_fea)
        # fused_correspondence_two1_3 = self.PatchCorrespondenceAggregation_two3(two1_fea, centerView3_L2_fea)
        # fused_correspondence_two3_1 = self.PatchCorrespondenceAggregation_two1(two3_fea, centerView1_L2_fea)
        # fused_correspondence_two3_3 = self.PatchCorrespondenceAggregation_two3(two3_fea, centerView3_L2_fea)

        # fused_correspondence_three1_1 = self.PatchCorrespondenceAggregation_three1(three1_fea, centerView1_L3_fea)
        # fused_correspondence_three1_3 = self.PatchCorrespondenceAggregation_three3(three1_fea, centerView3_L3_fea)
        # fused_correspondence_three3_1 = self.PatchCorrespondenceAggregation_three1(three3_fea, centerView1_L3_fea)
        # fused_correspondence_three3_3 = self.PatchCorrespondenceAggregation_three3(three3_fea, centerView3_L3_fea)
        # # print(fused_correspondence_one1_1.shape,fused_correspondence_one1_3.shape)
        # fused_correspondence_one1 = self.FeatureSelectionReduction1(torch.cat([fused_correspondence_one1_1,fused_correspondence_one1_3],dim=1))
        # fused_correspondence_one3 = self.FeatureSelectionReduction2(torch.cat([fused_correspondence_one3_1,fused_correspondence_one3_3],dim=1))
        # fused_correspondence_two1 = self.FeatureSelectionReduction3(torch.cat([fused_correspondence_two1_1,fused_correspondence_two1_3],dim=1))
        # fused_correspondence_two3 = self.FeatureSelectionReduction4(torch.cat([fused_correspondence_two3_1,fused_correspondence_two3_3],dim=1))
        # fused_correspondence_three1 = self.FeatureSelectionReduction5(torch.cat([fused_correspondence_three1_1,fused_correspondence_three1_3],dim=1))
        # fused_correspondence_three3 = self.FeatureSelectionReduction6(torch.cat([fused_correspondence_three3_1,fused_correspondence_three3_3],dim=1))



        # centerView1_fea = self.fusion_one1(torch.cat([centerView1_L1_fea,fused_correspondence_one1,F.interpolate(fused_correspondence_two1, scale_factor=2, mode="bilinear"),F.interpolate(fused_correspondence_three1, scale_factor=4, mode="bilinear")], dim=1))
        # centerView3_fea = self.fusion_one3(torch.cat([centerView3_L1_fea,fused_correspondence_one3,F.interpolate(fused_correspondence_two3, scale_factor=2, mode="bilinear"),F.interpolate(fused_correspondence_three3, scale_factor=4, mode="bilinear")], dim=1))
        
        interpolatedFeature = self.interpolation(left_fea,right_fea)
        # torch.Size([2, 64, 256, 104])
        Bi, Ci, Hi, Wi = interpolatedFeature.size()
        interpolatedFeature = interpolatedFeature.view(Bi, 1, -1, Hi, Wi)
        centerView1_fea = left_fea.view(Bi, 1, -1, Hi, Wi)
        centerView3_fea = right_fea.view(Bi, 1, -1, Hi, Wi)
        
        feats = torch.cat([centerView1_fea,interpolatedFeature,centerView3_fea], dim=1)
        B, T, C, H, W = feats.size()
        feats = self.featureFusion(feats)
        feats = feats.view(B * T, C, H, W)
        out = self.recon_trunk1(feats)
        out = self.recon_trunk2(out)
        out = self.recon_trunk3(out)
        out = self.recon_trunk4(out)
        # out = out + feats
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        return outs



if __name__ == "__main__":
    # input = torch.rand(2,64,128,128).cuda()
    centerView1 = torch.rand(1, 3, 256, 104).cuda()
    one1 = torch.rand(1, 8, 3, 256, 104).cuda()
    two1 = torch.rand(1, 8, 3, 256, 104).cuda()
    three1 = torch.rand(1, 8, 3, 256, 104).cuda()
    model = LFSTVSR().cuda()
    # centerView1_L1_fea, centerView3_L1_fea, one1_L1_fea, one3_L1_fea, two1_L1_fea, two3_L1_fea, three1_L1_fea, three3_L1_fea = model(centerView1, one1, two1, three1, centerView1, one1, two1, three1)
    out = model(centerView1, one1, two1, three1, centerView1, one1, two1, three1)
    print(out.shape)
    # print(centerView3_L1_fea.shape)
    # print(one1_L1_fea.shape)
    # print(one3_L1_fea.shape)
    # print(two1_L1_fea.shape)
    # print(two3_L1_fea.shape)
    # print(three1_L1_fea.shape)
    # print(three3_L1_fea.shape)



