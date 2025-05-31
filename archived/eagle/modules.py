import torch as pt
import torch.nn as nn
import vision_transformer as vits
import torch.nn.functional as F
import gc


def tensor_correlation(a, b):
    return pt.einsum("nchw,ncij->nhwij", a, b)

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def sample(t: pt.Tensor, coords: pt.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

@pt.jit.script
def super_perm(size: int, device: pt.device):
    perm = pt.randperm(size, device=device, dtype=pt.long)
    perm[perm == pt.arange(size, device=device)] += 1
    return perm % size

class CorrespondenceLoss(nn.Module):
    def __init__(self, neg_samples, pointwise, zero_clamp, 
                 shift_bias, shift_value, stabalize, feature_samples):
        super(CorrespondenceLoss, self).__init__()
        
        self.neg_samples = neg_samples
        self.pointwise = pointwise
        self.zero_clamp = zero_clamp
        
        self.shift_bias = shift_bias
        self.shift_value = shift_value
        
        self.stabalize = stabalize
        self.feature_samples = feature_samples
        
        self.mse_loss = nn.MSELoss()

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, POS: bool):
        with pt.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        min_val = 0.0 if self.zero_clamp else -9999.0
            
        if POS:
            shift = pt.abs(fd.mean() - cd.mean()-self.shift_bias)
        else:
            shift = (fd.mean() + cd.mean()-self.shift_bias) * self.shift_value
            
        clamp_max = .8 if self.stabalize else None
        loss = - cd.clamp(min_val, clamp_max) * (fd - shift)
      
        return loss, cd
    
    def id_loss(self, input_tensor):
    
        batch_size, H, W, _  = input_tensor.shape
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        downsampled_tensor = F.interpolate(input_tensor, scale_factor=0.5, mode='bilinear', align_corners=True)

        reshaped_tensor = downsampled_tensor.permute(0,2,3,1).view(batch_size, H//2 * W//2, -1)
        normalized_patches = F.normalize(reshaped_tensor,dim=-1)

        similarity_matrix_batched = pt.bmm(normalized_patches, normalized_patches.transpose(-2, -1))

        min_vals = similarity_matrix_batched.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        max_vals = similarity_matrix_batched.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        similarity_matrix_batched = (similarity_matrix_batched - min_vals) / (max_vals - min_vals)        
    
        I = pt.eye(similarity_matrix_batched.shape[1], device=similarity_matrix_batched.device)
        loss = self.mse_loss(similarity_matrix_batched, I.unsqueeze(0).repeat(batch_size, 1, 1))
        gc.collect()
        pt.cuda.empty_cache()
        return loss


    def forward(self,
                orig_feats: pt.Tensor, orig_feats_pos: pt.Tensor, orig_feats_pos_aug: pt.Tensor, 
                orig_code: pt.Tensor, orig_code_pos: pt.Tensor, orig_code_pos_aug: pt.Tensor, 
                ):

        coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]

        coords1 = pt.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = pt.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords3 = pt.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)
        
        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)
        
        feats_pos_aug = sample(orig_feats_pos_aug, coords3)
        code_pos_aug = sample(orig_code_pos_aug, coords3)
        
        pos_inter_loss, pos_inter_cd = self.helper(
            feats_pos_aug, feats_pos, code_pos_aug, code_pos, POS = True)
        
        neg_losses = []
        neg_cds = []
        feats_neg_list = []
        code_neg_list = []
        
        for _ in range(self.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            feats_neg_list.append(feats_neg)
            code_neg = sample(orig_code[perm_neg], coords2)
            code_neg_list.append(code_neg)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, POS=False)
            
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)

        
        neg_inter_loss = pt.cat(neg_losses, dim=0)
        neg_inter_cd = pt.cat(neg_cds, dim=0)
        
        return (
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd
                )

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothingCrossEntropy class.
        :param smoothing: The smoothing factor (float, default: 0.1).
                          This factor dictates how much we will smooth the labels.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        """
        Forward pass of the loss function.
        :param input: Predictions from the model (before softmax) (tensor).
        :param target: True labels (tensor).
        """
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class ClusterLookup(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = pt.nn.Parameter(pt.randn(n_classes, dim))

    def reset_parameters(self):
        with pt.no_grad():
            self.clusters.copy_(pt.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs: bool = False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = pt.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(pt.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(pt.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        
        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
            
class newLocalGlobalInfoNCE(nn.Module):
    def __init__(self, num_classes, dim, extra_clusters, centroid_mode, global_loss_weight, contrastive_temp):
        super(newLocalGlobalInfoNCE, self).__init__()
        
        self.learned_centroids = nn.Parameter(pt.randn(num_classes, dim))
        self.prototypes = pt.randn(num_classes + extra_clusters, dim, requires_grad=True)
        
        self.centroid_mode = centroid_mode
        self.global_loss_weight = global_loss_weight
        self.contrastive_temp = contrastive_temp
        
    def compute_centroid(self, features, labels):
        unique_labels = pt.unique(labels)
         
        centroids = []
        
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            
            if self.centroid_mode == 'mean':
                centroids.append(class_features.mean(0))
                
            elif self.centroid_mode == 'medoid':
                pairwise_dist = pt.cdist(class_features, class_features)
                centroids.append(class_features[pt.argmin(pairwise_dist.sum(0))])
                
            elif self.centroid_mode == 'learned':
                centroids.append(self.learned_centroids[label])
                
            elif self.centroid_mode == 'prototype':
                pairwise_dist = pt.cdist(class_features, class_features)
                
                prototype = class_features[pt.argmin(pairwise_dist.sum(0))]
                centroids.append(prototype)
                
                new_prototypes = self.prototypes.clone()
                new_prototypes[label] = prototype 
                
                self.prototypes = new_prototypes
                
        return pt.stack(centroids)
    
    def forward(self, S1, S2, segmentation_map, similarity_matrix):

        # label_smoothing_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # batch_size, patch_size = segmentation_map.size(0), segmentation_map.size(1)
        
        segmentation_map = segmentation_map.reshape(-1)
        S1_centroids = self.compute_centroid(S1, segmentation_map)
 
        local_logits = pt.mm(S1, S1_centroids.t()) / self.contrastive_temp
        global_logits = pt.mm(S2, S1_centroids.t()) / self.contrastive_temp

        mask = (segmentation_map.unsqueeze(1) == pt.unique(segmentation_map)) 
        labels = mask.float().argmax(dim=1)

        local_weights = (similarity_matrix.mean(dim=2).reshape(-1)) * 1.0
        global_weights = (similarity_matrix.mean(dim=2).reshape(-1)) * 1.0
        
        # if self.cfg.dataset_name=='cityscapes':
        #     local_loss = label_smoothing_criterion(local_logits, labels)
        #     global_loss = label_smoothing_criterion(global_logits, labels)
        # else:
        local_loss = F.cross_entropy(local_logits, labels, reduction='none')
        global_loss = F.cross_entropy(global_logits, labels, reduction='none')
            
        local_loss = (local_loss * local_weights).mean()
        global_loss = (global_loss * global_weights).mean()

        total_loss = ((1-self.global_loss_weight) * local_loss + self.global_loss_weight * global_loss) / 2
        
        return total_loss
        
class DinoFeaturizer(nn.Module):
    def __init__(self, dim, 
                 pretrained_weights, 
                 dino_patch_size, dino_feat_type, arch, 
                 proj_type, dropout):
        super().__init__()
        
        self.dim = dim
        patch_size = dino_patch_size
        self.patch_size = dino_patch_size
        self.feat_type = dino_feat_type
        self.dropout_flag = dropout
        
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.model.eval().cuda()
        
        self.dropout = pt.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if pretrained_weights is not None:
            state_dict = pt.load(pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False, map_location="cpu")
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = pt.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)
            
        self.n_feats = 384 * 3 if arch == "vit_small" else 768 * 3 
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = proj_type
        
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, 
                # return_class_feat=False
                ):
        # TODO: May not be a problem since it's already set to eval in the init().
        # self.model.eval()
        
        with pt.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            
            # get selected layer activations
            feat_all, attn_all, qkv_all = self.model.get_intermediate_feat(img)

            # high level
            feat, attn, qkv = feat_all[-1], attn_all[-1], qkv_all[-1]
            
            image_feat_high = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            image_k_high = qkv[1, :, :, 1:, :].reshape(feat.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = image_k_high.shape
            image_kk_high = image_k_high.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
                       
            # mid level
            feat_mid, attn_mid, qkv_mid = feat_all[-2], attn_all[-2], qkv_all[-2]
            
            image_feat_mid = feat_mid[:, 1:, :].reshape(feat_mid.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            image_k_mid = qkv_mid[1, :, :, 1:, :].reshape(feat_mid.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = image_k_mid.shape
            image_kk_mid = image_k_mid.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            
            # low level
            feat_low, attn_low, qkv_low = feat_all[-3], attn_all[-3], qkv_all[-3]
            # ---
            image_feat_low = feat_low[:, 1:, :].reshape(feat_low.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            image_k_low = qkv_low[1, :, :, 1:, :].reshape(feat_low.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = image_k_low.shape
            image_kk_low = image_k_low.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            
            image_feat = pt.cat([image_feat_low, image_feat_mid, image_feat_high], dim=1)
            image_kk  = pt.cat([image_kk_low, image_kk_mid, image_kk_high], dim=1)
            
            # if return_class_feat:
            #     return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            with pt.no_grad():
                code = self.cluster1(self.dropout(image_feat))
            code_kk = self.cluster1(self.dropout(image_kk))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
                code_kk += self.cluster2(self.dropout(image_kk))
        else:
            code = image_feat
            code_kk = image_kk

        if self.dropout_flag:
            image_feat = self.dropout(image_feat)
            image_kk = self.dropout(image_kk)
        
        return image_feat, image_kk, code, code_kk
