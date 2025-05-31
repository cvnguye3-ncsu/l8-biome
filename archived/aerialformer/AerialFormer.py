from importables.general.model import ModelSkeleton

from aerialformer_head import MDCDecoder
from swin_stem import SwinStemTransformer

IMG_SIZE = 224

class AerialFormer(ModelSkeleton):
    def __init__(self, resize_embedding, class_count):
        super().__init__(resize_embedding, class_count)
        
        self.input_shape = (3, IMG_SIZE, IMG_SIZE)
        
        decoder_norm_cfg = dict(type='SyncBN', requires_grad=True)
        
        # === ENCODER ===
        self.encoder = SwinStemTransformer(
                        pretrain_img_size=384,
                        embed_dims=128,
                        window_size=12,
                        depths=[2, 2, 18, 2],
                        num_heads=[4, 8, 16, 32],
                        conv_norm_cfg=decoder_norm_cfg)
        self.encoder_params = self.encoder.parameters()
        
        # === DECODER ===
        self.decoder = MDCDecoder(
                        in_channels=[64, 128, 256, 512, 1024],
                        in_index = [0, 1, 2, 3, 4],
                        channels=128,
                        norm_cfg=decoder_norm_cfg,
                        num_classes=4)
        self.decoder_params = self.decoder.parameters()
        
    def forward(self, x):
        latent = self.encoder(x)
        logits = self.decoder(latent)
        
        return logits
        
        