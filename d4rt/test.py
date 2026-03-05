"""
PyTorch Lightning Module for D4RT Testing
"""

import torch
import lightning as L
from typing import Dict, Any, Optional

from .models.d4rt_model import D4RTModel
from .utils.losses import D4RTLoss


class D4RTTestLit(L.LightningModule):
    """PyTorch Lightning module for testing D4RT"""
    
    def __init__(
        self,
        # Model config (should match training config)
        img_size=256,
        patch_size=16,
        encoder_embed_dim=1408,
        encoder_depth=40,
        encoder_num_heads=16,
        decoder_dim=512,
        decoder_num_heads=8,
        decoder_num_layers=8,
        max_frames=100,
        # Loss config (for evaluation metrics)
        lambda_3d=1.0,
        lambda_2d=0.1,
        lambda_normal=0.5,
        lambda_visibility=0.1,
        lambda_motion=0.1,
        lambda_confidence=0.2,
        # Model checkpoint path
        checkpoint_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = D4RTModel(
            img_size=img_size,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_dim=decoder_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_num_layers=decoder_num_layers,
            max_frames=max_frames
        )
        
        # Loss (for evaluation metrics)
        self.criterion = D4RTLoss(
            lambda_3d=lambda_3d,
            lambda_2d=lambda_2d,
            lambda_normal=lambda_normal,
            lambda_visibility=lambda_visibility,
            lambda_motion=lambda_motion,
            lambda_confidence=lambda_confidence,
            query_dim=decoder_dim
        )
        
        # Note: Checkpoint loading is handled by Lightning Trainer via ckpt_path
        # This is here for manual loading if needed
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                # Try loading just the model weights
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
    
    def forward(self, batch):
        """Forward pass"""
        video = batch['video']
        coords_uv = batch['coords_uv']
        t_src = batch['t_src']
        t_tgt = batch['t_tgt']
        t_cam = batch['t_cam']
        aspect_ratio = batch.get('aspect_ratio', None)
        video_orig = batch.get('video_orig', None)  # Original resolution video for patch extraction
        
        outputs = self.model(
            video=video,
            coords_uv=coords_uv,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            aspect_ratio=aspect_ratio,
            video_orig=video_orig
        )
        
        return outputs

    def extract_gt_data(self, batch):
        """
        Extract GT query data with the same priority as training:
        1) dataset-provided query GT (preferred),
        2) fallback legacy fields.
        """
        gt_3d = batch.get('gt_3d')
        if gt_3d is not None:
            gt_2d = batch.get('gt_2d_tgt')
            if gt_2d is None:
                gt_2d = batch.get('gt_2d')

            gt_visibility = batch.get('gt_visibility_tgt')
            if gt_visibility is None:
                gt_visibility = batch.get('gt_visibility')

            if gt_visibility is None:
                gt_visibility = torch.ones(
                    gt_3d.shape[:2], device=gt_3d.device, dtype=gt_3d.dtype
                )
            elif gt_visibility.dim() == 3:
                gt_visibility = gt_visibility.squeeze(-1)

            gt_visibility = gt_visibility.float()
            finite_mask = torch.isfinite(gt_3d).all(dim=-1)
            depth_mask = gt_3d[:, :, 2] > 1e-6
            visibility_mask = gt_visibility > 0.5
            mask = finite_mask & depth_mask & visibility_mask

            return {
                'gt_3d': gt_3d,
                'gt_2d': gt_2d,
                'gt_visibility': gt_visibility,
                'mask': mask,
            }

        return {
            'gt_3d': batch.get('gt_3d'),
            'gt_2d': batch.get('gt_2d'),
            'gt_visibility': batch.get('gt_visibility'),
            'mask': batch.get('mask'),
        }
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        outputs = self.forward(batch)
        pred_3d = outputs['coords_3d']
        pred_2d = outputs.get('coords_2d')
        pred_visibility_logits = outputs.get('visibility_logits')
        pred_motion = outputs.get('motion')
        pred_normal = outputs.get('normal')
        pred_confidence = outputs.get('confidence')

        gt_data = self.extract_gt_data(batch)
        gt_3d = gt_data['gt_3d']
        gt_2d = gt_data['gt_2d']
        gt_visibility = gt_data['gt_visibility']
        mask = gt_data['mask']

        gt_normal = batch.get('gt_normal')
        gt_motion = batch.get('gt_motion')
        
        # Compute losses for evaluation
        losses = self.criterion(
            pred_3d=pred_3d,
            pred_2d=pred_2d,
            pred_visibility_logits=pred_visibility_logits,
            pred_motion=pred_motion,
            pred_normal=pred_normal,
            pred_confidence=pred_confidence,
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            gt_visibility=gt_visibility,
            gt_motion=gt_motion,
            gt_normal=gt_normal,
            mask=mask
        )
        
        # Compute additional metrics
        metrics = {}
        
        if gt_3d is not None:
            # 3D position error (L2 distance)
            l2_error = torch.norm(pred_3d - gt_3d, dim=-1)  # (B, N)
            if mask is not None:
                l2_error = l2_error * mask
                metrics['test/l2_error_mean'] = (l2_error.sum() / (mask.sum() + 1e-6)).item()
            else:
                metrics['test/l2_error_mean'] = l2_error.mean().item()
            
            # Depth error
            depth_error = torch.abs(pred_3d[:, :, 2] - gt_3d[:, :, 2])  # (B, N)
            if mask is not None:
                depth_error = depth_error * mask
                metrics['test/depth_error_mean'] = (depth_error.sum() / (mask.sum() + 1e-6)).item()
            else:
                metrics['test/depth_error_mean'] = depth_error.mean().item()
        
        # Log losses and metrics
        for key, value in losses.items():
            self.log(f'test/{key}', value, on_step=False, on_epoch=True)
        
        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True)
        
        return {
            'pred_3d': pred_3d.detach().cpu(),
            'gt_3d': gt_3d.detach().cpu() if gt_3d is not None else None,
            'losses': {k: v.detach().cpu() for k, v in losses.items()},
            'metrics': metrics
        }
