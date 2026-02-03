"""
Test script for heterogeneous expert building and PLE backward compatibility.

This script verifies:
1. Expert building (MLP, CrossNet-v2, Identity)
2. ExpertOutputAlign functionality
3. PerTaskExpertAligner
4. PLE backward compatibility (homogeneous mode)
5. PLE heterogeneous mode (when config enables it)
"""
import torch
import sys


def test_expert_building():
    """Test individual expert builders."""
    print("=" * 60)
    print("Test 1: Expert Building")
    print("=" * 60)
    
    from src.models.mtl.experts import build_expert, build_expert_list
    
    # Test MLP expert
    spec_mlp = {'type': 'mlp', 'dims': [256, 128], 'activation': 'relu', 'dropout': 0.1}
    expert_mlp = build_expert(spec_mlp, in_dim=64, out_dim=128)
    print(f"  MLP expert output_dim: {expert_mlp.output_dim}")
    
    # Test CrossNet-v2 expert
    spec_cross = {
        'type': 'crossnet_v2', 
        'num_layers': 3, 
        'low_rank': 32, 
        'proj': {'enabled': True, 'out_dim': 128}
    }
    expert_cross = build_expert(spec_cross, in_dim=64, out_dim=128)
    print(f"  CrossNet-v2 expert output_dim: {expert_cross.output_dim}")
    
    # Test Identity expert
    spec_identity = {'type': 'identity'}
    expert_identity = build_expert(spec_identity, in_dim=64, out_dim=128)
    print(f"  Identity expert output_dim: {expert_identity.output_dim}")
    
    # Test forward
    x = torch.randn(4, 64)
    y_mlp = expert_mlp(x)
    y_cross = expert_cross(x)
    y_identity = expert_identity(x)
    print(f"  MLP output shape: {y_mlp.shape}")
    print(f"  CrossNet-v2 output shape: {y_cross.shape}")
    print(f"  Identity output shape: {y_identity.shape}")
    
    # Build expert list
    specs = [spec_mlp, spec_cross, spec_identity]
    expert_list = build_expert_list(specs, in_dim=64, out_dim=128)
    print(f"  Expert list length: {len(expert_list)}")
    
    print("  ✓ Expert building tests passed!")
    return expert_list, x


def test_output_align(expert_list, x):
    """Test ExpertOutputAlign and PerTaskExpertAligner."""
    print("\n" + "=" * 60)
    print("Test 2: Expert Output Alignment")
    print("=" * 60)
    
    from src.models.mtl.experts import ExpertOutputAlign, PerTaskExpertAligner
    
    # Get expert outputs
    expert_outputs = [expert(x) for expert in expert_list]
    stacked = torch.stack(expert_outputs, dim=2)  # [B, D, K]
    print(f"  Stacked expert outputs shape: {stacked.shape}")
    
    # Test ExpertOutputAlign
    aligner = ExpertOutputAlign(
        num_experts=3, 
        dim=128, 
        layernorm=True, 
        learnable_scale=True,
        dropout=0.0,
    )
    aligned = aligner(stacked)
    print(f"  Aligned output shape: {aligned.shape}")
    
    # Test PerTaskExpertAligner
    task_aligner = PerTaskExpertAligner(
        task_num_experts={'ctr': 5, 'cvr': 6},
        dim=128,
        layernorm=True,
        learnable_scale=True,
    )
    x_ctr = torch.randn(4, 128, 5)
    x_cvr = torch.randn(4, 128, 6)
    y_ctr = task_aligner('ctr', x_ctr)
    y_cvr = task_aligner('cvr', x_cvr)
    print(f"  CTR aligned shape: {y_ctr.shape}")
    print(f"  CVR aligned shape: {y_cvr.shape}")
    
    print("  ✓ Output alignment tests passed!")


def test_hetero_detection():
    """Test _should_use_hetero_experts function."""
    print("\n" + "=" * 60)
    print("Test 3: Heterogeneous Expert Detection")
    print("=" * 60)
    
    from src.models.mtl.ple import _should_use_hetero_experts
    
    # Case 1: No experts config -> homogeneous
    cfg1 = {'shared_num_experts': 4}
    assert not _should_use_hetero_experts(cfg1), "Expected homogeneous"
    print("  Case 1 (no experts): homogeneous ✓")
    
    # Case 2: hetero_enabled=False -> homogeneous
    cfg2 = {
        'hetero_enabled': False,
        'experts': {'shared': [{'type': 'mlp', 'dims': [128]}]}
    }
    assert not _should_use_hetero_experts(cfg2), "Expected homogeneous"
    print("  Case 2 (hetero_enabled=False): homogeneous ✓")
    
    # Case 3: Empty experts.shared -> homogeneous
    cfg3 = {'experts': {'shared': []}}
    assert not _should_use_hetero_experts(cfg3), "Expected homogeneous"
    print("  Case 3 (empty shared): homogeneous ✓")
    
    # Case 4: Valid config -> heterogeneous
    cfg4 = {
        'hetero_enabled': True,
        'experts': {'shared': [{'type': 'mlp', 'dims': [128]}]}
    }
    assert _should_use_hetero_experts(cfg4), "Expected heterogeneous"
    print("  Case 4 (valid config): heterogeneous ✓")
    
    # Case 5: Valid config without explicit hetero_enabled -> heterogeneous
    cfg5 = {
        'experts': {'shared': [{'type': 'mlp', 'dims': [128]}]}
    }
    assert _should_use_hetero_experts(cfg5), "Expected heterogeneous"
    print("  Case 5 (implicit enable): heterogeneous ✓")
    
    print("  ✓ Heterogeneous detection tests passed!")


def test_crossnet_v2():
    """Test CrossNet-v2 implementation."""
    print("\n" + "=" * 60)
    print("Test 4: CrossNet-v2 Details")
    print("=" * 60)
    
    from src.models.mtl.experts.registry import CrossNetV2, CrossNetV2Expert
    
    # Test full-rank CrossNet
    cross_full = CrossNetV2(in_dim=64, num_layers=3, low_rank=None, use_bias=True)
    x = torch.randn(4, 64)
    y = cross_full(x)
    print(f"  Full-rank CrossNet output shape: {y.shape}")
    assert y.shape == x.shape, "CrossNet should preserve input shape"
    
    # Test low-rank CrossNet
    cross_lr = CrossNetV2(in_dim=64, num_layers=3, low_rank=16, use_bias=True)
    y_lr = cross_lr(x)
    print(f"  Low-rank CrossNet output shape: {y_lr.shape}")
    assert y_lr.shape == x.shape, "Low-rank CrossNet should preserve input shape"
    
    # Test CrossNetV2Expert with projection
    expert = CrossNetV2Expert(
        in_dim=64,
        out_dim=128,
        num_layers=2,
        low_rank=16,
        proj_enabled=True,
    )
    y_expert = expert(x)
    print(f"  CrossNetV2Expert output shape: {y_expert.shape}")
    assert y_expert.shape == (4, 128), "Expert should project to out_dim"
    
    # Parameter count comparison
    params_full = sum(p.numel() for p in cross_full.parameters())
    params_lr = sum(p.numel() for p in cross_lr.parameters())
    print(f"  Full-rank params: {params_full:,}")
    print(f"  Low-rank (r=16) params: {params_lr:,}")
    print(f"  Reduction: {(1 - params_lr/params_full)*100:.1f}%")
    
    print("  ✓ CrossNet-v2 tests passed!")


def test_ple_backward_compat():
    """Test PLE backward compatibility - homogeneous mode should work unchanged."""
    print("\n" + "=" * 60)
    print("Test 5: PLE Backward Compatibility (Homogeneous Mode)")
    print("=" * 60)
    
    from src.models.mtl.ple import PLE, _should_use_hetero_experts
    from src.models.backbones.layers import MLP
    from torch import nn
    
    # Create a minimal mock backbone
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.deep_out_dim = 64
            self.out_dim = 64
            self.linear = nn.Linear(10, 64)
            
        def forward(self, features, dense_x=None, return_aux=False):
            x = torch.randn(features['batch_size'], 64)
            out = {
                'h': x,
                'deep_h': x,
                'logit_linear': torch.randn(features['batch_size']),
                'fm_logit': torch.randn(features['batch_size']),
            }
            if return_aux:
                out['aux'] = {}
            return out
    
    backbone = MockBackbone()
    
    # Old-style config (no experts, no hetero_enabled)
    ple_cfg = {
        'shared_num_experts': 2,
        'specific_num_experts': {'ctr': 1, 'cvr': 1},
        'expert_mlp_dims': [64],
        'dropout': 0.0,
        'activation': 'relu',
        'gate_type': 'linear',
        'gate_stabilize': {'enabled': False},
    }
    
    head_cfg = {
        'tasks': ['ctr', 'cvr'],
        'default': {'mlp_dims': [32], 'dropout': 0.0},
    }
    
    # Verify this is homogeneous mode
    assert not _should_use_hetero_experts(ple_cfg), "Should be homogeneous"
    
    # Build PLE
    ple = PLE(
        backbone=backbone,
        head_cfg=head_cfg,
        ple_cfg=ple_cfg,
        enabled_heads=['ctr', 'cvr'],
        log_gates=True,
    )
    
    # Verify internal state
    assert not ple._use_hetero_experts, "Should be homogeneous mode"
    assert ple.num_shared_experts == 2
    assert len(ple.shared_experts) == 2
    assert ple._expert_output_aligner is None, "Should not have aligner in homogeneous mode"
    print(f"  Homogeneous mode: shared={ple.num_shared_experts}, private={ple.num_specific_experts}")
    
    # Forward pass
    features = {'batch_size': 4}
    out = ple(features)
    
    assert 'ctr' in out, "Should have ctr output"
    assert 'cvr' in out, "Should have cvr output"
    assert out['ctr'].shape == (4,), f"CTR shape should be (4,), got {out['ctr'].shape}"
    assert out['cvr'].shape == (4,), f"CVR shape should be (4,), got {out['cvr'].shape}"
    
    # Check aux contains gates
    assert 'aux' in out, "Should have aux output"
    assert 'gates' in out['aux'], "Should have gate weights"
    
    print(f"  CTR output shape: {out['ctr'].shape}")
    print(f"  CVR output shape: {out['cvr'].shape}")
    print("  ✓ Backward compatibility test passed!")


def test_ple_hetero_mode():
    """Test PLE heterogeneous expert mode."""
    print("\n" + "=" * 60)
    print("Test 6: PLE Heterogeneous Expert Mode")
    print("=" * 60)
    
    from src.models.mtl.ple import PLE, _should_use_hetero_experts
    from torch import nn
    
    # Create a minimal mock backbone
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.deep_out_dim = 64
            self.out_dim = 64
            self.linear = nn.Linear(10, 64)
            
        def forward(self, features, dense_x=None, return_aux=False):
            x = torch.randn(features['batch_size'], 64)
            out = {
                'h': x,
                'deep_h': x,
                'logit_linear': torch.randn(features['batch_size']),
                'fm_logit': torch.randn(features['batch_size']),
            }
            if return_aux:
                out['aux'] = {}
            return out
    
    backbone = MockBackbone()
    
    # Heterogeneous expert config
    ple_cfg = {
        'hetero_enabled': True,
        'expert_out_dim': 64,
        'expert_output_align': {
            'layernorm': True,
            'learnable_scale': True,
            'dropout': 0.0,
        },
        'experts': {
            'shared': [
                {'name': 'mlp_deep', 'type': 'mlp', 'dims': [128, 64], 'activation': 'relu'},
                {'name': 'cross_v2', 'type': 'crossnet_v2', 'num_layers': 2, 'low_rank': 16},
            ],
            'private': {
                'ctr': [
                    {'name': 'ctr_mlp', 'type': 'mlp', 'dims': [64]},
                ],
                'cvr': [
                    {'name': 'cvr_identity', 'type': 'identity'},
                ],
            }
        },
        'gate_type': 'linear',
        'gate_stabilize': {'enabled': False},
    }
    
    head_cfg = {
        'tasks': ['ctr', 'cvr'],
        'default': {'mlp_dims': [32], 'dropout': 0.0},
    }
    
    # Verify this is heterogeneous mode
    assert _should_use_hetero_experts(ple_cfg), "Should be heterogeneous"
    
    # Build PLE
    ple = PLE(
        backbone=backbone,
        head_cfg=head_cfg,
        ple_cfg=ple_cfg,
        enabled_heads=['ctr', 'cvr'],
        log_gates=True,
    )
    
    # Verify internal state
    assert ple._use_hetero_experts, "Should be heterogeneous mode"
    assert ple.num_shared_experts == 2
    assert len(ple.shared_experts) == 2
    assert ple._expert_output_aligner is not None, "Should have aligner in hetero mode"
    
    # Check expert names
    assert 'ctr' in ple._expert_names
    assert 'cvr' in ple._expert_names
    assert ple._expert_names['ctr'] == ['mlp_deep', 'cross_v2', 'ctr_mlp']
    assert ple._expert_names['cvr'] == ['mlp_deep', 'cross_v2', 'cvr_identity']
    
    print(f"  Hetero mode: shared={ple.num_shared_experts}, private={ple.num_specific_experts}")
    print(f"  CTR experts: {ple._expert_names['ctr']}")
    print(f"  CVR experts: {ple._expert_names['cvr']}")
    
    # Forward pass
    features = {'batch_size': 4}
    out = ple(features)
    
    assert 'ctr' in out, "Should have ctr output"
    assert 'cvr' in out, "Should have cvr output"
    assert out['ctr'].shape == (4,), f"CTR shape should be (4,), got {out['ctr'].shape}"
    assert out['cvr'].shape == (4,), f"CVR shape should be (4,), got {out['cvr'].shape}"
    
    # Check aux contains expert routing metrics
    assert 'aux' in out, "Should have aux output"
    assert 'gates' in out['aux'], "Should have gate weights"
    assert 'hetero_experts_enabled' in out['aux'], "Should have hetero flag"
    assert out['aux']['hetero_experts_enabled'] == True
    
    print(f"  CTR output shape: {out['ctr'].shape}")
    print(f"  CVR output shape: {out['cvr'].shape}")
    print("  ✓ Heterogeneous mode test passed!")


def test_ple_fallback():
    """Test PLE fallback from hetero to homo mode."""
    print("\n" + "=" * 60)
    print("Test 7: PLE Fallback (hetero_enabled=False)")
    print("=" * 60)
    
    from src.models.mtl.ple import PLE, _should_use_hetero_experts
    from torch import nn
    
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.deep_out_dim = 64
            self.out_dim = 64
            
        def forward(self, features, dense_x=None, return_aux=False):
            x = torch.randn(features['batch_size'], 64)
            out = {
                'h': x,
                'deep_h': x,
                'logit_linear': torch.randn(features['batch_size']),
            }
            if return_aux:
                out['aux'] = {}
            return out
    
    backbone = MockBackbone()
    
    # Config with experts defined BUT hetero_enabled=False
    ple_cfg = {
        'hetero_enabled': False,  # Force homogeneous mode
        'shared_num_experts': 3,
        'specific_num_experts': {'ctr': 1, 'cvr': 2},
        'expert_mlp_dims': [64],
        'experts': {  # This should be IGNORED
            'shared': [
                {'name': 'mlp_deep', 'type': 'mlp', 'dims': [128]},
            ],
        },
        'gate_type': 'linear',
        'gate_stabilize': {'enabled': False},
    }
    
    head_cfg = {
        'tasks': ['ctr', 'cvr'],
        'default': {'mlp_dims': [32], 'dropout': 0.0},
    }
    
    # Verify fallback
    assert not _should_use_hetero_experts(ple_cfg), "Should fallback to homogeneous"
    
    ple = PLE(
        backbone=backbone,
        head_cfg=head_cfg,
        ple_cfg=ple_cfg,
        enabled_heads=['ctr', 'cvr'],
    )
    
    # Verify homogeneous mode is used
    assert not ple._use_hetero_experts, "Should be homogeneous after fallback"
    assert ple.num_shared_experts == 3, "Should use shared_num_experts from config"
    assert ple._expert_output_aligner is None, "Should not have aligner"
    
    print(f"  Fallback successful: using shared_num_experts={ple.num_shared_experts}")
    print("  ✓ Fallback test passed!")


def main():
    print("\n" + "=" * 60)
    print("PLE-Lite Heterogeneous Expert Test Suite")
    print("=" * 60 + "\n")
    
    try:
        expert_list, x = test_expert_building()
        test_output_align(expert_list, x)
        test_hetero_detection()
        test_crossnet_v2()
        test_ple_backward_compat()
        test_ple_hetero_mode()
        test_ple_fallback()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
