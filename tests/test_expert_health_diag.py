"""
Smoke test for expert health diagnostics module.
"""
import torch
from src.utils.expert_health_diag import (
    ExpertHealthDiagConfig, 
    ExpertHealthDiagnostics,
    compute_gini_coefficient,
    compute_expert_utilization,
    UtilizationConfig
)
from pathlib import Path
import tempfile


def test_gini_coefficient():
    """Test Gini coefficient computation."""
    print('Testing Gini coefficient...')
    gini_uniform = compute_gini_coefficient([0.25, 0.25, 0.25, 0.25])
    gini_skewed = compute_gini_coefficient([0.9, 0.05, 0.03, 0.02])
    print(f'  Uniform: {gini_uniform:.4f} (expected ~0)')
    print(f'  Skewed: {gini_skewed:.4f} (expected >0.5)')
    assert gini_uniform < 0.1, f"Uniform Gini should be ~0, got {gini_uniform}"
    assert gini_skewed > 0.5, f"Skewed Gini should be >0.5, got {gini_skewed}"


def test_expert_utilization():
    """Test expert utilization metrics."""
    print('Testing expert utilization...')
    gate_w = torch.tensor([
        [0.7, 0.1, 0.1, 0.1],  # expert 0 dominates
        [0.8, 0.1, 0.05, 0.05],
        [0.6, 0.2, 0.1, 0.1],
        [0.9, 0.05, 0.03, 0.02],
    ])
    expert_names = ['mlp_deep', 'mlp_shallow', 'cross_v2', 'cross_small']
    util_cfg = UtilizationConfig(dead_threshold=0.01, monopoly_threshold=0.8)
    util_metrics = compute_expert_utilization(gate_w, expert_names, util_cfg)
    
    print(f'  Top1 share: {util_metrics["expert_top1_share"]}')
    print(f'  Monopoly experts: {util_metrics["monopoly_experts"]}')
    print(f'  Dead experts: {util_metrics["dead_experts"]}')
    print(f'  Gini: {util_metrics["gini_coefficient"]:.4f}')
    
    # mlp_deep should be monopoly (top1 share = 1.0)
    assert util_metrics["monopoly_expert_count"] > 0, "Should detect monopoly"
    assert 'mlp_deep' in util_metrics["monopoly_experts"], "mlp_deep should be monopoly"


def test_diagnostics_manager():
    """Test ExpertHealthDiagnostics manager."""
    print('Testing ExpertHealthDiagnostics...')
    
    gate_w = torch.tensor([
        [0.7, 0.1, 0.1, 0.1],
        [0.8, 0.1, 0.05, 0.05],
        [0.6, 0.2, 0.1, 0.1],
        [0.9, 0.05, 0.03, 0.02],
    ])
    expert_names = ['mlp_deep', 'mlp_shallow', 'cross_v2', 'cross_small']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ExpertHealthDiagConfig(
            enabled=True,
            log_interval=10,
            log_on_valid=True,
        )
        diag = ExpertHealthDiagnostics(cfg, Path(tmpdir))
        
        # 收集数据
        for _ in range(3):
            diag.collect_gate_weights('ctr', gate_w)
            diag.collect_gate_weights('cvr', gate_w * 0.9)
        
        diag.set_expert_names('ctr', expert_names)
        diag.set_expert_names('cvr', expert_names)
        
        # 计算并记录
        metrics = diag.compute_and_log(step=100, epoch=1, phase='train')
        print(f'  Alerts: {metrics["alerts"]}')
        
        # 检查日志文件
        log_path = Path(tmpdir) / 'expert_health_diag.jsonl'
        assert log_path.exists(), "Log file should be created"
        print(f'  Log file created: {log_path}')
        
        with open(log_path) as f:
            content = f.read()
        print(f'  Log content length: {len(content)} chars')
        assert len(content) > 100, "Log should contain meaningful content"


def test_config_from_dict():
    """Test config parsing from dict."""
    print('Testing config from dict...')
    cfg_dict = {
        "enabled": True,
        "log_interval": 500,
        "log_on_valid": True,
        "utilization": {
            "enabled": True,
            "dead_threshold": 0.02,
            "monopoly_threshold": 0.75,
        },
        "output_stats": {
            "enabled": False,
        },
    }
    cfg = ExpertHealthDiagConfig.from_dict(cfg_dict)
    
    assert cfg.enabled == True
    assert cfg.log_interval == 500
    assert cfg.utilization.dead_threshold == 0.02
    assert cfg.utilization.monopoly_threshold == 0.75
    assert cfg.output_stats.enabled == False
    print('  Config parsed correctly')


if __name__ == "__main__":
    test_gini_coefficient()
    test_expert_utilization()
    test_diagnostics_manager()
    test_config_from_dict()
    print('\nAll tests passed!')
