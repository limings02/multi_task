"""Quick test for one batch to estimate total processing time."""
import sys
import time

def main():
    from src.cli.main import main as cli_main
    
    start = time.time()
    sys.argv = [
        'main', 'process',
        '--config', 'configs/dataset/featuremap.yaml',
        '--split-dir', 'data/splits/aliccp_entity_hash_v1',
        '--out', 'data/processed',
        '--batch-size', '200000',
        '--log-level', 'DEBUG'
    ]
    
    try:
        cli_main()
    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {time.time() - start:.1f}s")
    except Exception as e:
        print(f"\n\nError after {time.time() - start:.1f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
