#!/usr/bin/env python3
"""
Monitor GLASS training progress for keyboard dataset
"""

import os
import time
import glob
from datetime import datetime

def monitor_training():
    """Monitor the training progress"""
    model_dir = "results/models/backbone_0/keyboard_keyboard"
    tb_dir = os.path.join(model_dir, "tb")
    
    print("🔍 Monitoring GLASS Training for Keyboard Dataset")
    print("=" * 60)
    
    # Check if training directory exists
    if not os.path.exists(model_dir):
        print("❌ Training directory not found. Training may not have started.")
        return
    
    print(f"✅ Training directory: {model_dir}")
    
    # Check for checkpoints
    ckpt_files = glob.glob(os.path.join(model_dir, "ckpt*.pth"))
    if ckpt_files:
        print(f"✅ Checkpoints found: {len(ckpt_files)}")
        for ckpt in sorted(ckpt_files):
            ctime = datetime.fromtimestamp(os.path.getctime(ckpt))
            size = os.path.getsize(ckpt) / (1024 * 1024)  # MB
            print(f"   - {os.path.basename(ckpt)} ({size:.1f}MB) - {ctime}")
    else:
        print("⏳ No checkpoints yet - training in progress...")
    
    # Check TensorBoard logs
    if os.path.exists(tb_dir):
        tb_files = glob.glob(os.path.join(tb_dir, "events*"))
        if tb_files:
            print(f"✅ TensorBoard logs: {len(tb_files)} files")
            for tb_file in tb_files:
                ctime = datetime.fromtimestamp(os.path.getctime(tb_file))
                size = os.path.getsize(tb_file) / (1024 * 1024)  # MB
                print(f"   - {os.path.basename(tb_file)} ({size:.1f}MB) - {ctime}")
        else:
            print("⏳ No TensorBoard logs yet...")
    else:
        print("⏳ TensorBoard directory not created yet...")
    
    # Check for training visualizations
    train_viz_dir = "results/training/keyboard_keyboard"
    if os.path.exists(train_viz_dir):
        viz_files = glob.glob(os.path.join(train_viz_dir, "*.png"))
        print(f"✅ Training visualizations: {len(viz_files)} images")
    else:
        print("⏳ Training visualizations not created yet...")
    
    # Check for evaluation results
    eval_dir = "results/eval/keyboard_keyboard"
    if os.path.exists(eval_dir):
        eval_files = glob.glob(os.path.join(eval_dir, "*.png"))
        print(f"✅ Evaluation results: {len(eval_files)} images")
    else:
        print("⏳ Evaluation results not created yet...")
    
    # Check results CSV
    results_csv = "results/results.csv"
    if os.path.exists(results_csv):
        print(f"✅ Results CSV: {results_csv}")
        # Read and display latest results
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            if not df.empty:
                print("\n📊 Latest Results:")
                for _, row in df.iterrows():
                    if 'keyboard' in str(row['Row Names']):
                        print(f"   Dataset: {row['Row Names']}")
                        print(f"   Image AUROC: {row['image_auroc']:.4f}")
                        print(f"   Image AP: {row['image_ap']:.4f}")
                        print(f"   Pixel AUROC: {row['pixel_auroc']:.4f}")
                        print(f"   Pixel AP: {row['pixel_ap']:.4f}")
                        print(f"   Best Epoch: {row['best_epoch']}")
        except Exception as e:
            print(f"   Could not read results: {e}")
    else:
        print("⏳ Results CSV not created yet...")
    
    print("\n" + "=" * 60)
    print("💡 To monitor training in real-time:")
    print("   tensorboard --logdir results/models/backbone_0/keyboard_keyboard/tb/")
    print("\n💡 To check GPU usage:")
    print("   nvidia-smi")
    print("\n💡 To stop monitoring:")
    print("   Ctrl+C")

def continuous_monitoring():
    """Continuously monitor training progress"""
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            monitor_training()
            print(f"\n🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("⏳ Refreshing in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        continuous_monitoring()
    else:
        monitor_training() 