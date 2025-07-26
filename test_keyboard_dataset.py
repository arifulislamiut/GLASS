#!/usr/bin/env python3
"""
Test script for keyboard dataset setup
"""

import os
import sys
from datasets.keyboard import KeyboardDataset, DatasetSplit

def test_keyboard_dataset():
    """Test the keyboard dataset setup"""
    print("Testing Keyboard Dataset Setup...")
    
    # Dataset paths
    dataset_path = "datasets/keyboard"
    aug_path = "datasets/dtd/images"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset path found: {dataset_path}")
    
    # Check dataset structure
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    
    if not os.path.exists(train_path):
        print(f"❌ Train directory not found: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"❌ Test directory not found: {test_path}")
        return False
    
    print(f"✅ Train directory found: {train_path}")
    print(f"✅ Test directory found: {test_path}")
    
    # Check train data
    train_good_path = os.path.join(train_path, "good")
    if not os.path.exists(train_good_path):
        print(f"❌ Train good directory not found: {train_good_path}")
        return False
    
    train_images = [f for f in os.listdir(train_good_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Train good images: {len(train_images)}")
    
    # Check test data
    test_good_path = os.path.join(test_path, "good")
    test_defective_path = os.path.join(test_path, "defective")
    
    if not os.path.exists(test_good_path):
        print(f"❌ Test good directory not found: {test_good_path}")
        return False
    
    if not os.path.exists(test_defective_path):
        print(f"❌ Test defective directory not found: {test_defective_path}")
        return False
    
    test_good_images = [f for f in os.listdir(test_good_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_defective_images = [f for f in os.listdir(test_defective_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"✅ Test good images: {len(test_good_images)}")
    print(f"✅ Test defective images: {len(test_defective_images)}")
    
    # Check augmentation data
    if not os.path.exists(aug_path):
        print(f"⚠️  Augmentation path not found: {aug_path}")
        print("   This might cause issues during training. Consider downloading DTD dataset.")
    else:
        aug_images = []
        for root, dirs, files in os.walk(aug_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    aug_images.append(os.path.join(root, file))
        print(f"✅ Augmentation images: {len(aug_images)}")
    
    # Test dataset loading
    try:
        print("\n🔍 Testing dataset loading...")
        
        # Test train dataset
        train_dataset = KeyboardDataset(
            source=dataset_path,
            anomaly_source_path=aug_path,
            classname='keyboard',
            split=DatasetSplit.TRAIN,
            resize=288,
            imagesize=288
        )
        print(f"✅ Train dataset loaded: {len(train_dataset)} samples")
        
        # Test test dataset
        test_dataset = KeyboardDataset(
            source=dataset_path,
            anomaly_source_path=aug_path,
            classname='keyboard',
            split=DatasetSplit.TEST,
            resize=288,
            imagesize=288
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        
        # Test data loading
        print("\n🔍 Testing data loading...")
        sample = train_dataset[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"✅ Image shape: {sample['image'].shape}")
        print(f"✅ Is anomaly: {sample['is_anomaly']}")
        
        # Test test sample
        test_sample = test_dataset[0]
        print(f"✅ Test sample - Is anomaly: {test_sample['is_anomaly']}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 Keyboard dataset setup is ready!")
    print("\nTo start training:")
    print("  bash shell/run-keyboard.sh")
    print("\nOr manually:")
    print("  python main.py --gpu 0 --test ckpt net -b wideresnet50 -le layer2 -le layer3 dataset -d keyboard keyboard datasets/keyboard datasets/dtd/images")
    
    return True

if __name__ == "__main__":
    test_keyboard_dataset() 