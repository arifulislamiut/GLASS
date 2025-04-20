# Set paths
$datapath = "C:/Users/arifu/Documents/Projects/factoryAI/GLASS/datasets/mvtec_anomaly_detection"
$augpath = "C:/Users/arifu/Documents/Projects/factoryAI/GLASS/datasets/dtd/images"

# Define classes
$classes = @(
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
)

# Build flags array
$flags = @()
foreach ($class in $classes) {
    $flags += "-d $class"
}

# Move one directory up
Set-Location ..

# Build and run the python command
python main.py `
    --gpu 0 `
    --seed 0 `
    --test ckpt `
    net `
    -b wideresnet50 `
    -le layer2 `
    -le layer3 `
    --pretrain_embed_dimension 1536 `
    --target_embed_dimension 1536 `
    --patchsize 3 `
    --meta_epochs 640 `
    --eval_epochs 1 `
    --dsc_layers 2 `
    --dsc_hidden 1024 `
    --pre_proj 1 `
    --mining 1 `
    --noise 0.015 `
    --radius 0.75 `
    --p 0.5 `
    --step 20 `
    --limit 392 `
    dataset `
    --distribution 0 `
    --mean 0.5 `
    --std 0.1 `
    --fg 1 `
    --rand_aug 1 `
    --batch_size 8 `
    --resize 288 `
    --imagesize 288 `
    $flags `
    mvtec `
    $datapath `
    $augpath
