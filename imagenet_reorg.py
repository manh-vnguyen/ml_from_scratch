import os, shutil

root = "/mnt/9a2c028e-6c0b-4d1a-ada1-2349e5ef78f7/data/ImageNet2012"
val_dir = os.path.join(root, "val")

# File containing 50000 WNIDs in the same order as ILSVRC2012_val_00000001.JPEG ... 50000
wnid_file = os.path.join(root, "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")

with open(wnid_file) as f:
    wnids = [line.strip() for line in f if line.strip()]

for i, wnid in enumerate(wnids, start=1):
    img = f"ILSVRC2012_val_{i:08d}.JPEG"
    src = os.path.join(val_dir, img)
    dst_dir = os.path.join(val_dir, wnid)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, img)
    if os.path.exists(src):
        shutil.move(src, dst)