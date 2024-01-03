# Utilis module: a module for auxilary functions

* multiprocessing for pre-processing NIFTI volumes
* metrics calculation and logging
* adding noise to simulate noisy images for denoising

# Inference
__using whole-brain inference__
1. Generate LR whole brain volume
2. Infer them to get whole brain SR volume
3. Suggested to be used on `device=cpu` when your GPU mem is low

__or using individual-patch inference__

1. Take a whole brain volume, downsample to LR
2. Patch them into overlapped volumes
3. Infer individual LR patches to get SR patches
4. Assemble them to get a SR whole brain volume
