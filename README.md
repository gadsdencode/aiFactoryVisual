# create env
conda env create -f environment.yml
conda activate aifactory

# GPU users (optional): install CUDA-enabled PyTorch
# (Skip this if you want CPU-only. Windows/NVIDIA example:)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia