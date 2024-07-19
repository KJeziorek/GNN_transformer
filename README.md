pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install lightning wandb matplotlib numba pybind11

# Install YOLOX
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
pip install -v -e .


python setup.py build_ext --inplace

