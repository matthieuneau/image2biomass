echo "cd /workspace/image2biomass && source .venv/bin/activate" >> ~/.bashrc

git clone https://github.com/matthieuneau/image2biomass.git

cd image2biomass
mkdir models
mkdir data

uv sync

mkdir ~/.kaggle
touch ~/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

echo "export KAGGLE_API_TOKEN=KGAT_0c7513e14e359cef47320a78630ee7dd" >> ~/.bashrc
source ~/.bashrc

source .venv/bin/activate

wandb login

wandb artifact get matthieu-neau/image2biomass/image2biomass_dataset:v0 --root data
