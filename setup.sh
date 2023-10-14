# disable the restart dialogue and install several packages
sudo sed -i "/#\$nrconf{restart} = 'i';/s/.*/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf
sudo apt-get update
sudo apt install wget unzip git python3 python3-venv build-essential net-tools libgl-dev libgl1 libglib2.0-0 libgl1-mesa-glx -y

# AWS official install method
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
/bin/rm -r aws awscliv2.zip

# Download latest sd-webui
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

# install CUDA (from https://developer.nvidia.com/cuda-downloads)
#wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
#sudo sh cuda_12.0.0_525.60.13_linux.run --silent

# install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo -u ubuntu git lfs install --skip-smudge

# download the SD model v2.1 and move it to the SD model directory
sudo -u ubuntu git clone --depth 1 https://huggingface.co/stabilityai/stable-diffusion-2-1-base
cd stable-diffusion-2-1-base/
sudo -u ubuntu git lfs pull --include "v2-1_512-ema-pruned.ckpt"
sudo -u ubuntu git lfs install --force
cd ..
sudo mv stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt stable-diffusion-webui/models/Stable-diffusion/
sudo rm -rf stable-diffusion-2-1-base/

# download the corresponding config file and move it also to the model directory (make sure the name matches the model name)
wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml
cp v2-inference.yaml stable-diffusion-webui/models/Stable-diffusion/v2-1_512-ema-pruned.yaml

# change ownership of the web UI so that a regular user can start the server
sudo chown -R ubuntu:ubuntu stable-diffusion-webui/

# Setup log rotation
sudo cp /home/ubuntu/sd-webui-aws-deploy/sd-webui.logrotate /etc/logrotate.d/sd-webui

# Create a systemd service
sudo cp /home/ubuntu/sd-webui-aws-deploy/sd-webui.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable sd-webui.service
#   sudo systemctl start sd-webui.service

nohup bash webui.sh --listen --api --api-auth=username:password --loglevel 9 --use-cpu all --no-half --skip-torch-cuda-test --enable-insecure-extension-access >> /var/log/sd-webui.log &

