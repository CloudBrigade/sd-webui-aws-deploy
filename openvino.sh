# Make sure Python version is 3.10+
python3 -m venv sd_env
source sd_env/bin/activate
git clone https://github.com/openvinotoolkit/stable-diffusion-webui.git
cd stable-diffusion-webui

export PYTORCH_TRACING_MODE=TORCHFX
#export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half" 

# Launch the WebUI
#./webui.sh 

