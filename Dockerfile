# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies.
# Note: Binaries are not available for some python packages (e.g. pesq), so pip must compile them
# locally. This is why gcc, g++ and python3.8-dev are included in the list below.
# For some weird reason, you need both python3.8-venv and python3-venv to create a python3.8 
# virtual environment on Ubuntu 18. Cuda 11.8 is used instead of 12 for backwards compatibility. 
# Cuda 11.8 supports compute capability 3.5 through 9.0
FROM nvidia/cuda:11.8.0-base-ubuntu18.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    ffmpeg \
    python3.8-dev \
    python3.8-venv \
    python3-venv \
    wget

# Switch to a limited user
ARG LIMITED_USER=luna
RUN useradd --create-home --shell /bin/bash $LIMITED_USER
USER $LIMITED_USER

# Some Docker directives (such as COPY and WORKDIR) and linux command options (such as wget's directory-prefix option)
# do not expand the tilde (~) character to /home/<user>, so define a temporary variable to use instead.
ARG HOME_DIR=/home/$LIMITED_USER

# Download the VQGAN and HiFi-GAN reconstruction models and the super resolution HiFi-GAN model.
RUN mkdir -p ~/hay_say/temp_downloads/pretrained_models && \
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wlilvBtlBiAUEqqdqE0AEqo-UKx2X_cL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wlilvBtlBiAUEqqdqE0AEqo-UKx2X_cL" -O ~/hay_say/temp_downloads/pretrained_models/vqgan32_universal_57000.ckpt && rm -rf /tmp/cookies.txt &&\
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12gRIdg65xWiSScvFUFPT5JoPRsijQN90' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12gRIdg65xWiSScvFUFPT5JoPRsijQN90" -O ~/hay_say/temp_downloads/pretrained_models/hifirec && rm -rf /tmp/cookies.txt &&\
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa" -O ~/hay_say/temp_downloads/pretrained_models/hifisr && rm -rf /tmp/cookies.txt

# Create virtual environments for SortAnon's Controllable TalkNet and Hay Say's
# controllable_talknet_server
RUN python3.8 -m venv ~/hay_say/.venvs/controllable_talknet; \
    python3.8 -m venv ~/hay_say/.venvs/controllable_talknet_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while
# we're at it to handle modules that use PEP 517, and install cython which is required for building
# other python packages. Specify a version number for numpy or else it will install one that conflicts
# with Controllable Talknet's requirements file.
RUN ~/hay_say/.venvs/controllable_talknet/bin/pip install --timeout=300 --no-cache-dir --upgrade wheel pip cython numpy==1.19.5; \
    ~/hay_say/.venvs/controllable_talknet_server/bin/pip install --timeout=300 --no-cache-dir --upgrade wheel pip cython numpy==1.19.5

# Install all python dependencies for controllable_talknet.
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# ControllableTalkNet code itself. Cloning the repo after installing the requirements helps the Docker cache optimize
# build time. See https://docs.docker.com/build/cache
RUN ~/hay_say/.venvs/controllable_talknet/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    numpy==1.20.1 \
    scipy==1.7.0 \
    tensorflow==2.7.4 \
    dash==1.21.0 \
    dash-bootstrap-components==0.13.0 \
    jupyter-dash==0.4.0 \
    psola==0.0.1 \
    wget==3.2 \
    unidecode==1.2.0 \
    pysptk==0.2.1 \
    frozendict==2.0.3 \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0+cu113 \
    torchtext==0.12.0 \
    torch_stft==0.1.4 \
    kaldiio==2.18.0 \
    pydub==0.25.1 \
    pyannote.audio==1.1.2 \
    g2p_en==2.1.0 \
    pesq==0.0.2 \
    pystoi==0.3.3 \
    crepe==0.0.14 \
    resampy==0.2.2 \
    ffmpeg-python==0.2.0 \
    tqdm==4.65.0 \
    gdown==4.6.0 \
    editdistance==0.5.3 \
    ipywidgets==7.6.3 \
    torchcrepe==0.0.20 \
    taming-transformers-rom1504==0.0.6 \
    einops==0.3.2 \
    tensorflow-hub==0.12.0 \
    flask==2.0.3 \
    werkzeug==2.0.3 \
    jinja2==3.0.1 \
    astroid==2.5.6 \
    pytorch-lightning==1.6.0 \
    torchmetrics==0.6.0 \
    protobuf==3.19.6 \
    hmmlearn==0.3.0 \
    git+https://github.com/SortAnon/NeMo.git@ef81d2e

# There is a weird dependency issue between pesq, numpy, numba, and NeMo. pesq somehow gets compiled
# against the wrong version of numpy when numba (a dependency of NeMo) is installed, so we must recompile
# it. This is a known issue described here: https://github.com/NVIDIA/NeMo/issues/3658
RUN ~/hay_say/.venvs/controllable_talknet/bin/python -m pip uninstall -y pesq; \
    ~/hay_say/.venvs/controllable_talknet/bin/python -m pip install --timeout=300 --no-cache-dir pesq==0.0.2

# Install the dependencies for the Hay Say interface code.
RUN ~/hay_say/.venvs/controllable_talknet_server/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    hay_say_common==1.0.8 \
    jsonschema==4.19.1

# Clone controllable_talknet and checkout a specific commit that is known to work with this docker
# file and with Hay Say.
RUN git clone -b main --single-branch -q https://github.com/SortAnon/ControllableTalkNet ~/hay_say/controllable_talknet
WORKDIR $HOME_DIR/hay_say/controllable_talknet
RUN git reset --hard a6f2b53442fb58925a652aae410fa048321d8359

# Clone SortAnon's hifi-gan fork and checkout a specific commit that is known to work with this docker
# file and with Hay Say.
RUN git clone -b master --single-branch -q https://github.com/SortAnon/hifi-gan ~/hay_say/controllable_talknet/hifi-gan
WORKDIR $HOME_DIR/hay_say/controllable_talknet/hifi-gan
RUN git reset --hard 42c270d4f79a6966edf92ef9ee17e2bc8b9977b5

# Clone the Hay Say Interface code.
RUN git clone -b main --single-branch https://github.com/hydrusbeta/controllable_talknet_server ~/hay_say/controllable_talknet_server/

# Remove all usages of tensorflow. The presence of Tensorflow in the code is known to cause errors with at least one
# low-end GPU.
# Leaving this commented out for now because torchcrepe returns slightly different results from crepe.
# RUN sed -i -e '11d;37,38d' ~/hay_say/controllable_talknet/controllable_talknet.py &&\
#     sed -i -e '484,485d' ~/hay_say/controllable_talknet/core/extract.py &&\
#     sed -i '484 i\        device = "cuda" if torch.cuda.is_available() else "cpu"\n        output_freq = torchcrepe.predict(\n            audio_torch.type(torch.int16).type(torch.float32),\n            22050,\n            hop_length=256,\n            fmin=50,\n            fmax=800,\n            model="full",\n            decoder=torchcrepe.decode.viterbi,\n            # return_periodicity=True,\n            batch_size=128,\n            device=device,\n        )\n        output_freq = output_freq.squeeze(0).cpu().numpy()[: len(f0s_wo_silence)]\n' ~/hay_say/controllable_talknet/core/extract.py &&\
#     sed -i -e '28d;31,32d;329,394d;462,479d' ~/hay_say/controllable_talknet/core/extract.py &&\
#     ~/hay_say/.venvs/controllable_talknet/bin/pip uninstall -y tensorflow tensorflow-hub crepe

# Modify the input text to eliminate repeated periods separated by any number of spaces, to avoid a bug in NeMo:
RUN sed -i 's/input.replace(".", ". ")/re.sub(r"\\.(\\s*\\.)*", ". ", input)/' ~/hay_say/controllable_talknet/core/extract.py && \
    sed -i '1 i\# This file has been modified for the Hay Say project around November 2023.\n\n# As required by the license for Controllable TalkNet, this modified version of\n# Controllable TalkNet is released under the GNU Affero General Public License\n# along with the additional conditions under section 7 of the original license,\n# which can be found here:\n#   https://github.com/SortAnon/ControllableTalkNet/blob/128b1c18b4d77a1726414bab764268d3e6dd68ca/LICENSE\n\n' ~/hay_say/controllable_talknet/core/extract.py

# Create the models and results directories. The server will place symbolic links in the models directory that point to
# the actual model files. The VQGAN, Hi-fidelity reconstruction and super-resolution HiFi-GAN models also go in there.
# Controllable TalkNet will write files in the results directory before controllable_talknet_server transfers them
# elsewhere.
RUN mkdir ~/hay_say/controllable_talknet/models && \
    mkdir ~/hay_say/controllable_talknet/results

# Expose port 6574, the port that Hay Say uses for controllable_talknet.
# Also expose port 8050, in case someone want to use the original Controllable TalkNet UI.
EXPOSE 6574
EXPOSE 8050

# Move the VQGAN and HiFi-GAN reconstruction models and the super resolution HiFi-GAN model to the expected directory:
RUN mv ~/hay_say/temp_downloads/pretrained_models/* ~/hay_say/controllable_talknet/models/

# Controllable Talknet downloads some models, e.g. the NeMo TTS phonemes model, when the
# controllable_talknet module is first loaded. Let's Load it ahead of time now so the user doesn't need
# to wait for them to download later and so they can run this architecture offline. Relative pathing is
# used in some modules, so we must set the working directory to controllable_talknet.
WORKDIR $HOME_DIR/hay_say/controllable_talknet
RUN ~/hay_say/.venvs/controllable_talknet/bin/python -c "import controllable_talknet"

# Run the Hay Say interface on startup
CMD ["/bin/sh", "-c", "~/hay_say/.venvs/controllable_talknet_server/bin/python ~/hay_say/controllable_talknet_server/main.py --cache_implementation file"]
