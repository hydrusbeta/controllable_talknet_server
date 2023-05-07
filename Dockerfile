# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies.
# Note: Binaries are not available for some python packages (e.g. pesq), so pip must compile them
# locally. This is why gcc, g++ and python3.8-dev are included in the list below.
# For some weird reason, you need both python3.8-venv and python3-venv to create a python3.8 
# virtual environment on Ubuntu 18. Cuda 11.8 is used instead of 12 for backwards compatibility. 
# Cuda 11.8 supports compute capability 3.5 through 9.0
FROM nvidia/cuda:11.8.0-base-ubuntu18.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone.
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    ffmpeg \
    python3.8-dev \
    python3.8-venv \
    python3-venv \
    wget

# todo: Is there a better way to refer to the home directory (~)?
ARG HOME_DIR=/root

# Download controllable_talknet and checkout a specific commit that is known to work with this docker 
# file and with Hay Say.
RUN git clone -b main --single-branch -q https://github.com/SortAnon/ControllableTalkNet ~/hay_say/controllable_talknet
WORKDIR $HOME_DIR/hay_say/controllable_talknet
RUN git reset --hard 5ee364f5bb1fe63fcde2b690507bd7cd89bfe268

# Download SortAnon's hifi-gan fork and checkout a specific commit that is known to work with this docker
# file and with Hay Say.
RUN git clone -b master --single-branch -q https://github.com/SortAnon/hifi-gan ~/hay_say/controllable_talknet/hifi-gan
WORKDIR $HOME_DIR/hay_say/controllable_talknet/hifi-gan
RUN git reset --hard 38d31dd0612fd7ca16153dba1f6caf0b8309aa3c

# Create virtual environments for SortAnon's Controllable TalkNet and Hay Say's
# controllable_talknet_server
RUN python3.8 -m venv ~/hay_say/.venvs/controllable_talknet; \
    python3.8 -m venv ~/hay_say/.venvs/controllable_talknet_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while
# we're at it to handle modules that use PEP 517, and install cython which is required for building
# other python packages. Specify a version number for numpy or else it will install one that conflicts
# with Controllable Talknet's requirements file.
RUN ~/hay_say/.venvs/controllable_talknet/bin/pip install --no-cache-dir --upgrade wheel pip cython numpy==1.19.5; \
    ~/hay_say/.venvs/controllable_talknet_server/bin/pip install --no-cache-dir --upgrade wheel pip cython numpy==1.19.5

# Specify versions of protobuf and hmmlearn in the requirements file to avoid dependency issues and add
# SortAnon's Nemo. Since the requirements file has been modified, we must also display a notice, as per
# the license terms.
RUN echo "protobuf==3.20.3" >> ~/hay_say/controllable_talknet/requirements.txt; \
    echo "hmmlearn==0.2.5" >> ~/hay_say/controllable_talknet/requirements.txt; \
    echo "git+https://github.com/SortAnon/NeMo.git@ef81d2e" >> ~/hay_say/controllable_talknet/requirements.txt; \
    sed -i '1 i\# This file was modified for the Hay Say project in April 2023. \n# This modified version is hereby released under the same GNU Affero General \n# Public License located at controllable_talknet/LICENSE, along with any \n# conditions added under section 7.\n\n' ~/hay_say/controllable_talknet/requirements.txt

# Install all python dependencies for controllable_talknet
RUN ~/hay_say/.venvs/controllable_talknet/bin/pip install --no-cache-dir -r ~/hay_say/controllable_talknet/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu111

# There is a weird dependency issue between pesq, numpy, numba, and NeMo. pesq somehow gets compiled
# against the wrong version of numpy when numba (a dependency of NeMo) is installed, so we must recompile
# it. This is a known issue described here: https://github.com/NVIDIA/NeMo/issues/3658
RUN ~/hay_say/.venvs/controllable_talknet/bin/python -m pip uninstall -y pesq; \
    ~/hay_say/.venvs/controllable_talknet/bin/python -m pip install --no-cache-dir pesq==0.0.2

# Add command line functionality to Controllable Talknet
RUN git clone https://github.com/hydrusbeta/controllable_talknet_command_line ~/hay_say/controllable_talknet_command_line && \
    cp ~/hay_say/controllable_talknet_command_line/command_line_interface.py ~/hay_say/controllable_talknet/;

# Download the Hay Say Interface code and install its dependencies
RUN git clone https://github.com/hydrusbeta/controllable_talknet_server ~/hay_say/controllable_talknet_server/ && \
    ~/hay_say/.venvs/controllable_talknet_server/bin/pip install --no-cache-dir -r ~/hay_say/controllable_talknet_server/requirements.txt;

# Expose port 6574, the port that Hay Say uses for controllable_talknet.
# Also expose port 8050, in case someone want to use the original Controllable TalkNet UI.
EXPOSE 6574
EXPOSE 8050

# Create the models directory. The server will place symbolic links in here that point to the
# actual model files.
RUN mkdir /root/hay_say/controllable_talknet/models

# Controllable Talknet downloads some models, e.g. the NeMo TTS phonemes model, when the 
# controllable_talknet module is first loaded. Let's Load it ahead of time now so the user doesn't need 
# to wait for them to download later and so they can run this architecture offline. Relative pathing is
# used in some modules, so we must set the working directory to controllable_talknet.
WORKDIR $HOME_DIR/hay_say/controllable_talknet
RUN ~/hay_say/.venvs/controllable_talknet/bin/python -c "import controllable_talknet"

# Run the Hay Say interface on startup
CMD ["/bin/sh", "-c", "/root/hay_say/.venvs/controllable_talknet_server/bin/python /root/hay_say/controllable_talknet_server/main.py"]
