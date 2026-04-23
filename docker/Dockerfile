# syntax=docker/dockerfile:1.7-labs
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG WORKDIR="/workspace"
WORKDIR ${WORKDIR}

# Run apt-get (dkpg) without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

########################################
# SECTION 0: Create a new user         #
########################################
ARG USER_ID=1000
ARG USER=vr
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
  && rm -rf /var/lib/apt/lists/* \
  && useradd -m -s /bin/bash -u "${USER_ID}" "${USER}" \
  && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers


########################################
# SECTION 1: Essentials                #
########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl parallel \
    git cmake unzip \
    software-properties-common \
    # Setup locale language config \
    locales \
  && apt-get upgrade -y \
  && rm -rf /var/lib/apt/lists/* \
  # Setup locale language config \
  && locale-gen "en_US.UTF-8" \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  # Set timezone \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8


########################################
# SECTION 2: Install Steam & ALVR      #
########################################
RUN curl -sSL https://repo.steampowered.com/steam/archive/stable/steam.gpg \
    -o /usr/share/keyrings/steam.gpg \
  && echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/steam.gpg] "\
    "https://repo.steampowered.com/steam/ stable steam" \
    | tee /etc/apt/sources.list.d/steam-stable.list > /dev/null \
  && dpkg --add-architecture i386 \
  && apt-get update && apt-get install -y --no-install-recommends \
    -o Dpkg::Options::="--force-confold" \
    libgl1-mesa-dri:amd64 libgl1-mesa-dri:i386 \
    libgl1-mesa-glx:amd64 libgl1-mesa-glx:i386 \
    steam-launcher \
    # ALVR \
    build-essential pkg-config gcc yasm nasm curl \
    libclang-dev libssl-dev \
    pulseaudio-utils libasound2-dev libjack-dev libspeechd-dev libpipewire-0.3-dev \
    libgtk-3-dev libvulkan-dev libunwind-dev \
    libx264-dev libx265-dev \
    libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
    libxkbcommon-dev libdrm-dev libva-dev \
    libxkbcommon-x11-0 libsdl2-2.0-0 \
    libnvidia-encode-560 \
  && rm -rf /var/lib/apt/lists/*

# ----- Steam cmd ----- #
# ARG STEAMCMDDIR="/home/${USER}/steamcmd"
# USER ${USER}
# RUN mkdir -p "${STEAMCMDDIR}" \
#   && curl -fsSL 'https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz' | tar xvzf - -C "${STEAMCMDDIR}" \
#   && "${STEAMCMDDIR}/steamcmd.sh" +quit \
#   && ln -s "${STEAMCMDDIR}/linux32/steamclient.so" "${STEAMCMDDIR}/steamservice.so" \
#   && mkdir -p "/home/${USER}/.steam/sdk32" \
#   && ln -s "${STEAMCMDDIR}/linux32/steamclient.so" "/home/${USER}/.steam/sdk32/steamclient.so" \
#   && ln -s "${STEAMCMDDIR}/linux32/steamcmd" "${STEAMCMDDIR}/linux32/steam" \
#   && mkdir -p "/home/${USER}/.steam/sdk64" \
#   && ln -s "${STEAMCMDDIR}/linux64/steamclient.so" "/home/${USER}/.steam/sdk64/steamclient.so" \
#   && ln -s "${STEAMCMDDIR}/linux64/steamcmd" "${STEAMCMDDIR}/linux64/steam" \
#   && ln -s "${STEAMCMDDIR}/steamcmd.sh" "${STEAMCMDDIR}/steam.sh"
# USER root


########################################
# SECTION 3: Build ALVR                #
########################################
COPY --chown=${USER} --parents alvr_green ${WORKDIR}

USER ${USER}
# Install Rust using rustup
# Build ALVR
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y \
  && cd "${WORKDIR}/alvr_green" \
  && ./install.sh \
  && sed -i 's|^prefix=.*|prefix=/usr/local/cuda|' "./alvr/xtask/deb/cuda.pc" \
  && export PKG_CONFIG_PATH="$(pwd)/alvr/xtask/deb" \
  && export PATH="${HOME}/.cargo/bin:${PATH}" \
  && cargo xtask prepare-deps --platform linux \
  && cargo xtask build-streamer --release
USER root


##################################################
# SECTION 4: Install Python Dependencies w/ pip  #
##################################################
ARG PIP_REQ_PKG_DIR="."

COPY "${PIP_REQ_PKG_DIR}/requirements.txt" /tmp
COPY --chown=${USER} --parents README.md pyproject.toml vr_teleop ${WORKDIR}
RUN apt-get update && apt-get install -y --no-install-recommends \
    # python3 with up-to-date pip \
    python3 python3-pip python3-dev \
    # EigenPy dependencies \
    libeigen3-dev libboost-python-dev \
    # pinocchio dependencies \
    liburdfdom-dev liburdfdom-headers-dev libboost-filesystem-dev libboost-serialization-dev \
    # pyrealsense2 dependencies \
    libusb-1.0-0 \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install -r /tmp/requirements.txt \
  && rm -r /tmp/* \
  # EigenPy \
  && git clone --recursive --depth 1 --branch v3.9.0 https://github.com/stack-of-tasks/eigenpy.git \
  && cd eigenpy && mkdir build && cd build \
  && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF -DBUILD_TESTING_SCIPY=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
  && make -j && make install \
  && cd ../.. && rm -rf eigenpy \
  # pinocchio \
  && git clone --recursive --depth 1 --branch v3.2.0 https://github.com/stack-of-tasks/pinocchio.git \
  && cd pinocchio && mkdir build && cd build \
  && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
  && make -j && make install \
  && cd ../.. && rm -rf pinocchio \
  # Install VRTeleop as editable, append to PYTHONPATH because COPY does not include code \
  && python3 -m pip install -e "${WORKDIR}[dev]" --config-settings editable_mode=compat \
  && echo "\n# Append VRTeleop to PYTHONPATH\nexport PYTHONPATH=\"\${PYTHONPATH:+\${PYTHONPATH}:}${WORKDIR}\"" >> "/home/${USER}/.bashrc" \
  && python3 -m pip cache purge


##################################################
# SECTION 5: Install ManiSkill and SAPIEN        #
##################################################
COPY --chown=${USER} --parents 3rd_party/ManiSkill ${WORKDIR}
# Install in-dev packages (mani_skill, sapien)
RUN python3 -m pip install -e "${WORKDIR}/3rd_party/ManiSkill" --config-settings editable_mode=compat \
  && echo "\n# Append ManiSkill to PYTHONPATH\nexport PYTHONPATH=\"\${PYTHONPATH:+\${PYTHONPATH}:}${WORKDIR}/3rd_party/ManiSkill\"" >> /home/${USER}/.bashrc \
  && python3 -m pip install \
    https://github.com/$( \
      curl -s https://github.com/haosulab/SAPIEN/releases/expanded_assets/nightly | \
      sed -n 's/^.*<a .*href="\([^"]*\)".*>.*$/\1/p' | \
      grep -E "3$(python3 --version | \
      sed -n 's/^.*3\.\([0-9]*\)\..*$/\1/p').*linux.*x86_64.whl"\
    ) \
  && sapien install demo_arena \
  && python3 -m pip cache purge
# Sapien Vulkan requirement
# nvidia_icd.json: https://github.com/haosulab/ManiSkill/issues/9
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json


########################################
# SECTION N: Additional config & MISC  #
########################################
# Set user_entrypoint.sh as ENTRYPOINT for changing active user ID
ENV CHOWN_USER_DIRS="${CHOWN_USER_DIRS:+${CHOWN_USER_DIRS}:}${WORKDIR}"
COPY docker/user_entrypoint.sh /
ENTRYPOINT ["/user_entrypoint.sh"]
CMD ["/bin/bash"]
