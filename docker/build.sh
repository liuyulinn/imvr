#!/bin/bash
# This script builds docker images for ImVR
#
# Copyright 2024-2025 Kolin Guo. All rights reserved. Authorized usage only.
#
# Author: Kolin Guo <guokolin@gmail.com>

############################################################
# Section 0.5: Bash Error Handling                         #
############################################################
# -e: exit immediately on non-zero status
# -E: trap on ERR is inherited by functions, command substitutions, and subshell envs
# -u: errors out when using an already unset variable ("unbound variable")
# -o pipefail: uses the status of last command with non-zero status as pipeline's status
set -eEu -o pipefail
trap 'catch' ERR # Trap all errors (status != 0) and call catch()
catch() {
  local err="$?"
  local err_command="$BASH_COMMAND"
  set +xv # disable trace printing

  echo -e "\nCaught error in ${BASH_SOURCE[1]}:${BASH_LINENO[0]} ('${err_command}' exited with status ${err})" >&2
  echo "Traceback (most recent call last, command might not be complete):" >&2
  for ((i = 0; i < ${#FUNCNAME[@]} - 1; i++)); do
    local funcname="${FUNCNAME[$i]}"
    [ "$i" -eq "0" ] && funcname=$err_command
    echo -e "  ($i) ${BASH_SOURCE[$i+1]}:${BASH_LINENO[$i]}\t'${funcname}'" >&2
  done
  exit "$err"
}

############################################################
# Section 1: Helper Function Definition                    #
############################################################
# Function to check if a path matches any pattern in $2
# Input is path($1) and pattern($2) separated by ':'
path_matched() {
  local path="$1"
  ORIGINAL_IFS=$IFS
  IFS=':'
  read -ra KEEP_ARRAY <<< "$2"
  for keep_path in "${KEEP_ARRAY[@]}"; do
    if [[ "$path" == $keep_path ]]; then
      IFS=$ORIGINAL_IFS
      return 0  # Path should be kept
    fi
  done
  IFS=$ORIGINAL_IFS
  return 1  # Path should not be kept
}

# Function to extract environment variable names from docker inspect output
# Input is container/image_name($1)
get_env_vars() {
  docker inspect --format '{{range $index, $value := .Config.Env}}{{println $value}}{{end}}' "$1" | cut -d '=' -f 1
}

############################################################
# Section 2: Initializing steam & SteamVR                  #
############################################################
# Move to the folder of repo root, so later commands can use relative paths
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$(dirname "$SCRIPT_PATH")")
cd "$SCRIPT_DIR"

# Keep these volumes in the final image. Remove all other mounted volumes.
KEEP_VOLUMES="/VRTeleop*"

docker compose build base

CONTAINER_NAME="vrteleop_steam_build"
IMAGE_NAME="kolinguo/vrteleop:latest"

if [ 1 -eq $(docker container ls -a | grep -w "${CONTAINER_NAME}$" | wc -l) ] ; then
  docker rm -f "$CONTAINER_NAME"
fi
if [ 1 -eq $(docker container ls -a | grep -w "${CONTAINER_NAME}_rmvol$" | wc -l) ] ; then
  docker rm -f "${CONTAINER_NAME}_rmvol"
fi
# Update steam by running it once. Close the login window to continue.
# Run alvr_dashboard exactly 3 times to initialize it. The goals for each invocation are:
# * Click 'Launch SteamVR' to initialize SteamVR. Close SteamVR windows first and then ALVR Dashboard.
# * Click 'Launch SteamVR' and add/trust ALVR client (Quest3 / Vision Pro) at 127.0.0.1.
#   Change Settings > Connection > Stream protocol > Set to TCP.
#   Close ALVR Dashboard.
# * Click 'Launch SteamVR' and test successful streaming to ALVR client (Quest3 / Vision Pro)
CMD_INIT="steam; "\
"/VRTeleop/adbforwarder/ADBForwarder & "\
"cd /VRTeleop/alvr_green; "\
"./build/alvr_streamer_linux/bin/alvr_dashboard; "\
"./build/alvr_streamer_linux/bin/alvr_dashboard; "\
"./build/alvr_streamer_linux/bin/alvr_dashboard; "\
"sed -i '/cargo/d' ~/.bashrc && "\
"echo -e '\n# Rust\nexport CARGO_HOME=/opt/rust/cargo RUSTUP_HOME=/opt/rust/rustup\n. \"\${CARGO_HOME:-\${HOME}/.cargo}/env\"' >> ~/.bashrc && "\
"sudo mkdir -p /opt/rust && "\
"sudo mv -v ~/.cargo /opt/rust/cargo && "\
"sudo mv -v ~/.rustup /opt/rust/rustup && "\
"sudo chown -R \${USER}:\${USER} /opt/rust && "\
"sed -i 's|\$HOME/.cargo/bin|/opt/rust/cargo/bin|g' /opt/rust/cargo/env"
docker compose run --name "$CONTAINER_NAME" base "$CMD_INIT"

# Get environment variables from the image
IMAGE_ENV_VARS=$(get_env_vars "$IMAGE_NAME")

echo -e "\e[1;36mCommitting 1st round...\e[0m"
docker commit "$CONTAINER_NAME" "$IMAGE_NAME"

echo -e "\e[1;36mRemoving mounted volumes...\e[0m"
# Get all mounted volumes
VOLUMES=$(docker inspect -f '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' "$CONTAINER_NAME")
CMD_REMOVE="rm -rf /run/user/*"
# Iterate through volumes and remove those not in KEEP_VOLUMES
for volume in $VOLUMES; do
  if ! path_matched "$volume" "$KEEP_VOLUMES"; then
    echo -e "  Removing \e[1;33m${volume}\e[0m"
    CMD_REMOVE+=" ${volume}"
  else
    echo -e "  Keeping \e[1;32m${volume}\e[0m"
  fi
done
docker run --name "${CONTAINER_NAME}_rmvol" --entrypoint /bin/bash "$IMAGE_NAME" -c "$CMD_REMOVE"

echo -e "\e[1;36mSet container-defined env variables to null...\e[0m"
# Get environment variables from the container
CONTAINER_ENV_VARS=$(get_env_vars "${CONTAINER_NAME}_rmvol")
# Compare environment variables
UNSET_ENV_VARS=$(comm -23 <(echo "$CONTAINER_ENV_VARS" | sort) <(echo "$IMAGE_ENV_VARS" | sort))
echo -e "Environment variables in container but not in image:\n\e[1;33m${UNSET_ENV_VARS}\e[0m"

echo -e "\e[1;36mCommitting final round...\e[0m"
docker commit \
  -c "ENV $(for var in $UNSET_ENV_VARS; do echo -n "$var= "; done)" \
  -c "ENTRYPOINT [\"/user_entrypoint.sh\"]" \
  -c "CMD [\"/bin/bash\"]" \
  "${CONTAINER_NAME}_rmvol" "$IMAGE_NAME"

docker rm -f "${CONTAINER_NAME}" "${CONTAINER_NAME}_rmvol"

NEW_IMAGE_NAME="${IMAGE_NAME/%latest/$(date +"%Y%m%d_%H%M%S")}"
docker tag "$IMAGE_NAME" "$NEW_IMAGE_NAME"
docker push "$NEW_IMAGE_NAME"
docker push "$IMAGE_NAME"
