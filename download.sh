#!/bin/bash

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
# Section 1: Download adbforwarder & alvr_green            #
############################################################
# Move to the folder of repo root, so later commands can use relative paths
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR"

python3 -m pip install gdown

if [ ! -d "adbforwarder" ] ; then
  gdown https://drive.google.com/uc?id=1rJYVEbVVyRdrQ2IWGZ9zLneW6QS_bQr7
  tar -xzf adbforwarder.tar.gz
  rm -rf adbforwarder.tar.gz
fi

if [ ! -d "alvr_green" ] ; then
  gdown https://drive.google.com/uc?id=1Krij5D1O6dGbqbBEXZX7J9imRGbvUOy3
  tar -xzf alvr_green.tar.gz
  rm -rf alvr_green.tar.gz
fi
