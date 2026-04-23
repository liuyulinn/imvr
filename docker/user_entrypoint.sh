#!/bin/bash
# This script sets up a new user with USER_ID and change directory ownership accordingly
#
# Copyright 2024-2025 Kolin Guo. All rights reserved. Authorized usage only.
#
# Author: Kolin Guo <guokolin@gmail.com>
#
# Version: 0.1.3
# Last modified: 2024-09-20

USER_NAME="${USER_NAME:-vr}"
DEBUG="${DEBUG:-}" # Debug printing

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
echo_debug() {
  if [ -n "$DEBUG" ]; then
    echo "$@"
  fi
}

# Function to show progress_bar
# Input is currentState($1) and totalState($2)
# Output example:
# Progress [1000/1000] : [########################################] 100%
progress_bar() {
  # Process data
  declare -i _progress=(${1}*100/${2}*100)/100
  declare -i _done=(${_progress}*4)/10
  declare -i _left=40-$_done
  _fill=$(printf "%${_done}s")
  _empty=$(printf "%${_left}s")

  # 1.2 Build progressbar strings and print the ProgressBar line
  # 1.2.1 Output example:
  # 1.2.1.1 Progress : [########################################] 100%
  printf "\rProgress [${1}/${2}] : [${_fill// /#}${_empty// /-}] ${_progress}%%"
}

# Faster version of chown when running on HDD
# * Runs chown -R first, kills it if it takes more than 0.1 second.
# * Then runs find . | parallel --bar chown OWNER
# Note: no group support for now.
#
# Input is OWNER($1) and directory($2)
chown_fast() {
  chown -R "$1" "$2" 2>/dev/null &
  CHOWN_PID=$!

  sleep 0.1 # Wait for 0.1 second

  # Check if the process is still running
  if kill -0 "$CHOWN_PID" 2>/dev/null; then
    echo_debug "chown -R is taking too long. Killing the process..."
    kill -9 "$CHOWN_PID"

    echo_debug "Using find and parallel chown..."
    find "$2" ! -user "$1" -print0 2>/dev/null | parallel -0 --bar "chown ${1} 2>/dev/null"
  else
    wait "$CHOWN_PID"
  fi
}

############################################################
# Section 2: Get USER_ID from environment variable         #
############################################################
echo_debug -e "---------- Begin of user_entrypoint.sh ----------"

# If USER_ID is null, check if SSH_AUTH_SOCK is defined and matches the pattern
# This is a workaround since docker compose.yaml cannot get host user id automatically.
if [ -z "${USER_ID:-}" ]; then
  if [[ ${SSH_AUTH_SOCK:-} =~ ^/run/user/([0-9]+)/keyring/ssh$ ]]; then
    USER_ID="${BASH_REMATCH[1]}"
  elif [ -z "${SSH_AUTH_SOCK:-}" ]; then
    echo -e "\e[31mPlease provide USER_ID or SSH_AUTH_SOCK environment variable.\e[0m"
    exit 1
  else
    echo -e "\e[31mSSH_AUTH_SOCK environment variable has unexpected format: '${SSH_AUTH_SOCK}'\e[0m"
    exit 2
  fi
fi
echo_debug -e "Received USER_ID: \e[33m${USER_ID}\e[0m"

############################################################
# Section 3: Create user or change user ID                 #
############################################################
# Check if user with USER_NAME exists
if id -u "$USER_NAME" &>/dev/null; then
  current_uid=$(id -u "$USER_NAME")

  # Check if the user with USER_NAME does not already have USER_ID
  if [ "$current_uid" -ne "$USER_ID" ]; then
    trap - ERR # Turn off ERR trap to ignore find errors
    declare -i total_files=$(find /home -uid "$current_uid" 2>/dev/null | wc -l)
    trap 'catch' ERR # Trap all errors (status != 0) and call catch()

    # If too many file need to be updated,
    # Run usermod in background and print progress_bar based on # of files
    if [ "$total_files" -gt "1000" ]; then
      echo -e "\e[36mFound too many files while changing user ID (usermod). Please wait for it to finish.\e[0m"

      usermod -u "$USER_ID" "$USER_NAME" &
      PID=$!
      while kill -0 "$PID" 2>/dev/null; do
        trap - ERR # Turn off ERR trap to ignore find errors
        declare -i remaining_files=$(find /home -uid "$current_uid" 2>/dev/null | wc -l)
        trap 'catch' ERR # Trap all errors (status != 0) and call catch()
        declare -i finished_files=${total_files}-${remaining_files}
        progress_bar ${finished_files} ${total_files}
        sleep 2
      done
      wait "$PID"
      echo -e "\n\e[36mFinished changing ID of user '${USER_NAME}' to ${USER_ID}.\e[0m\n"
    else
      usermod -u "$USER_ID" "$USER_NAME"
    fi

    groupmod -g "$USER_ID" "$USER_NAME"
    chown -R "$USER_NAME":"$USER_NAME" "/home/${USER_NAME}"
  fi
else
  # Create user if it user with USER_NAME doesn't exist
  useradd -m -s /bin/bash -u "$USER_ID" "$USER_NAME"
  echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers
fi

############################################################
# Section 4: Change directory ownership                    #
############################################################
chown -R "$USER_NAME":"$USER_NAME" "/run/user/${USER_ID}"
echo_debug -e "----- Begin changing directory ownership -----"
ORIGINAL_IFS=$IFS
IFS=':'
for dir in ${CHOWN_USER_DIRS:-}; do
  echo_debug -e "Processing ${dir} ..."
  if [ -d "$dir" ]; then
    declare -i total_files=$(find "$dir" ! -user "$USER_NAME" 2>/dev/null | wc -l)
    if [ "$total_files" -gt "0" ]; then
      echo "chown ${total_files} files in ${dir} ..."
      chown_fast "$USER_NAME" "$dir"
    fi
  fi
done
IFS=$ORIGINAL_IFS
unset CHOWN_USER_DIRS

############################################################
# Section N: Execute the command as the created user       #
############################################################
echo_debug -e "---------- End of user_entrypoint.sh ----------\n"
exec sudo -u "$USER_NAME" -HE --preserve-env=BASH_ENV /bin/bash -c "$@"
