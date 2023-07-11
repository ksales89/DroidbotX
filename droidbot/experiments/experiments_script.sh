#!/bin/bash

function install_apk() {
  local apk_path="$1"
  local emulator_port="5554"

  adb kill-server
  sleep 5
  adb start-server
  sleep 5

  ~/Android/Sdk/emulator/emulator -avd "emulator-$emulator_port" -port $emulator_port -wipe-data -no-snapshot-save &
  sleep 30 # Wait for the emulator to start

  adb -s "emulator-$emulator_port" install -g -r -d -t $apk_path
  sleep 10 # Wait for the APK installation
}

function run_droidbot() {
  local apk_path="$1"
  local output_dir="$2"
  local emulator_port="5554"

  python3 ~/DroidbotX/droidbot/start_q_learning.py -a $apk_path -d "emulator-$emulator_port" -is_emulator -o "$output_dir/saida-droidbot-$emulator_port" -policy gym -t 7200 &
  #droidbot -a $apk_path -d emulator-$emulator_port -is_emulator -o $output_dir/saida-droidbot-$emulator_port -t 7200 &
  sleep 5
}

# Function to copy coverage files
function copy_coverage_files() {
  local local_results_dir="$1"
  local apk_name="$2"
  local emulator_port="5554"
  local exec="$3"

  adb -s emulator-$emulator_port pull /sdcard/coverage.ec $local_results_dir/$apk_name-$exec-$((i * 10))min-coverage.ec
  echo "Pulled in: $local_results_dir/$apk_name-$((i * 10))min-$emulator_port-coverage.ec"
}

# Directories
apps_directory="$HOME/DroidbotX/droidbot/experiments/apps"
output_directory="$HOME/Documentos/experiments/outputX"
local_results_directory="$HOME/DroidbotX/droidbot/experiments/results_cov/all_coverage"

# Find all APKs in the "apps" directory
apk_paths=$(find "$apps_directory" -name "*.apk")

# Process each APK found
for apk_path in $apk_paths; do
  apk_name=$(basename "$apk_path" .apk)

  echo "APK Path: $apk_path"
  echo "APK name: $apk_name"

  # Create output directory for each APK
  output_dir="$output_directory/$apk_name"
  mkdir -p "$output_dir"

  # Repeat the process four times
  for ((j = 1; j <= 2; j++)); do
    echo "Iteration: $j"

    # Clean up previous files
    rm -f "$HOME/DroidbotX/fdroid/q_function.npy" \
        "$HOME/DroidbotX/fdroid/states.json" \
        "$HOME/DroidbotX/fdroid/transition_function.npy" \
        "$HOME/DroidbotX/q_function.npy" \
        "$HOME/DroidbotX/states.json" \
        "$HOME/DroidbotX/transition_function.npy"

    # Install APK on emulator
    install_apk "$apk_path"

    # Run DroidBot on emulator
    run_droidbot "$apk_path" "$output_dir"

    # Copy coverage files from emulator to local folder
    for ((i = 0; i <= 12; i++)); do
      copy_coverage_files "$local_results_directory" "$apk_name" "$j"  
      sleep 600 # Wait for 10 minutes
    done
  done
done
