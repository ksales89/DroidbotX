#!/bin/bash

function install_apk() {
  local apk_path="$1"
  local emulator_ports=("5554" "5556" "5558" "5560")

  adb kill-server
  sleep 5
  adb start-server
  sleep 5

  for port in "${emulator_ports[@]}"; do
    ~/Android/Sdk/emulator/emulator -avd "emulator-$port" -port $port -wipe-data -no-snapshot-save &
    sleep 30 # Aguardar o emulador iniciar

    adb -s "emulator-$port" install -g -r -d -t $apk_path
    sleep 10 # Aguardar a instalação do APK
  done
}

function run_droidbot() {
  local apk_path="$1"
  local output_dir="$2"
  local emulator_ports=("5554" "5556" "5558" "5560")

  for port in "${emulator_ports[@]}"; do
    python3 ~/DroidbotX/droidbot/start_q_learning.py -a $apk_path -d "emulator-$port" -is_emulator -o "$output_dir/saida-droidbot-$port" -policy gym -t 7200 &
    #droidbot -a $apk_path -d emulator-$port -is_emulator -o $output_dir/saida-droidbot-$port -t 7200 &
    sleep 5
  done
}
# Loop para salvar o arquivo de cobertura a cada 10 minutos durante 2 horas
function copy_coverage_files() {
  local local_results_dir="$1"
  local apk_name="$2"
  
  for emulator_port in 5554 5556 5558 5560; do
    adb -s emulator-$emulator_port pull /sdcard/coverage.ec $local_results_dir/$apk_name-$((i * 10))min-$emulator_port-coverage.ec
    echo "Pulled in: $local_results_dir/$apk_name-$((i * 10))min-$emulator_port-coverage.ec"
  done
}

# Caminhos e diretórios
apps_directory="$HOME/DroidbotX/droidbot/experiments/apps"
output_directory="$HOME/Documentos/experiments/outputX_Multi"
local_results_directory="$HOME/DroidbotX/droidbot/experiments/results_cov/all_coverage"

# Encontrar todos os APKs na pasta "apps"
apk_paths=$(find "$apps_directory" -name "*.apk")

# Percorrer cada APK encontrado
for apk_path in $apk_paths; do
  apk_name=$(basename "$apk_path" .apk)

  echo "APK Path: $apk_path"
  echo "APK name: $apk_name"

  # Criar diretório de saída para cada APK
  output_dir="$output_directory/$apk_name"
  mkdir -p "$output_dir"

  # Iniciar emuladores e instalar APK
  install_apk "$apk_path"

  # Executar o comando DroidBot nos emuladores
  run_droidbot "$apk_path" "$output_dir"

  # Copiar o arquivo coverage.ec dos emuladores para a pasta local
  for ((i = 0; i <= 12; i++)); do
    copy_coverage_files "$local_results_directory" "$apk_name"   
    sleep 600 # Aguardar 10 minutos
  done
  
done
