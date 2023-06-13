#!/bin/bash

# Função para executar o comando DroidBot nos emuladores
function run_droidbot() {
  local apk_path="$1"
  local output_dir="$2"


  droidbot -a $apk_path -d emulator-5554 -is_emulator -o $output_dir/saida-droidbot-5554 -t 7200 &
  sleep 5
 
  droidbot -a $apk_path -d emulator-5556 -is_emulator -o $output_dir/saida-droidbot-5556 -t 7200 &
  sleep 5

  droidbot -a $apk_path -d emulator-5558 -is_emulator -o $output_dir/saida-droidbot-5558 -t 7200 &
  sleep 5

  droidbot -a $apk_path -d emulator-5560 -is_emulator -o $output_dir/saida-droidbot-5560 -t 7200 &
  sleep 5

}

function copy_coverage_files(){
  local local_results_dir="$1"
  local emulator_port="$2"
  local apk_name="$3"
  local coverage_dir="$local_results_dir/emulator_$emulator_port"
  local time="$4"

  #adb -s emulator-$emulator_port pull /sdcard/coverage.ec $local_results_dir/$time-$emulator_port-$apk_name.ec
  adb -s emulator-$emulator_port pull /sdcard/coverage.ec $local_results_dir/$apk_name-$time-$emulator_port-coverage.ec
  
}

# Caminhos e diretórios
apk_name=farmerdiary-debug
apk_path=~/droidbot/experiments/apps/$apk_name.apk
output_dir=~/Documentos/experiments/output/$apk_name
local_results_dir=~/droidbot/experiments/results_cov


# Iniciar emuladores e instalar APK
# install_apk "$apk_path"

# Executar o comando DroidBot nos emuladores
run_droidbot "$apk_path" "$output_dir"

# Aguardar 1 horas (3600 segundos)
sleep 3600

# Copiar o arquivo coverage.ec dos emuladores para a pasta local
for emulator_port in 5554 5556 5558 5560; do
  copy_coverage_files "$local_results_dir" "$emulator_port" "$apk_name" "parcial"
done

sleep 3600

for emulator_port in 5554 5556 5558 5560; do
  copy_coverage_files "$local_results_dir" "$emulator_port" "$apk_name" "final"
done
