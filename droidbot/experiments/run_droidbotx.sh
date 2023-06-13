#!/bin/bash

# Função para executar o comando DroidBot nos emuladores
function run_droidbot() {
  local apk_path="$1"
  local output_dir="$2"
  
  python3 start_q_learning.py -a $apk_path -d emulator-5554 -o $output_dir/saida-droidbot-5554 -is_emulator -policy gym -t 7200 &
  sleep 5
  python3 start_q_learning.py -a $apk_path -d emulator-5556 -o $output_dir/saida-droidbot-5556 -is_emulator -policy gym -t 7200 &
  sleep 5
  python3 start_q_learning.py -a $apk_path -d emulator-5558 -o $output_dir/saida-droidbot-5558 -is_emulator -policy gym -t 7200 &
  sleep 5
  python3 start_q_learning.py -a $apk_path -d emulator-5560 -o $output_dir/saida-droidbot-5560 -is_emulator -policy gym -t 7200 &
  sleep 5

}

function copy_coverage_files(){
  local local_results_dir="$1"
  local emulator_port="$2"
  local apk_name="$3"
  local coverage_dir="$local_results_dir/emulator_$emulator_port"
  local time="$4"

  adb -s emulator-$emulator_port pull /sdcard/coverage.ec $local_results_dir/$apk_name-$time-$emulator_port-coverage.ec
  
}

# Caminhos e diretórios
apk_name=AtimeTrack
apk_path=~/droidbot/experiments/apps/$apk_name.apk
output_dir=~/Documentos/test-novaDRL/output/$apk_name
local_results_dir=~/droidbot/experiments/results_cov

# Executar o comando DroidBot nos emuladores
run_droidbot "$apk_path" "$output_dir"

# Aguardar 10min
sleep 600
# Copiar o arquivo coverage.ec dos emuladores para a pasta local
for emulator_port in 5554 5556 5558 5560; do
  copy_coverage_files "$local_results_dir" "$emulator_port" "$apk_name" "10min"
done


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
