#!/bin/bash

# Função para instalar APK nos emuladores
function open_emulator() {
  local apk_path="$1"
  adb kill-server
  sleep 5
  adb start-server
  sleep 5

  # Iniciar emulador
  ~/Android/Sdk/emulator/emulator -avd emulator-teste1 -port 5554 -wipe-data -no-snapshot-save &
  sleep 30 # Aguardar o emulador iniciar
  ~/Android/Sdk/emulator/emulator -avd emulator-teste2 -port 5556 -wipe-data -no-snapshot-save &
  sleep 30 # Aguardar o emulador iniciar
  ~/Android/Sdk/emulator/emulator -avd emulator-teste3 -port 5558 -wipe-data -no-snapshot-save &
  sleep 30 # Aguardar o emulador iniciar
  ~/Android/Sdk/emulator/emulator -avd emulator-teste4 -port 5560 -wipe-data -no-snapshot-save &
  sleep 30 # Aguardar o emulador iniciar
  ~/Android/Sdk/emulator/emulator -avd emulator-teste5 -port 5562 -wipe-data -no-snapshot-save &
  sleep 30 # Aguardar o emulador iniciar


}

# Iniciar emuladores e instalar APK
open_emulator
