#!/bin/bash

if [ -z "$mod_name" ]; then
  echo "Error: No mod specified. Please provide a mod name (e.g., car, horse)."
  exit 1
fi

if [[ -z "${checkpoint_paths[$mod_name]}" ]]; then
  echo "Error: Mod '$mod_name' is not found in the configuration. Please provide a valid mod name."
  exit 1
fi
