#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
cd ..
if [[ ! -d out ]]; then
    mkdir -p out
fi
make
