#!/usr/bin/env bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH=$(dirname "$0")/../src/build/
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH=../src/build
fi
# TODO windows