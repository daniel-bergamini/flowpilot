#!/bin/bash

tmux kill-window -t 0

cd flowpilot
git pull
git lfs pull

export LOGPRINT=debug
export LOGLEVEL=debug

git branch stable

./launch_flowpilot_full.sh
