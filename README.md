# RL and IL based Hybrid Model for Torcs

# Preview

# Installation
## Prerequisite
  - Internet connection, the faster the better
  - Around 2GB free for the
  - sudo rights

An automated installation script is provided as `install_script.sh` which supports Ubuntu (and CentOS Linux but untested).

To have a smooth install, make sure to have installed Anaconda supporting python 3.6 version.
The script will install:
  - The Torcs binaries and required dependencies
  - Anaconda Virtual Environment management and the required virtual env

!!! Important: Before building the vtorcs-RL-color, edit the gym_torqs/vtorcs-RL-colors/src/interfaces/graphic.h, and change the z3r0 in /home/z3r0/... with your Linux user username.
Run `source install_script.sh` to install most of the required elements.

In case it doesn't work, consider checking out the content of the script
anbd following each step manually.

Also requires manual setting of environment variables
```
export TORCS_DATA_DIR="/home/$USER/torcs_data/" # Used to record player data, make sure the folder structure exists
export DISPLAY=:1 # Used by xvfb-run for headless training, namely on the server
```

# Training
Activate the previously created virtual env with
```
conda activate baselines-torcs
```

## DDPG
```
python -m baselines.torcs_ddpg.main
```
The result will be dumped in ./baselines/torcs_ddpg/result

## GAIL
```
python -m baselines.gail.run_mujoco
```
The result will be dumped in ./baselines/gail/result

# Hybrid (Remi)
```
python -m baselines.remi.run_mujoco

```
The result will be dumped in ./baselines/remi/result

## Training shortcuts (Bonus)

Creates shortcuts for fast training and checking. (Preferably add them to your ~/.bashrc file)
## Torcs DDPG
```
alias xvfb-torcs-ddpg-train="xvfb-run -a -s \"-screen $DISPLAY 640x480x24\" python -m baselines.ddpg_torqs.main" # Training
alias xvfb-torcs-ddpg-rec="xvfb-run -a -s \"-screen $DISPLAY 640x480x24\" python -m baselines.ddpg_torqs.record_data" # Recod data
```
## Torcs GAIL over Human player
```
alias xvfb-torcs-gail-train="xvfb-run -a -s \"-screen $DISPLAY 640x480x24\" python -m baselines.gail.run_mujoco" #Training
alias xvfb-torcs-gail-eval="xvfb-run -a -s \"-screen $DISPLAY 640x480x24\" python -m baselines.gail.gail-eval-torcs" # Evaluating
```
## Torcs Hybrid Model with offline DDPG

alias xvfb-torcs-remi-train="xvfb-run -a -s \"-screen $DISPLAY 640x480x24\" python -m baselines.remi.run_mujoco" # Training
alias xvfb-torcs-remi-eval="xvfb-run -a -s \"-screen $DISPLAY 640x480x24\" python -m baselines.remi.remi-eval" # Eval


# Play

## DDPG
```
python -m baselines.torcs_ddpg.play --checkpoint=/path/to/.../torcs-ddpg-XXXX-XX-XX-XX-XX-XXX-XXXXXX/model_data/epoch_XXXX.ckpt
```

## GAIL
```
python -m baselines.gail.play --load_model_path=baselines/gail/.../checkpoint/torcs_gail/torcs_gail_XXXX
```

## Hybrid (Remi)
```
python -m baselines.remi.play --load_model_path=baselines/remi/result/.../checkpoint/torcs_remi/torcs_remi_XXXX
```

## Playing Shorctus (Bonus)
Add the following to ~/.bashrc, and reload it.
```
alias torcs-ddpg-play="python -m baselines.ddpg_torqs.play" # DDPG
alias gail-torqs-play="python -m baselines.gail.play" # GAIL
alias remi2-torqs-play="python -m baselines.remi.play" # Hybrid (ReMi)
