#!/bin/bash

# Installing System Dependencies
# Distro check Helper
if [ -f /etc/os-release ]; then
    # freedesktop.org and systemd
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    # linuxbase.org
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    # For some versions of Debian/Ubuntu without lsb_release command
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    # Older Debian/Ubuntu/etc.
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
    # Older SuSE/etc.
    ...
elif [ -f /etc/redhat-release ]; then
    # Older Red Hat, CentOS, etc.
    ...
else
    # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
    OS=$(uname -s)
    VER=$(uname -r)
fi
# Installing NodeJS 11.x and Dependencies
echo "${OS}"" was detected on ""${HOSTNAME}; Installing system dependencies"
if [ "${OS}" == "Ubuntu" ] ; then
  # Bare system dependencies
  sudo apt-get install -y mesa-utils libalut-dev libvorbis-dev cmake libxrender-dev libxrender1 libxrandr-dev zlib1g-dev libpng16-dev freeglut3 freeglut3-dev xvfb

  # Creating and accessing Sandbox folder for dependencies install
  if [ ! -d "dependecies" ]; then
    mkdir dependencies
  fi
  rm -rf ./dependencies/*
  cd dependencies

  PROC_COUNT="$(nproc)"
  # Installing PLIB dependecies
  echo "Installing PLIB 1.8.5"
  wget http://plib.sourceforge.net/dist/plib-1.8.5.tar.gz
  tar xvfz plib-1.8.5.tar.gz --same-owner
  cd plib-1.8.5
  export CFLAGS="-fPIC"
  export CPPFLAGS=$CFLAGS
  export CXXFLAGS=$CFLAGS
  ./configure
  make -j"${PROC_COUNT}"
  sudo make install
  export CFLAGS=
  export CPPFLAGS=
  export CXXFLAGS=
  cd ..
  rm -rf plib-1.8.5.tar.gz
  # End Installing PLIB

  # Installng OpenAL
  echo "Installing OpenAl 1.17.2"
  wget https://kcat.strangesoft.net/openal-releases/openal-soft-1.17.2.tar.bz2
  tar xfvj openal-soft-1.17.2.tar.bz2 --same-owner
  cd openal-soft-1.17.2/build
  cmake ..
  make -j"${PROC_COUNT}"
  sudo make install
  cd ../..
  rm -rf openal-soft-1.17.2.tar.bz2
fi

if [ "${OS}" == "CentOS Linux" ] ; then
  # System dependencies
  yum install -y mesa-libGL{,-devel} freealut{,-devel} libvorbis{,-devel} cmake3 libXrender{,-devel} libXrandr{,-devel} zlib{,-devel} libpng{,-devel}
fi

# Instaling Torcs itself
echo "Installing Torcs"
git clone https://github.com/dosssman/gym_torqs.git
cd gym_torqs
git checkout torqs_raceconfig
git reset --hard 1283706db42d8a4c1af5558644ed0114595bd51d
cd vtorcs-RL-color
./configure
# make -j"${PROC_COUNT}"
make
sudo make install
sudo make datainstall
cd ../../..

# Installing Anaconda if required
if [ -z $(which conda) ] ; then
  echo "Installing Anaconda"
else
  echo "Conda was detected, skipping installation"
fi

# Creating conda environment
# TODO Add CPU or GPU param for tensorflow
echo "Creating conda environment"
if [ -z $(conda info --envs |  grep "baselines-torcs") ]; then
	echo "Conda env not detected. Proceeding to creation:"
	# Create envs with necessary packages
	conda create -n "baselines-torcs" python=3.6 -y
	conda activate "baselines-torcs"
	conda install --yes tensorflow-gpu=1.8 gcc_linux-64 gxx_linux-64
	conda install -c conda-forge --yes tqdm
	conda install -y mpi4py
	pip install gym joblib psutil
  # Install the baselines itself
  pip install -e .
  # Install the GymTorcs dependency
  # This will be cloned to the current directory
  git clone https://github.com/dosssman/GymTorcs.git
  pip install -e GymTorcs/.
	conda deactivate
else
	echo "Env seems already installed."
fi

echo "Install complete"
