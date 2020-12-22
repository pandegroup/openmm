set -euxo pipefail

sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update -qq

CUDA_APT=${CUDA_VERSION/./-}
if [[ ${CUDA_VERSION} == 10.* ]]; then CUFFT="cuda-cufft"; else CUFFT="libcufft"; fi
sudo apt-get install -y \
libgl1-mesa-dev cuda-compiler-${CUDA_APT} \
cuda-drivers cuda-driver-dev-${CUDA_APT} \
cuda-cudart-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} \
${CUFFT}-${CUDA_APT} ${CUFFT}-dev-${CUDA_APT} \
cuda-nvprof-${CUDA_APT}
sudo apt-get clean

export CUDA_HOME=/usr/local/cuda-${CUDA_APT/-/.}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export PATH=${CUDA_HOME}/bin:${PATH}

echo "CUDA_HOME=${CUDA_HOME}" >> ${GITHUB_ENV}
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ${GITHUB_ENV}
echo "PATH=${PATH}" >> ${GITHUB_ENV}