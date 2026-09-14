# Sourced by the repository build.sh after selecting SRC_DIR.
# This environment runs its C simulator on CPU and its learned policy on GPU.
if [ "$USE_GPU_ENV" = 1 ] || [ "$MODE" = cpu ] || [ "$MODE" = web ]; then
    echo "Balatro supports native CUDA training and --local/--fast rendering."
    exit 1
fi
mkdir -p build/balatro config
${CC:-gcc} -O3 -fPIC -I. -Iocean/balatro -c ocean/balatro/balatro_core.c -o build/balatro/balatro_core.o
ar rcs build/balatro/libbalatro_core.a build/balatro/balatro_core.o
INCLUDES+=(-Iocean/balatro)
LINK_ARCHIVES+=("$(pwd)/build/balatro/libbalatro_core.a")
