#!/usr/bin/env bash


if [ $# -eq 0 ]; then
        folder="build/debug"
        flag="-DCMAKE_BUILD_TYPE=Debug"
elif [ $# -eq 1 ] && [ $1 == "release" ]; then
        folder="build/release"
        flag="-DCMAKE_BUILD_TYPE=Release"
else
        echo "unvalid arguments"
        exit 1
fi

# cmake . -B$folder $flag -DCMAKE_C_COMPILER=/usr/bin/gcc-6
cmake . -B$folder $flag
if [ $? -ne 0 ]; then
        echo "cmake failed."
        exit 1
fi
cd $folder
make
if [ $? -ne 0 ]; then
        echo "make failed."
        exit 1
fi
