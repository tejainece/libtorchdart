macos_fix_torchlib:
	sudo xattr -r -d com.apple.quarantine libtorch/lib/

download_libtorch_macos:
	# TODO

download_libtorch_linux:
	wget https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.9.1%2Bcu130.zip
	unzip libtorch-shared-with-deps-2.9.1+cu130.zip
	rm libtorch-shared-with-deps-2.9.1+cu130.zip

download_libtorch_windows:
	# TODO

prepare_ubuntu:
	sudo apt-get -y install cuda-toolkit-13-0

prepare_macos:
	brew install libomp

populate_compile_commands:
	(cd build && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

build: torch_ffi.cpp generator.cpp include/torch_ffi.h
	(cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && cmake --build .)