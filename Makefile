NCC = nvcc
CXX = g++
CC = gcc
FLAGS = -O3
NVFLAGS = -use_fast_math

ifeq (1, $(DEBUG))
FLAGS := $(FLAGS) -g
endif

.PHONY: all
all: bin/main

bin/main: obj/main.o obj/lodepng.o
	@echo Linking...
	@mkdir -p bin
	@$(NCC) -o $@ $^ $(FLAGS) $(NVFLAGS)

obj/main.o: src/main.cu src/lodepng.h src/cutil_math.h src/geometry.h src/read_scene.h
	@echo Compiling $@
	@mkdir -p obj
	@$(NCC) -o $@ -c $< $(FLAGS) $(NVFLAGS)

obj/lodepng.o: src/lodepng.cpp src/lodepng.h
	@echo Compiling $@
	@mkdir -p obj
	@$(CXX) -o $@ -c $< $(FLAGS)

.PHONY:clean
clean:
	@rm -f *.png
	@rm -f -r obj/
	@rm -f -r bin/
