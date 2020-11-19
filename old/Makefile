# Simple Makefile for Edges project

CXX = g++
CXXFLAGS = -Wall -g -std=c++11

LIBS = /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4

TARGET = main.out

SRC_DIRS ?= ./src
SRCS := $(shell find $(SRC_DIRS) -name *.cpp)

MKDIRS = mkdir -p out

main: mkdirs
	$(CXX) $(CXXFLAGS) $(LIBS) -o $(TARGET) $(SRCS)

mkdirs:
	$(MKDIRS)

.PHONY: clean all
clean:
	\rm -f main.out
	\rm -f -r out