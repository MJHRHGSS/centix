ifeq ($(OS),Windows_NT)
	EXT :=.exe
else
	EXT :=
endif
CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude
SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,out/%.o,$(SRC))
TARGET = bin/cx$(EXT)
all:$(TARGET)
$(TARGET):$(OBJ)
	$(CC) $(OBJ) -o $@
out/%.o:src/%.c 
	$(CC) $(CFLAGS) -c $< -o $@
