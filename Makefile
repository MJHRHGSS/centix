ifeq ($(OS),Windows_NT)
    EXT := .exe
else
    EXT :=
endif
CC = gcc
CFLAGS = -O3 -Ofast -march=native -mtune=native -funroll-loops \
         -fno-math-errno -ffast-math -fopenmp -flto \
         -fno-asynchronous-unwind-tables -fno-stack-protector \
         -fno-exceptions -frename-registers -funsafe-math-optimizations \
         -fPIC -std=c11 -Iinclude
LDFLAGS = -lm -fopenmp
SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,out/%.o,$(SRC))
TARGET = bin/cx$(EXT)

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

out/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf out/* bin/*
