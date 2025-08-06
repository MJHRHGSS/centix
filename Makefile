CC=gcc
CFLAGS=-Wall -Wextra -Iinclude
SRC=src/main.c
OBJ=$(SRC:.c=.o)
TARGET=out/cx
PHONY:all
all:$(TARGET)
$(TARGET):$(OBJ)
	$(CC) $(OBJ) -o $@
%.o:%.c 
	$(CC) $(CFLAGS) -c $< -o $@
