CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I.

TARGET = tests

all: $(TARGET)

$(TARGET): tests.cpp types.h move.h pokemon.h battle.h
	$(CXX) $(CXXFLAGS) tests.cpp -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
