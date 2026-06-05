CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I.

TARGET = tests
LIB_TARGET = libbattle.so

all: $(TARGET) $(LIB_TARGET)

$(TARGET): tests.cpp types.h move.h pokemon.h battle.h
	$(CXX) $(CXXFLAGS) tests.cpp -o $(TARGET)

$(LIB_TARGET): bindings.cpp types.h move.h pokemon.h battle.h move_db.h
	$(CXX) $(CXXFLAGS) -shared -fPIC bindings.cpp -o $(LIB_TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) $(LIB_TARGET)

.PHONY: all run clean
