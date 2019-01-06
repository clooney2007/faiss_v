#-include makefile.inc

SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

CXX          = g++
CPPFLAGS     = -DFINTEGER=int
CXXFLAGS     = -fPIC -m64 -Wno-sign-compare -g -O2 -std=c++11
CPUFLAGS     = -msse4 -mpopcnt
LDFLAGS      = -fopenmp
LIBS         = -lopenblas  

SHAREDEXT   = so
SHAREDFLAGS = -shared

ifeq ($(OS),Darwin)
	SHAREDEXT   = dylib
	SHAREDFLAGS = -dynamiclib -undefined dynamic_lookup
endif

############################
# Building

main: $(OBJ)
	$(CXX) -o $@ $^ $(LIBS)
%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

clean:
	rm -f libfaiss.*
	rm -f $(OBJ)
	rm -f main


.PHONY: all clean default demos install installdirs py test uninstall
