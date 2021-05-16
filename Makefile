SRCDIR := src
TESTDIR := test
OBJDIR := .objs
DEPDIR := .deps

BIN := photobooth

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
DEPS := $(SRCS:$(SRCDIR)/%.cpp=$(DEPDIR)/%.d)

LIB_OBJS := $(filter-out $(OBJDIR)/$(BIN).o, $(OBJS))

TEST_SRCS := $(wildcard $(TESTDIR)/*.cpp)
TEST_OBJS := $(TEST_SRCS:$(TESTDIR)/%.cpp=$(OBJDIR)/%.o)
TEST_BIN := $(TEST_SRCS:$(TESTDIR)/%.cpp=%)
TEST_DEPS := $(TEST_SRCS:$(TESTDIR)/%.cpp=$(DEPDIR)/%.d)

$(shell mkdir -p $(dir $(OBJS)) >/dev/null)
$(shell mkdir -p $(dir $(DEPS)) >/dev/null)

CXX := g++
LD := g++

CXXFLAGS := \
  -O3 \
  -std=c++17 \
  -I/usr/include/opencv4 \
  -I/home/ginkage/tensorflow \
  -I/home/ginkage/tensorflow/tensorflow/lite/tools/make/downloads/eigen \
  -I/home/ginkage/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
  -march=armv8-a \
  -funsafe-math-optimizations \
  -ftree-vectorize \
  -fPIC

LDFLAGS := \
  -L/home/ginkage/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib \
  -Wl,--no-export-dynamic \
  -Wl,--exclude-libs,ALL \
  -Wl,--gc-sections \
  -Wl,--as-needed

LDLIBS := \
  -lrt \
  -lpthread \
  -lopencv_imgcodecs \
  -lopencv_core \
  -lopencv_highgui \
  -lopencv_imgproc \
  -lopencv_videoio \
  -ltensorflow-lite \
  -ledgetpu \
  -ldl

DEPFLAGS = -MT $@ -MD -MP -MF $(DEPDIR)/$*.d

$(BIN): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(TEST_OBJS): $(OBJDIR)/%.o : $(TESTDIR)/%.cpp
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(TEST_BIN): % : $(OBJDIR)/%.o $(LIB_OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

-include $(DEPS) $(TEST_DEPS)

all: $(BIN)

test: $(TEST_BIN)

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(DEPDIR) $(BIN) $(TEST_BIN)
