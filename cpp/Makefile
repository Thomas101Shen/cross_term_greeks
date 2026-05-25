CXX      ?= c++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -Werror -O2 -g
CPPFLAGS ?= -Iinclude

BUILD_DIR := build
SRC_DIR   := src
TEST_DIR  := tests

SRCS := $(shell find $(SRC_DIR) -name '*.cpp' | sort)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/obj/%.o,$(SRCS))
TEST_SRCS := $(shell find $(TEST_DIR) -name 'test_*.cpp' | sort)
TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/%,$(TEST_SRCS))

.PHONY: all test clean format format-check
.SECONDARY: $(OBJS)

all: $(TEST_BINS)

test: $(TEST_BINS)
	@for test_bin in $(TEST_BINS); do \
		echo "Running $$test_bin"; \
		$$test_bin; \
	done

$(BUILD_DIR)/obj/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cpp $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(OBJS) -o $@

clean:
	rm -rf $(BUILD_DIR)

format:
	@echo "Formatting code..."
	@find include src tests \( -name '*.hpp' -o -name '*.cpp' \) 2>/dev/null | xargs clang-format -i
	@echo "Code formatted"

format-check:
	@echo "Checking code formatting..."
	@find include src tests \( -name '*.hpp' -o -name '*.cpp' \) 2>/dev/null | xargs clang-format --dry-run --Werror
	@echo "All files are properly formatted"
