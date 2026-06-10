CXX ?= c++
BUILD_DIR ?= build
CXXFLAGS ?= -O2 -std=c++17 -Iinclude -Wall -Wextra

.PHONY: all test clean

all: $(BUILD_DIR)/compile_model $(BUILD_DIR)/infer

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/compile_model: tools/compile_model.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tools/compile_model.cpp -o $(BUILD_DIR)/compile_model

$(BUILD_DIR)/infer: src/infer_main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/infer_main.cpp -o $(BUILD_DIR)/infer

test: | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_ops.cpp -o $(BUILD_DIR)/test_ops && ./$(BUILD_DIR)/test_ops
	$(CXX) $(CXXFLAGS) tests/test_arena_planner.cpp -o $(BUILD_DIR)/test_arena_planner && ./$(BUILD_DIR)/test_arena_planner
	$(CXX) $(CXXFLAGS) tests/test_end_to_end.cpp -o $(BUILD_DIR)/test_end_to_end && ./$(BUILD_DIR)/test_end_to_end
	$(CXX) $(CXXFLAGS) tests/test_zero_alloc.cpp -o $(BUILD_DIR)/test_zero_alloc && ./$(BUILD_DIR)/test_zero_alloc

clean:
	rm -rf $(BUILD_DIR)
