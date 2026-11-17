// src/glm53/glm_main.cpp - glm_generate: the GLM-5.3-Flash engine path on its own (the same entry point `strata`
// switches to for a glm5-next model), as strata-q36 is for Qwen3.6.
#include "strata/glm53/engine.hpp"

int main(int argc, char** argv) { return strata::glm53::run(argc, argv); }
