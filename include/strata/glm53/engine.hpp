// include/strata/glm53/engine.hpp - the GLM-5.3-Flash (GGUF architecture glm5-next) engine path as a library, as
// include/strata/qwen35/engine.hpp is for Qwen3.6: `strata` hands a glm5-next model to it at the top of main() (the
// family switch), and glm_generate is a thin wrapper around the same entry point.  See src/glm53/glm_generate.cu.
#pragma once

namespace strata::glm53 {

/// The engine's command line (strata's own flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --pool-workers; the family's own: --expert-cache-gb, --stream-min,
/// --stream-threads, --pin-experts, --prompt-chunk, --ids, --n).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::glm53
