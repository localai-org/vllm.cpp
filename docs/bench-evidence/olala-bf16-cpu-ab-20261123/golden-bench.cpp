// Scratch oracle-side harness: greedy token ids from the llama.cpp olala fork.
// Not part of vllm.cpp; produces golden ids for the W2d token gate.
#include "llama.h"
#include <cstdio>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

static double now() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt = nullptr;
    int n_gen = 8;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m")) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p")) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n")) n_gen = atoi(argv[++i]);
    }
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);

    // tokenize with special-token parsing, no BOS (the reference caches nothing at t<0)
    int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, false, true);
    std::vector<llama_token> toks(n_prompt);
    llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), (int)toks.size(), false, true);
    fprintf(stderr, "prompt tokens (%zu):", toks.size());
    for (auto t : toks) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    std::vector<llama_token> out = toks;
    llama_batch batch = llama_batch_init(512, 0, 1);
    int n_pos = 0;
    double prompt_ms = 0, gen_ms = 0;
    for (int g = -1; g < n_gen; ++g) {
        batch.n_tokens = 0;
        llama_token last = out.back();
        int start = g < 0 ? 0 : n_pos;  // first pass: whole prompt
        for (int i = start; i < (int)out.size(); ++i) {
            batch.token[batch.n_tokens] = out[i];
            batch.pos[batch.n_tokens] = i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (i == (int)out.size() - 1);
            ++batch.n_tokens;
        }
        n_pos = (int)out.size();
        double t0 = now();
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
        double dt = now() - t0;
        if (g < 0) prompt_ms = dt; else gen_ms += dt;
        if (g + 1 >= n_gen) break;
        const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        int best = 0;
        for (int t = 1; t < llama_vocab_n_tokens(vocab); ++t)
            if (logits[t] > logits[best]) best = t;
        out.push_back(best);
        char buf[128];
        int n = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        fprintf(stderr, "gen[%d] = %d '%.*s'\n", g + 1, best, n > 0 ? n : 0, buf);
    }
    fprintf(stderr, "TIMING prompt_ms=%.3f gen_ms=%.3f n_gen=%d\n", prompt_ms, gen_ms, n_gen);
    printf("golden_ids = [");
    for (size_t i = 0; i < out.size(); ++i)
        printf("%s%lld", i ? ", " : "", (long long)out[i]);
    printf("]\n");
    return 0;
}
