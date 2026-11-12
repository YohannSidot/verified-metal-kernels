# Optional: llama.cpp Metal as a reference

`bench.py` runs against MLX and PyTorch MPS by default. In the published numbers, llama.cpp Metal was also a
reference, and the strongest one at two shapes (prefix N=32 P=4096; dsa B=4 L=16384). To include it, build
llama.cpp's `test-backend-ops` with the patch in this folder, which adds the benchmark cases of this package
(selected by the environment variables `KMAC_PERF` and `KMAC_CASE`) and prints a per-iteration median.

Requires Xcode command-line tools, CMake and Ninja.

```bash
git clone https://github.com/ggml-org/llama.cpp llama.cpp
git -C llama.cpp checkout 0a94a3f289687883617b9a2abbc7e0ade2d6b6c4
git -C llama.cpp apply ../llama_cpp/test-backend-ops.patch      # path relative to the llama.cpp folder
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build llama.cpp/build --target test-backend-ops

python bench.py --llama llama.cpp/build/bin/test-backend-ops
```

How it is timed: llama.cpp runs in its own process and its own harness, warm, repeating the op on the same tensors
(median of at least 30 iterations at 30 rounds, 10 below). It is measured in 3 slots that alternate with the
in-process rounds. For DSA, its time is the sum of its 4 ggml ops (indexer, top-k, mask, sparse flash attention).
The in-process side runs a chain of 16 calls on distinct buffers after a cache flush, so the comparison is not identical in
method; small shapes depend on this.

The patch modifies `tests/test-backend-ops.cpp` of llama.cpp (MIT License); the unchanged context lines it quotes
remain under the llama.cpp license.
