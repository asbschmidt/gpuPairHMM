# cuPairHMM
cuPairHMM: Ultra-fast GPU-based PairHMM for DNA Variant Calling



Build with `make`

Run with `./align --inputfile inputfile.in` , or `./align --inputfile inputfile.in --checkResults` . `--checkResults` will additionally recompute all alignments on the cpu and compare them to the gpu results.

Run with `./align --peakBenchFloat` to benchmark all kernel configs.
