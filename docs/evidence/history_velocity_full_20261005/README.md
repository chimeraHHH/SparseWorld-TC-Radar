# Saved numeric evidence

The 32 NPZ files preserve the original numeric bytes. MANIFEST.json records SHA256 and byte sizes. No model is executed by the CPU reconstruction.

`summary.json` contains final/best paired scene bootstrap (2000 samples, seed20260927), all four horizons and 17 class IoUs, binary IoUs, and H2 interventions. H2 best10 equals final10; H8 best6 is separate.

Compare using `tools/compare_forecast_results.py`'s `load_confusions` and `compare` with the stated seed. Pair scene names, sample indices, and ground-truth row sums before computing. Token pairing was independently verified using the original hook reports; intervention JSONs did not emit tokens and were bound through their identical indices to the hook's full/subset token lists.

Weight audit basis: complete server CPU load audit and SHA; no independent full weight download/reload on the local machine. Formal cost measurements are pending; preparation and failed cached paths are excluded.

Original hook full/subset JSONs are included without rewriting bytes. They expose the actual indices/tokens used for full pairing and fixed-subset membership. Interventions retain original numeric matrices; their tokens were not originally emitted.
