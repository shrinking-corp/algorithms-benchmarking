## FAQ - ASWS Benchmarking Tool

### 1. The application fails with a missing dataset error.
Make sure you have cloned the dataset repository into `data/dataset` as described in the README. The dataset is not included by default.

### 2. I get an authentication or API key error.
Check that you have created a `.env` file with your OpenAI API key (or the correct key for your LLM provider). The variable name should match `api_key_env` in your config.

### 3. The tool times out or is very slow.
- The default timeout for LLM calls is high (600 seconds) to handle slow responses. If you experience frequent timeouts, check your network connection or try reducing `max_concurrent_llm` in the config.
- Some LLM providers may have rate limits. The tool will retry on rate limit errors, but repeated failures may require waiting or reducing concurrency.

### 4. Results are not saved or are missing.
Check that the `results_path` directory exists and is writable. Also, ensure you have permission to write to the cache directory.

### 5. How do I add a new LLM or algorithm?
See the "Extending" section in the main README. Implement your client/adapter/optimizer and register it in the appropriate factory.

### 6. I get a crash or traceback during benchmarking.
- Check the logs in the `logs/` directory for detailed error messages.
- Most errors are fail-fast: the run will stop on the first critical error to avoid corrupting results.
- If you fix the issue, you can resume from the last checkpoint (see `results/checkpoints/`).

### 7. How do I change the optimization or evaluation parameters?
Edit `config.yaml` and adjust the `benchmark` section (e.g., `n_iterations`, `population_size`, etc.).

### 8. Can I use a different language or metric?
- Supported languages for CodeBLEU: python, java, javascript, cpp (set in `benchmark.language`).
- To add a new metric, extend `src/metrics.py` and `src/fitness.py`.

### 9. Where can I find example configurations?
See `config.yaml.example` in the repository for a template with comments.

### 10. Who do I contact for help?
Open an issue on the GitHub repository or contact the maintainers listed in the project documentation.
