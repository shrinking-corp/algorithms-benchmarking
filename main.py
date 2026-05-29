"""
ASWS Benchmarking - entry point.
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.benchmarker import Benchmarker

load_dotenv()

def setup_logging(level: str = "INFO", run_id: str = "") -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{run_id or 'benchmark'}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    
    # Suppress verbose third-party loggers
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASWS LLM Parameter Benchmarking")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Configuration file '{config_path}' does not exist.")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(config.get("benchmark", {}).get("log_level", "INFO"), run_id=run_id)
    logger = logging.getLogger(__name__)
    logger.info("Starting ASWS benchmarking...")

    benchmarker = Benchmarker(config)
    results = asyncio.run(benchmarker.run())

    print("\n========= RESULTS =========")
    for llm_name, model_results in results.items():
        for model_name, alg_results in model_results.items():
            for alg_name, result in alg_results.items():
                print(
                    f"  LLM={llm_name:20s} | Model={model_name:20s} | Alg={alg_name:20s} | "
                    f"best_fitness={result['best_fitness']:.4f} | "
                    f"baseline={result.get('baseline_avg_codebleu', 0.0):.4f} | "
                    f"final={result.get('final_avg_codebleu', 0.0):.4f} | "
                    f"retention={result.get('retention', 0.0):.4f} | "
                    f"params={result['best_params']}"
                )
    print("===========================")


if __name__ == "__main__":
    main()
