import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import time
from typing import List, Optional
import aiohttp
import numpy
import logging
import argparse
import json
import random
from openai import APIConnectionError, AsyncOpenAI
from aiohttp import ClientSession
from httpx_aiohttp import AiohttpTransport
from openai import DefaultAsyncHttpxClient
from openai.types.chat import (
    ChatCompletionStreamOptionsParam,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


SAMPLE_PROMPTS = [
    "Explain how blockchain technology works, and provide a real-world example of its application outside of cryptocurrency.",
    "Compare and contrast the philosophies of Nietzsche and Kant, including their views on morality and human nature.",
    "Imagine you're a travel blogger. Write a detailed post describing a week-long adventure through rural Japan.",
    "Write a fictional letter from Albert Einstein to a modern-day physicist, discussing the current state of quantum mechanics.",
    "Provide a comprehensive explanation of how transformers work in machine learning, including attention mechanisms and positional encoding.",
    "Draft a business proposal for launching a new AI-powered productivity app, including target audience, key features, and a monetization strategy.",
    "Simulate a panel discussion between Elon Musk, Marie Curie, and Sun Tzu on the topic of 'Leadership in Times of Crisis'.",
    "Describe the process of photosynthesis in depth, and explain its importance in the global carbon cycle.",
    "Analyze the impact of social media on political polarization, citing relevant studies or historical examples.",
    "Write a short science fiction story where humans discover a parallel universe that operates under different physical laws.",
    "Explain the role of the Federal Reserve in the U.S. economy and how it manages inflation and unemployment.",
    "Describe the architecture of a modern web application, from frontend to backend, including databases, APIs, and deployment.",
    "Write an essay discussing whether artificial general intelligence (AGI) poses an existential threat to humanity.",
    "Summarize the key events and consequences of the Cuban Missile Crisis, and reflect on lessons for modern diplomacy.",
    "Create a guide for beginners on how to train a custom LLM using open-source tools and publicly available datasets.",
]


@dataclass
class PercentileResults:
    average: float
    p50: float
    p95: float
    p99: float


@dataclass
class BenchmarkResults:
    model: str
    total_requests: int
    successful_requests: int
    success_rate: float
    concurrency: int
    request_timeout: int
    max_completion_tokens: int
    total_time: float
    requests_per_second: float
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens_per_second: float
    total_prompt_tokens_per_second: float
    total_completion_tokens_per_second: float
    latency: PercentileResults
    completion_tokens_per_second: PercentileResults
    time_to_first_token: PercentileResults


async def process_stream(stream):
    first_token_time = None
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.usage:
            return first_token_time, chunk.usage
    return first_token_time, None


def get_random_prompt(prompt_multiplier):
    """
    Returns a random prompt from the SAMPLE_PROMPTS list, repeated prompt_multiplier times.
    """
    # Add a random prefix to avoid prefix cache hits
    random_prefix = str(random.randint(100000, 999999))
    return (
        random_prefix + " " + (random.choice(SAMPLE_PROMPTS) + " ") * prompt_multiplier
    )


async def make_chat_completion_request(
    client: AsyncOpenAI,
    model,
    max_completion_tokens,
    ignore_eos,
    request_timeout,
    prompt_multiplier,
):
    start_time = time.time()
    content = get_random_prompt(prompt_multiplier)
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_completion_tokens,
            stream=True,
            stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
            extra_body={"ignore_eos": ignore_eos} if ignore_eos else None,
        )
        first_token_time, usage = await asyncio.wait_for(
            process_stream(stream), timeout=request_timeout
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = (first_token_time - start_time) * 1000 if first_token_time else None
        return usage, elapsed_time, ttft
    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except APIConnectionError as e:
        logging.error(f"API connection error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None


async def make_embedding_request(
    client: AsyncOpenAI, model, request_timeout, prompt_multiplier
):
    start_time = time.time()
    content = get_random_prompt(prompt_multiplier)
    try:
        response = await asyncio.wait_for(
            client.embeddings.create(model=model, input=content),
            timeout=request_timeout,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = None  # Embeddings do not have a time to first token in the same way as chat completions

        return response.usage, elapsed_time, ttft
    except asyncio.TimeoutError:
        logging.warning(f"Embedding request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during embedding request: {str(e)}")
        return None


async def worker(
    client,
    model,
    semaphore,
    queue,
    results,
    max_completion_tokens,
    ignore_eos,
    request_timeout,
    embeddings=False,
    prompt_multiplier=1,
    pbar=None,
):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            logging.debug(f"Starting request {task_id}")
            if embeddings:
                result = await make_embedding_request(
                    client, model, request_timeout, prompt_multiplier
                )
            else:
                result = await make_chat_completion_request(
                    client,
                    model,
                    max_completion_tokens,
                    ignore_eos,
                    request_timeout,
                    prompt_multiplier,
                )
            if result:
                results.append(result)
            else:
                logging.warning(f"Request {task_id} failed")
            queue.task_done()
            if pbar:
                pbar.update(1)
            logging.debug(f"Finished request {task_id}")


def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return numpy.percentile(values, 100 - percentile)
    return numpy.percentile(values, percentile)


async def preflight_check(client, model, embeddings=False) -> bool:
    if embeddings:
        result = await make_embedding_request(client, model, 16)
    else:
        result = await make_chat_completion_request(client, model, 16, False, 60, 1)
    return result is not None


def set_headers(aiohttp_session: ClientSession, headers: Optional[List[str]]):
    if headers:
        for header in headers:
            if ":" not in header:
                raise ValueError(f"Invalid header format: {header}. Expected Key:Value")
            key, value = header.split(":", 1)
            aiohttp_session.headers[key.strip()] = value.strip()


async def main(
    model,
    num_requests,
    concurrency,
    request_timeout,
    max_completion_tokens,
    ignore_eos,
    server_url,
    api_key,
    headers=None,
    embeddings=False,
    prompt_multiplier=1,
) -> Optional[BenchmarkResults]:
    connector = aiohttp.TCPConnector(
        limit=2000,
        force_close=True,
    )
    async with ClientSession(connector=connector) as aiohttp_session:
        if headers:
            set_headers(aiohttp_session, headers)
        transport = AiohttpTransport(client=aiohttp_session)
        httpx_client = DefaultAsyncHttpxClient(
            transport=transport, timeout=request_timeout
        )
        client = AsyncOpenAI(
            base_url=f"{server_url}/v1",
            api_key=api_key,
            http_client=httpx_client,
            max_retries=0,
        )

        if not await preflight_check(client, model, embeddings=embeddings):
            raise Exception(
                "Preflight check failed. Please check configuration and the service status."
            )

        semaphore = asyncio.Semaphore(concurrency)
        queue = asyncio.Queue()
        results = []

        # Add tasks to the queue
        for i in range(num_requests):
            await queue.put(i)

        # Add sentinel values to stop workers
        for _ in range(concurrency):
            await queue.put(None)

        pbar = tqdm(
            total=num_requests,
            desc="Running Benchmark requests",
            unit="request",
            dynamic_ncols=True,
        )

        # Create worker tasks
        workers = [
            asyncio.create_task(
                worker(
                    client,
                    model,
                    semaphore,
                    queue,
                    results,
                    max_completion_tokens,
                    ignore_eos,
                    request_timeout,
                    embeddings,
                    prompt_multiplier,
                    pbar=pbar,
                )
            )
            for _ in range(concurrency)
        ]

        start_time = time.time()

        # Wait for all tasks to complete
        await queue.join()
        await asyncio.gather(*workers)

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        return calculate_results(
            model,
            concurrency,
            request_timeout,
            max_completion_tokens,
            total_elapsed_time,
            num_requests,
            results,
        )


def calculate_results(
    model,
    concurrency,
    request_timeout,
    max_completion_tokens,
    total_elapsed_time,
    num_requests,
    results,
):
    # Calculate metrics
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    tokens_per_second_list = []
    prompt_tokens_per_second_list = []
    completion_tokens_per_second_list = []
    for usage, elapsed_time, _ in results:
        if usage is not None:
            total_tokens += usage.total_tokens
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            prompt_tokens_per_second = (
                usage.prompt_tokens / elapsed_time if elapsed_time > 0 else 0
            )
            completion_tokens_per_second = (
                usage.completion_tokens / elapsed_time if elapsed_time > 0 else 0
            )
            tokens_per_second = (
                usage.total_tokens / elapsed_time if elapsed_time > 0 else 0
            )
            tokens_per_second_list.append(tokens_per_second)
            prompt_tokens_per_second_list.append(prompt_tokens_per_second)
            completion_tokens_per_second_list.append(completion_tokens_per_second)

    latencies = [
        elapsed_time for _, elapsed_time, _ in results if elapsed_time is not None
    ]
    ttft_list = [ttft for _, _, ttft in results if ttft is not None]

    successful_requests = len(results)
    success_rate = successful_requests / num_requests if num_requests > 0 else 0
    requests_per_second = (
        successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_completion_tokens_per_second = (
        sum(completion_tokens_per_second_list) / len(completion_tokens_per_second_list)
        if completion_tokens_per_second_list
        else 0
    )
    total_tokens_per_second = (
        total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
    )
    total_prompt_tokens_per_second = (
        prompt_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
    )
    total_completion_tokens_per_second = (
        completion_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
    )
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0

    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    completion_tps_percentiles = [
        calculate_percentile(completion_tokens_per_second_list, p, reverse=True)
        for p in percentiles
    ]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]

    return BenchmarkResults(
        model=model,
        total_requests=num_requests,
        successful_requests=successful_requests,
        success_rate=success_rate,
        concurrency=concurrency,
        request_timeout=request_timeout,
        max_completion_tokens=max_completion_tokens,
        total_time=total_elapsed_time,
        requests_per_second=requests_per_second,
        total_tokens=total_tokens,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_tokens_per_second=total_tokens_per_second,
        total_prompt_tokens_per_second=total_prompt_tokens_per_second,
        total_completion_tokens_per_second=total_completion_tokens_per_second,
        latency=PercentileResults(
            average=avg_latency,
            p50=latency_percentiles[0],
            p95=latency_percentiles[1],
            p99=latency_percentiles[2],
        ),
        completion_tokens_per_second=PercentileResults(
            average=avg_completion_tokens_per_second,
            p50=completion_tps_percentiles[0],
            p95=completion_tps_percentiles[1],
            p99=completion_tps_percentiles[2],
        ),
        time_to_first_token=PercentileResults(
            average=avg_ttft,
            p50=ttft_percentiles[0],
            p95=ttft_percentiles[1],
            p99=ttft_percentiles[2],
        ),
    )


def fmt_line(label, *values, width=40):
    label_part = f"{label:<{width}}"
    value_part = " ".join(str(v) for v in values)
    return f"{label_part}{value_part}"


def fmt_float(v, suffix=""):
    return f"{v:.2f}{suffix}"


def output_benchmark_results_pretty(
    results: BenchmarkResults, file: str = None, embeddings: bool = False
):

    lines = []
    lines.append("============== Serving Benchmark Result ===============")
    lines.append(fmt_line("Model:", results.model))
    lines.append(
        fmt_line(
            "Total requests:",
            f"{results.successful_requests}/{results.total_requests}({results.success_rate:.2%})",
        )
    )
    lines.append(fmt_line("Concurrency:", results.concurrency))
    lines.append(fmt_line("Benchmark duration (s):", fmt_float(results.total_time)))
    lines.append(
        fmt_line("Request throughput (req/s):", fmt_float(results.requests_per_second))
    )
    lines.append(fmt_line("Total input tokens:", results.total_prompt_tokens))
    if not embeddings:
        lines.append(fmt_line("Total output tokens:", results.total_completion_tokens))

    output_tok_per_sec = (
        results.total_completion_tokens / results.total_time
        if results.total_time > 0
        else 0
    )
    total_tok_per_sec = (
        results.total_tokens / results.total_time if results.total_time > 0 else 0
    )
    if not embeddings:
        lines.append(
            fmt_line("Output token throughput (tok/s):", fmt_float(output_tok_per_sec))
        )
    lines.append(
        fmt_line("Total token throughput (tok/s):", fmt_float(total_tok_per_sec))
    )
    lines.append("------------------- Request Latency -------------------")
    lines.append(fmt_line("Average latency (s):", fmt_float(results.latency.average)))
    lines.append(fmt_line("P50 latency (s):", fmt_float(results.latency.p50)))
    lines.append(fmt_line("P95 latency (s):", fmt_float(results.latency.p95)))
    lines.append(fmt_line("P99 latency (s):", fmt_float(results.latency.p99)))
    if not embeddings:
        lines.append("--------------- Output Token Per Second ---------------")
        lines.append(
            fmt_line(
                "Average TPS (tok/s):",
                fmt_float(results.completion_tokens_per_second.average),
            )
        )
        lines.append(
            fmt_line(
                "P50 TPS (tok/s):", fmt_float(results.completion_tokens_per_second.p50)
            )
        )
        lines.append(
            fmt_line(
                "P95 TPS (tok/s):", fmt_float(results.completion_tokens_per_second.p95)
            )
        )
        lines.append(
            fmt_line(
                "P99 TPS (tok/s):", fmt_float(results.completion_tokens_per_second.p99)
            )
        )

        lines.append("----------------- Time to First Token -----------------")
        lines.append(
            fmt_line(
                "Average TTFT (ms):", fmt_float(results.time_to_first_token.average)
            )
        )
        lines.append(
            fmt_line("P50 TTFT (ms):", fmt_float(results.time_to_first_token.p50))
        )
        lines.append(
            fmt_line("P95 TTFT (ms):", fmt_float(results.time_to_first_token.p95))
        )
        lines.append(
            fmt_line("P99 TTFT (ms):", fmt_float(results.time_to_first_token.p99))
        )
    lines.append("=" * 55)

    output = "\n".join(lines)

    if file:
        with open(file, "w") as f:
            f.write(output + "\n")
        logging.info(f"Pretty benchmark results saved to {file}")
    else:
        print(output)


def output_benchmark_results_json(
    results: BenchmarkResults, result_file=None, embeddings: bool = False
):
    # Round all floats in results to two decimal places for output
    def _round_floats(obj, ndigits=2):
        if is_dataclass(obj):
            obj = asdict(obj)
        if isinstance(obj, dict):
            return {k: _round_floats(v, ndigits) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v, ndigits) for v in obj]
        if isinstance(obj, float):
            return round(obj, ndigits)
        return obj

    formatted_results = _round_floats(results, 2)
    if result_file:
        with open(result_file, "w") as f:
            json.dump(formatted_results, f, indent=2)
        logging.info(f"Results saved to {result_file}")
    else:
        print(json.dumps(formatted_results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Chat Completions API")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model"
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to make (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Timeout for each request in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens in the completion (default: 1024)",
    )
    parser.add_argument(
        "--prompt-multiplier",
        type=int,
        default=1,
        help="Repeat the randomly selected prompt N times to create longer inputs",
    )
    parser.add_argument(
        '--ignore-eos',
        action='store_true',
        help='Set ignore_eos flag when sending the benchmark request. This will not stop the stream when the model generates an EOS token.',
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1",
        help="URL of the GPUStack server",
    )
    parser.add_argument("--api-key", type=str, default="fake", help="GPUStack API key")
    parser.add_argument(
        "--result-file",
        type=str,
        help="Result file path to save benchmark json results",
    )
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        dest="headers",
        help="Custom HTTP header in Key:Value format. May be specified multiple times.",
    )
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Run embedding benchmark instead of chat completions',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format instead of pretty format',
    )
    args = parser.parse_args()

    try:
        results = asyncio.run(
            main(
                args.model,
                args.num_requests,
                args.concurrency,
                args.request_timeout,
                args.max_completion_tokens,
                args.ignore_eos,
                args.server_url,
                args.api_key,
                args.headers,
                args.embeddings,
                args.prompt_multiplier,
            )
        )
        if args.json:
            output_benchmark_results_json(
                results, args.result_file, embeddings=args.embeddings
            )
        else:
            output_benchmark_results_pretty(
                results, args.result_file, embeddings=args.embeddings
            )
    except Exception as e:
        logging.error(f"Benchmarking failed: {str(e)}")
        exit(1)
