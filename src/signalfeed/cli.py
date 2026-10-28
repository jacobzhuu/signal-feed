import asyncio
import typer
from . import settings  # noqa: F401 - initialize cache paths before model imports

app = typer.Typer(no_args_is_help=True)


@app.command()
def prepare():
    from .data import prepare as run

    run()


@app.command()
def encode():
    from .data import encode as run

    run()


@app.command()
def features():
    from .generator import prepare_codes
    from .features import build_statistics

    prepare_codes()
    build_statistics()


@app.command()
def train(model: str = "sid", seed: int = 42, epochs: int = 3, limit: int = 0):
    if model == "sid":
        from .generator import train as run
    elif model == "ranker":
        from .ranker import train as run
    else:
        raise typer.BadParameter("model must be sid or ranker")
    run(seed, epochs, limit)


@app.command()
def evaluate(
    seed: int = 42,
    ranking_limit: int = 0,
    retrieval_limit: int = 1000,
    split: str = "test",
):
    from .evaluation import evaluate as run

    run(seed, ranking_limit, retrieval_limit, split)


@app.command()
def sync(limit: int = 1000):
    from .ingest import sync as run

    asyncio.run(run(limit))


@app.command()
def upstream_smoke(steps: int = 32):
    from .upstream_smoke import run

    run(steps)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    uvicorn.run("signalfeed.api:app", host=host, port=port)


@app.command()
def tune_fusion(seed: int = 42):
    from .evaluation import tune_fusion as run

    run(seed)


@app.command()
def retrain_online():
    from .online_learning import retrain

    retrain()


@app.command()
def infer(history: str = "", seed: int = 42, limit: int = 10):
    """Infer from comma-separated MIND news IDs using saved checkpoints."""
    import json
    from .inference import recommend

    if not 1 <= limit <= 100:
        raise typer.BadParameter("limit must be between 1 and 100")
    result = recommend(
        [i.strip() for i in history.split(",") if i.strip()], seed, limit
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
