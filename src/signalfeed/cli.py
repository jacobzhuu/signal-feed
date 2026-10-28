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
    import os

    os.environ.setdefault("SIGNALFEED_PREWARM", "1")

    uvicorn.run("signalfeed.api:app", host=host, port=port)


@app.command()
def tune_fusion(seed: int = 42):
    from .evaluation import tune_fusion as run

    run(seed)


@app.command()
def retrain_online(cold: bool = False):
    from .online_learning import retrain

    retrain(cold=cold)


@app.command()
def train_cold(seed: int = 42, limit: int = 30000, epochs: int = 3):
    from .cold_training import train

    train(seed, limit, epochs)


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


@app.command()
def bind_offline(attestation: str):
    """Explicitly attest already audited legacy checkpoints; never done on load."""
    from .provenance import bind_model
    from .settings import path

    if len(attestation.strip()) < 12:
        raise typer.BadParameter(
            "Provide the audit/run reference supporting this attestation"
        )
    for file in path("models").glob("*/model.pt"):
        if file.parent.name.startswith(("sid-", "ranker-")):
            bind_model(file.parent, origin="audited legacy: " + attestation)


@app.command()
def activate_online(version: str = "", cold: bool = False):
    """Validated rollback/publication, serialized with training. Empty restores content."""
    import json
    from .settings import path
    from .publication import publication_lock, atomic_json
    from .online_schema import CONTRACT
    from .provenance import sha256
    import re
    import torch
    from .ranker import Ranker

    if cold:
        from .coldfuse import CONTRACT

    root = path("cold-online-models" if cold else "online-models")
    with publication_lock(root):
        pointer = root / "active.json"
        if not version:
            pointer.unlink(missing_ok=True)
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
            raise typer.BadParameter("Invalid version")
        manifest = json.loads((root / version / "manifest.json").read_text())
        if (
            any(manifest.get(k) != v for k, v in CONTRACT.items())
            or manifest["version"] != version
        ):
            raise typer.BadParameter("Incompatible model contract")
        file = root / version / "model.pt"
        if sha256(file) != manifest["model_sha256"]:
            raise typer.BadParameter("Model checksum mismatch")
        candidate = Ranker()
        candidate.load_state_dict(
            torch.load(file, map_location="cpu", weights_only=True)
        )
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            raise typer.BadParameter("Non-finite model")
        atomic_json(pointer, manifest)


if __name__ == "__main__":
    app()
