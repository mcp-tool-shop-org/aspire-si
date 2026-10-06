"""
ASPIRE CLI - command line interface for training.
"""

import json
import os
import shutil
import sys
from multiprocessing import freeze_support
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aspire import __version__

app = typer.Typer(
    help="ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"aspire [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """ASPIRE: Teaching AI to develop judgment, not just knowledge."""
    pass


@app.command()
def train(
    config: Path = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    prompts_file: Path = typer.Option(None, "--prompts", "-p", help="Path to prompts JSON"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory"),
    teacher: str = typer.Option("claude", "--teacher", "-t", help="Teacher model to use"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
):
    """Train a model using ASPIRE."""
    freeze_support()

    from aspire.config import AspireConfig
    from aspire.trainer import AspireTrainer

    # Load or create config
    if config and config.exists():
        cfg = AspireConfig.from_yaml(config)
    else:
        cfg = AspireConfig()

    # Override with CLI args
    cfg.training.output_dir = output_dir
    cfg.training.num_epochs = epochs
    cfg.teacher.default_teacher = teacher

    # Load prompts
    if prompts_file and prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
    else:
        # Demo prompts
        prompts = [
            "Explain recursion in programming.",
            "What is the difference between a list and a tuple in Python?",
            "How does HTTP work?",
        ]
        console.print("[yellow]No prompts file provided, using demo prompts[/yellow]")

    # Train
    trainer = AspireTrainer(cfg)
    trainer.train(prompts)

    console.print("[bold green]Training complete![/bold green]")


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint directory"),
    prompts_file: Path = typer.Option(..., "--prompts", "-p", help="Path to prompts JSON"),
    output: Path = typer.Option(None, "--output", "-o", help="Output results file"),
):
    """Evaluate a trained model."""
    freeze_support()

    import asyncio

    from aspire.config import AspireConfig
    from aspire.trainer import AspireTrainer

    # Load config from checkpoint
    cfg = AspireConfig.from_yaml(checkpoint / "config.yaml")

    # Load prompts
    with open(prompts_file) as f:
        prompts = json.load(f)

    # Create trainer and load checkpoint
    trainer = AspireTrainer(cfg)
    trainer.load_checkpoint(checkpoint)

    # Evaluate
    metrics = asyncio.run(trainer._evaluate(prompts))

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}")

    console.print(table)

    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)


@app.command()
def dialogue(
    prompt: str = typer.Argument(..., help="Prompt to generate dialogue for"),
    teacher: str = typer.Option("socratic", "--teacher", "-t", help="Teacher persona"),
    turns: int = typer.Option(3, "--turns", "-n", help="Number of dialogue turns"),
    model: str = typer.Option("microsoft/Phi-3-mini-4k-instruct", "--model", "-m", help="Student model"),
):
    """Generate a single adversarial dialogue."""
    freeze_support()

    import asyncio

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aspire.dialogue import DialogueGenerator
    from aspire.teachers import get_teacher

    console.print(f"[bold]Generating dialogue with {teacher} teacher[/bold]\n")

    # Load student model
    console.print("Loading student model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    student = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto",
    )

    # Create teacher
    teacher_model = get_teacher(teacher)

    # Generate dialogue
    generator = DialogueGenerator(
        student_model=student,
        student_tokenizer=tokenizer,
        teacher=teacher_model,
        max_turns=turns,
    )

    dialogue = asyncio.run(generator.generate_dialogue(prompt))

    # Display
    console.print(f"\n[bold cyan]Prompt:[/bold cyan] {prompt}")
    console.print(f"\n[bold green]Initial Response:[/bold green]\n{dialogue.initial_response}")

    for turn in dialogue.history.turns:
        console.print(f"\n[bold yellow]Challenge ({turn.challenge.challenge_type.value}):[/bold yellow]")
        console.print(turn.challenge.content)
        console.print("\n[bold green]Response:[/bold green]")
        console.print(turn.student_response)

    score = dialogue.final_evaluation.overall_score
    console.print(f"\n[bold magenta]Final Score:[/bold magenta] {score:.1f}/10")
    console.print(f"\n[bold]Reasoning:[/bold]\n{dialogue.final_evaluation.reasoning}")

    if dialogue.final_evaluation.improved_response:
        console.print("\n[bold blue]Improved Response:[/bold blue]")
        console.print(dialogue.final_evaluation.improved_response)


@app.command()
def teachers():
    """List available teacher personas."""
    from aspire.teachers.registry import TeacherRegistry

    table = Table(title="Available Teachers")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    teacher_info = {
        "claude": "Default Claude-based teacher",
        "openai": "GPT-4 based teacher",
        "socratic": "Teaches through questions, never gives answers directly",
        "scientific": "Demands evidence, rigor, and falsifiable claims",
        "creative": "Encourages novel thinking and unconventional approaches",
        "adversarial": "Stress-tests reasoning through devil's advocacy",
        "compassionate": "Balances challenge with encouragement",
    }

    for name in TeacherRegistry.list():
        desc = teacher_info.get(name, "Custom teacher")
        table.add_row(name, desc)

    console.print(table)


@app.command()
def init(
    output: Path = typer.Option(Path("aspire-config.yaml"), "--output", "-o", help="Output path"),
):
    """Create a default configuration file."""
    from aspire.config import AspireConfig

    cfg = AspireConfig()
    cfg.to_yaml(output)
    console.print(f"[green]Created config file: {output}[/green]")


@app.command()
def doctor():
    """Check your environment for ASPIRE compatibility."""
    console.print(
        Panel.fit(f"[bold]ASPIRE Environment Check[/bold]\nVersion {__version__}", border_style="blue")
    )
    console.print()

    all_good = True

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):  # noqa: UP036
        console.print(f"[green]OK[/green]  Python {py_version} (>= 3.10 required)")
    else:
        console.print(f"[red]ERROR[/red]  Python {py_version} (>= 3.10 required)")
        all_good = False

    # PyTorch and CUDA
    try:
        import torch

        console.print(f"[green]OK[/green]  PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"[green]OK[/green]  CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")

            if gpu_mem < 8:
                console.print("[yellow]WARN[/yellow]  GPU has < 8GB VRAM - may need 4-bit quantization")
        else:
            console.print("[yellow]WARN[/yellow]  CUDA not available - training will be slow on CPU")
    except ImportError:
        console.print("[red]ERROR[/red]  PyTorch not installed")
        all_good = False

    # Transformers
    try:
        import transformers

        console.print(f"[green]OK[/green]  Transformers {transformers.__version__}")
    except ImportError:
        console.print("[red]ERROR[/red]  Transformers not installed")
        all_good = False

    # API Keys
    console.print()
    console.print("[bold]API Keys:[/bold]")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        masked = anthropic_key[:8] + "..." + anthropic_key[-4:] if len(anthropic_key) > 12 else "***"
        console.print(f"[green]OK[/green]  ANTHROPIC_API_KEY set ({masked})")
    else:
        console.print("[yellow]WARN[/yellow]  ANTHROPIC_API_KEY not set")
        console.print("         Set with: [cyan]export ANTHROPIC_API_KEY=your-key[/cyan]")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        masked = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        console.print(f"[green]OK[/green]  OPENAI_API_KEY set ({masked})")
    else:
        console.print("[dim]--[/dim]    OPENAI_API_KEY not set (optional)")

    # Disk space
    console.print()
    console.print("[bold]Storage:[/bold]")

    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        # Get size
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        size_gb = total_size / (1024**3)
        console.print(f"[dim]--[/dim]    HuggingFace cache: {size_gb:.1f} GB at {cache_dir}")
    else:
        console.print("[dim]--[/dim]    HuggingFace cache: not yet created")

    # Check free space
    try:
        total, used, free = shutil.disk_usage(Path.home())
        free_gb = free / (1024**3)
        if free_gb > 20:
            console.print(f"[green]OK[/green]  Free disk space: {free_gb:.1f} GB")
        elif free_gb > 10:
            console.print(f"[yellow]WARN[/yellow]  Free disk space: {free_gb:.1f} GB (models can be large)")
        else:
            console.print(f"[red]ERROR[/red]  Free disk space: {free_gb:.1f} GB (may be insufficient)")
            all_good = False
    except Exception:
        pass

    # Summary
    console.print()
    if all_good:
        console.print(
            Panel.fit(
                "[bold green]All checks passed![/bold green]\nYou're ready to use ASPIRE.",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold red]Some issues found.[/bold red]\nPlease address the errors above.",
                border_style="red",
            )
        )

    raise typer.Exit(0 if all_good else 1)


@app.command()
def diagnose(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check environment: Python, ML dependencies, GPU, config."""
    import importlib

    checks: list[dict[str, str | None]] = []
    all_ok = True

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(
        {
            "check": "python_version",
            "status": "ok" if py_ok else "fail",
            "value": py_ver,
            "hint": None if py_ok else "ASPIRE requires Python 3.10+",
        }
    )
    if not py_ok:
        all_ok = False

    # Core ML dependencies
    core_deps = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("pydantic", "pydantic"),
        ("typer", "typer"),
        ("rich", "rich"),
    ]
    for name, module_name in core_deps:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            checks.append({"check": f"dep.{name}", "status": "ok", "value": ver, "hint": None})
        except ImportError:
            checks.append(
                {
                    "check": f"dep.{name}",
                    "status": "fail",
                    "value": "missing",
                    "hint": f"pip install {name}",
                }
            )
            all_ok = False

    # GPU check
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            checks.append(
                {
                    "check": "gpu",
                    "status": "ok",
                    "value": f"{gpu_name} ({vram:.1f} GB)",
                    "hint": None,
                }
            )
        else:
            checks.append(
                {
                    "check": "gpu",
                    "status": "info",
                    "value": "no CUDA GPU detected",
                    "hint": "Training will use CPU (slow). Install CUDA toolkit for GPU support.",
                }
            )
    except ImportError:
        checks.append(
            {
                "check": "gpu",
                "status": "info",
                "value": "torch not installed",
                "hint": "Install torch for GPU detection",
            }
        )

    # Package version
    checks.append({"check": "aspire_version", "status": "ok", "value": __version__, "hint": None})

    if json_output:
        console.print_json(data={"ok": all_ok, "checks": checks})
    else:
        table = Table(title=f"ASPIRE v{__version__} - environment diagnostics")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Value")
        table.add_column("Hint", style="dim")
        for c in checks:
            status_str = (
                "[green]OK[/green]"
                if c["status"] == "ok"
                else ("[yellow]INFO[/yellow]" if c["status"] == "info" else "[red]FAIL[/red]")
            )
            table.add_row(c["check"], status_str, c.get("value", ""), c.get("hint", "") or "")
        console.print(table)
        console.print()
        if all_ok:
            console.print("[bold green]All checks passed.[/bold green]")
        else:
            console.print("[bold red]Some checks failed. See hints above.[/bold red]")

    raise typer.Exit(0 if all_ok else 1)


if __name__ == "__main__":
    app()
