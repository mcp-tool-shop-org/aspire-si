"""
Example: Training an ASPIRE Code Critic.

This example shows how to:
1. Collect code from GitHub repos
2. Generate training pairs with teacher critiques
3. Train a critic to predict code quality
"""

from multiprocessing import freeze_support


def main():
    from aspire.integrations.code import (
        CodeTeacher,
        CodeCritic,
        AspireCodeTrainer,
        CodeAspireConfig,
    )
    from aspire.integrations.code.config import Language
    from aspire.integrations.code.data import (
        GitHubRepoCollector,
        generate_training_pairs,
        create_balanced_dataset,
    )

    print("=" * 60)
    print("ASPIRE Code Critic Training")
    print("=" * 60)
    print()

    # Configuration
    config = CodeAspireConfig()
    config.training.epochs = 5
    config.training.batch_size = 4
    config.critic.pretrained_model = "microsoft/codebert-base"

    # Create teacher for generating critiques
    teacher = CodeTeacher(
        personas=[
            "correctness_checker",
            "style_guide",
            "security_auditor",
        ],
        strategy="vote",
        use_llm=False,  # Use static analysis only (no API needed)
    )

    # Option 1: Collect code from GitHub
    print("Collecting code samples...")

    try:
        collector = GitHubRepoCollector()

        # Collect from quality repos
        samples = []
        for repo, filename, code in collector.collect_from_quality_repos(
            Language.PYTHON,
            files_per_repo=10,
        ):
            samples.append((repo, filename, code))
            if len(samples) >= 50:  # Limit for demo
                break

        print(f"Collected {len(samples)} code samples")

    except Exception as e:
        print(f"Could not collect from GitHub: {e}")
        print("Using synthetic examples instead...")

        # Option 2: Use synthetic examples
        samples = [
            ("example", "good.py", '''
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''),
            ("example", "bad.py", '''
def f(x):
    y = eval(x)
    return y
'''),
            ("example", "medium.py", '''
def process_items(items):
    result = []
    for item in items:
        if item is not None:
            result.append(item * 2)
    return result
'''),
        ] * 10  # Repeat for more data

    # Generate training pairs
    print("Generating training pairs with teacher critiques...")
    pairs = generate_training_pairs(
        teacher=teacher,
        code_samples=samples,
        language=Language.PYTHON,
    )

    print(f"Generated {len(pairs)} training pairs")

    # Show score distribution
    scores = [p.critique.overall_score for p in pairs]
    print(f"Score distribution: min={min(scores):.1f}, max={max(scores):.1f}, mean={sum(scores)/len(scores):.1f}")

    # Balance the dataset
    balanced_pairs = create_balanced_dataset(pairs, score_bins=5)
    print(f"Balanced dataset: {len(balanced_pairs)} pairs")

    # Split into train/val
    split_idx = int(len(balanced_pairs) * 0.8)
    train_pairs = balanced_pairs[:split_idx]
    val_pairs = balanced_pairs[split_idx:]

    # Create trainer
    print("\nInitializing trainer...")
    trainer = AspireCodeTrainer(
        config=config,
        teacher=teacher,
        student_model=None,  # Critic only for this example
    )

    # Train critic
    print("\nTraining critic...")
    losses = trainer.train_critic(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=config.training.epochs,
    )

    # Evaluate
    print("\nEvaluating...")
    metrics = trainer.evaluate(val_pairs)

    print("\nResults:")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  Correlation: {metrics['correlation']:.3f}")

    # Test on new code
    print("\nTesting on new code...")

    test_codes = [
        '''
def safe_divide(a: float, b: float) -> float:
    """Safely divide two numbers."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
''',
        '''
def bad(x):
    return eval(x)
''',
    ]

    for i, code in enumerate(test_codes, 1):
        score = trainer.score_code(code)
        print(f"  Code {i}: Score = {score:.1f}/10")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("The critic has internalized coding judgment.")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()  # Windows compatibility
    main()
