"""CLI commands for QSAR model training and prediction."""

import click


@click.group()
def model():
    """QSAR model training, evaluation, and prediction."""
    pass


@model.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Training data file.")
@click.option("--activity-col", required=True,
              help="Column name with activity values.")
@click.option("--model-type", default="random_forest",
              type=click.Choice(["random_forest", "gradient_boosting", "xgboost"]),
              help="ML model type.")
@click.option("--task", default="regression",
              type=click.Choice(["regression", "classification"]),
              help="Task type.")
@click.option("--model-path", "-m", default="model.joblib",
              type=click.Path(), help="Output model file path.")
@click.option("--feature-source", default="descriptors",
              type=click.Choice(["descriptors", "fingerprints"]),
              help="Feature type to use.")
@click.option("--fp-type", default="morgan_r2_2048",
              help="Fingerprint type (if feature-source=fingerprints).")
@click.option("--test-size", type=float, default=0.2,
              help="Fraction for test set.")
@click.option("--split", default="random",
              type=click.Choice(["random", "scaffold"]),
              help="Split method.")
@click.option("--scale/--no-scale", default=True,
              help="Scale features with StandardScaler.")
@click.option("--cv", type=int, default=0,
              help="Run N-fold cross-validation (0 = skip).")
@click.pass_context
def train(ctx, input_path, activity_col, model_type, task, model_path,
          feature_source, fp_type, test_size, split, scale, cv):
    """Train a QSAR model on molecular data."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.descriptors import compute_descriptors_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase2.qsar.data_prep import (
        extract_feature_matrix, extract_labels,
        random_split, scaffold_split, scale_features,
    )
    from drugflow.phase2.qsar.models import train_model
    from drugflow.phase2.qsar.evaluation import evaluate_model, cross_validate
    from drugflow.phase2.qsar.persistence import save_model

    # Load and prepare data
    click.echo(f"Loading data from {input_path}...")
    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    # Compute features based on source type
    if feature_source == "descriptors":
        click.echo("Computing descriptors...")
        dataset = compute_descriptors_dataset(dataset)
    else:
        click.echo("Computing fingerprints...")
        dataset = compute_fingerprints_dataset(dataset)

    # Extract features and labels
    click.echo("Extracting feature matrix...")
    X, feature_names, valid_indices = extract_feature_matrix(
        dataset,
        feature_source=feature_source,
        fp_type=fp_type if feature_source == "fingerprints" else None,
    )
    y = extract_labels(dataset, activity_col, valid_indices)

    click.echo(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Split data
    click.echo(f"Splitting data ({split}, test_size={test_size})...")
    if split == "scaffold":
        X_train, X_test, y_train, y_test = scaffold_split(
            dataset, X, y, valid_indices, test_size=test_size,
        )
    else:
        X_train, X_test, y_train, y_test = random_split(
            X, y, test_size=test_size,
        )

    click.echo(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # Scale features
    scaler = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    # Cross-validation
    if cv > 0:
        click.echo(f"\nRunning {cv}-fold cross-validation...")
        cv_results = cross_validate(
            X_train, y_train,
            model_type=model_type, task=task, n_folds=cv,
        )
        primary = "r2" if task == "regression" else "accuracy"
        mean = cv_results["mean_metrics"].get(primary, 0)
        std = cv_results["std_metrics"].get(primary, 0)
        click.echo(f"  CV {primary}: {mean:.4f} (+/- {std:.4f})")

    # Train model
    click.echo(f"\nTraining {model_type} ({task})...")
    qsar_model = train_model(
        X_train, y_train,
        model_type=model_type,
        task=task,
        feature_names=feature_names,
        scaler=scaler,
        dataset_name=input_path,
    )

    # Evaluate on test set
    y_pred = qsar_model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred, task)

    click.echo(f"\nTest set metrics:")
    for name, val in test_metrics.items():
        click.echo(f"  {name}: {val:.4f}")

    # Feature importances
    importances = qsar_model.get_feature_importances()
    if importances:
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        click.echo(f"\nTop features:")
        for name, imp in sorted_imp[:10]:
            click.echo(f"  {name}: {imp:.4f}")

    # Save
    save_model(qsar_model, model_path)
    click.echo(f"\nModel saved to {model_path}")


@model.command()
@click.option("--model-path", "-m", required=True,
              type=click.Path(exists=True), help="Trained model file.")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Test data file.")
@click.option("--activity-col", required=True,
              help="Column name with true activity values.")
@click.pass_context
def evaluate(ctx, model_path, input_path, activity_col):
    """Evaluate a trained model on test data."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.descriptors import compute_descriptors_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase2.qsar.persistence import load_model
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix, extract_labels
    from drugflow.phase2.qsar.evaluation import evaluate_model

    qsar_model = load_model(model_path)
    click.echo(f"Model: {qsar_model.model_type} ({qsar_model.task})")

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    # Determine feature source from model metadata
    if qsar_model.feature_names and qsar_model.feature_names[0].startswith(("morgan_", "maccs_")):
        feature_source = "fingerprints"
        dataset = compute_fingerprints_dataset(dataset)
    else:
        feature_source = "descriptors"
        dataset = compute_descriptors_dataset(dataset)

    X, _, valid_indices = extract_feature_matrix(
        dataset,
        feature_source=feature_source,
        feature_names=qsar_model.feature_names if feature_source == "descriptors" else None,
        fp_type=qsar_model.feature_names[0].rsplit("_bit_", 1)[0] if feature_source == "fingerprints" else None,
    )
    y = extract_labels(dataset, activity_col, valid_indices)

    y_pred = qsar_model.predict(X)
    metrics = evaluate_model(y, y_pred, qsar_model.task)

    click.echo(f"\nEvaluation metrics ({len(y)} samples):")
    for name, val in metrics.items():
        click.echo(f"  {name}: {val:.4f}")


@model.command()
@click.option("--model-path", "-m", required=True,
              type=click.Path(exists=True), help="Trained model file.")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input molecule file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output predictions file.")
@click.pass_context
def predict(ctx, model_path, input_path, output):
    """Predict activity for new molecules using a trained model."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.descriptors import compute_descriptors_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase2.qsar.persistence import load_model
    from drugflow.phase2.qsar.prediction import predict_dataset
    from drugflow.phase1.data.writers import write_file

    qsar_model = load_model(model_path)
    click.echo(f"Model: {qsar_model.model_type} ({qsar_model.task})")

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    # Determine feature source
    if qsar_model.feature_names and qsar_model.feature_names[0].startswith(("morgan_", "maccs_")):
        feature_source = "fingerprints"
        fp_type = qsar_model.feature_names[0].rsplit("_bit_", 1)[0]
        dataset = compute_fingerprints_dataset(dataset)
    else:
        feature_source = "descriptors"
        fp_type = None
        dataset = compute_descriptors_dataset(dataset)

    click.echo(f"Predicting {len(dataset.valid_records)} molecules...")
    dataset = predict_dataset(
        qsar_model, dataset,
        feature_source=feature_source, fp_type=fp_type,
    )

    # Show sample predictions
    count = 0
    for rec in dataset.valid_records[:5]:
        pred = rec.properties.get("predicted_activity")
        if pred is not None:
            click.echo(
                f"  {rec.source_id or rec.record_id}: "
                f"predicted={pred:.4f}"
            )
            count += 1

    write_file(dataset, output)
    click.echo(f"\nPredictions saved to {output}")


@model.command()
@click.option("--model-path", "-m", required=True,
              type=click.Path(exists=True), help="Trained model file.")
@click.pass_context
def info(ctx, model_path):
    """Show information about a trained model."""
    from drugflow.phase2.qsar.persistence import load_model

    qsar_model = load_model(model_path)
    summary = qsar_model.summary()

    click.echo(f"Model: {summary['model_type']}")
    click.echo(f"Task: {summary['task']}")
    click.echo(f"Features: {summary['n_features']}")

    if summary.get("training_metrics"):
        click.echo("\nTraining metrics:")
        for name, val in summary["training_metrics"].items():
            click.echo(f"  {name}: {val:.4f}")

    if summary.get("metadata"):
        click.echo("\nMetadata:")
        meta = summary["metadata"]
        for key in ["trained_at", "n_train_samples", "dataset_name"]:
            if key in meta:
                click.echo(f"  {key}: {meta[key]}")

    if summary.get("top_features"):
        click.echo("\nTop features by importance:")
        for name, imp in summary["top_features"]:
            click.echo(f"  {name}: {imp:.4f}")
