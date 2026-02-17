from Output_evaluate_gnn_metrics import evaluate_model

if __name__ == "__main__":

    results = evaluate_model(
        model_path="models_gnn/best_model.pt",
        num_episodes=30,
        deterministic=True
    )

    print("\n===== PAPER-LEVEL METRICS =====")
    print(results)
