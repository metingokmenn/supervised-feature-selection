from src.data_handler import DataHandler
from src.feature_selection import FeatureSelector
from src.trainer import ModelTrainer
from src.evaluator import Evaluator

def main():
    # 1. Data Preparation
    data_path = "dataset/OnlineNewsPopularity.csv"
    handler = DataHandler(data_path)
    X_train, X_test, y_train, y_test = handler.load_and_preprocess()

    # 2. Initialize Modules
    selector = FeatureSelector(X_train, y_train)
    trainer = ModelTrainer(X_train, y_train)
    evaluator = Evaluator(X_test, y_test)

    results_list = []
    feature_sets = {}
    
    # Helper function to run a cycle
    def run_cycle(features, name):
        # Train
        train_stats = trainer.train_and_validate(features)
        # Evaluate
        metrics, preds = evaluator.evaluate(
            train_stats['model'], 
            train_stats['scaler'], 
            features, 
            name, 
            train_stats
        )
        return metrics, preds

    # --- SCENARIO A: All Features [cite: 29] ---
    all_features = X_train.columns.tolist()
    feature_sets["All Features"] = all_features
    res_all, _ = run_cycle(all_features, "All Features")
    results_list.append(res_all)

    # --- SCENARIO B: 3 Feature Selection Methods (15 Features) [cite: 18] ---
    
    # 1. Filter Method
    feat_filter = selector.filter_method(n_features=15)
    feature_sets["Filter"] = feat_filter
    res_filt, _ = run_cycle(feat_filter, "Filter")
    results_list.append(res_filt)

    # 2. Wrapper Method
    feat_wrapper = selector.wrapper_method(n_features=15)
    feature_sets["Wrapper"] = feat_wrapper
    res_wrap, _ = run_cycle(feat_wrapper, "Wrapper")
    results_list.append(res_wrap)

    # 3. Embedded Method
    feat_embed = selector.embedded_method(n_features=15)
    feature_sets["Embedded"] = feat_embed
    res_embed, _ = run_cycle(feat_embed, "Embedded")
    results_list.append(res_embed)

    # --- SCENARIO C: Optimization of Best Method [cite: 32] ---
    # Identify best method based on Test Accuracy (excluding 'All Features')
    best_method_row = max(results_list[1:], key=lambda x: x['Test Accuracy'])
    best_method_name = best_method_row['Method']
    print(f"\n[Analysis] Best performing method identified: {best_method_name}")

    # Search for optimal k features (range 5 to 40)
    best_k_score = 0
    best_k_features = []
    
    for k in range(5, 45, 5):
        if best_method_name == "Filter":
            current_feats = selector.filter_method(n_features=k)
        elif best_method_name == "Wrapper":
            current_feats = selector.wrapper_method(n_features=k)
        elif best_method_name == "Embedded":
            current_feats = selector.embedded_method(n_features=k)
        else:
            continue

        # Quick check using CV Accuracy from training
        t_stats = trainer.train_and_validate(current_feats)
        if t_stats['cv_accuracy'] > best_k_score:
            best_k_score = t_stats['cv_accuracy']
            best_k_features = current_feats
        
        print(f"   -> Optimization k={k}, CV Acc={t_stats['cv_accuracy']:.4f}")

    # Final Train & Evaluate for Optimum Model
    optimum_name = f"{best_method_name} (Optimum)"
    feature_sets[optimum_name] = best_k_features
    res_opt, y_pred_opt = run_cycle(best_k_features, optimum_name)
    results_list.append(res_opt)

    # --- REPORTING ---
    metrics_df = evaluator.save_results(results_list, feature_sets)
    print("\nFinal Results Table:")
    print(metrics_df[["Method", "Feature Count", "Test Accuracy", "Test F1-Score"]])

    # Plot Confusion Matrix for the best model [cite: 38]
    evaluator.plot_confusion_matrix(y_pred_opt, optimum_name)

    print("\n[Done] All tasks completed. Please check the 'results' directory.")

if __name__ == "__main__":
    main()