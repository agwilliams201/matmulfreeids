import time
import psutil
import argparse
import platform
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from model import *

try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
except ImportError:
    PYRAPL_AVAILABLE = False

def plot_metrics(results, model_type):
    """Plot metrics for the model run."""
    if 'precision' in results[model_type] and 'recall' in results[model_type]:
        plt.figure(figsize=(10, 6))
        plt.plot(results[model_type]['recall'], results[model_type]['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_type} - Precision-Recall Curve')
        plt.savefig(f'{model_type}_pr_curve.png')
        plt.close()

    if 'cpu_usage' in results[model_type]:
        plt.figure(figsize=(10, 6))
        plt.plot(results[model_type]['cpu_usage'])
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.title(f'{model_type} - CPU Usage Over Time')
        plt.savefig(f'{model_type}_cpu_usage.png')
        plt.close()

def estimate_cpu_power(cpu_percent):
    """Estimate CPU power consumption based on usage."""
    tdp_watts = 65
    return (tdp_watts * cpu_percent) / 100.0

def run_models_and_measure(data_dir, chunk_size, load_model=True, epochs=20):
    """Run both models and measure their performance metrics."""
    results = {}
    
    is_linux = platform.system() == 'Linux'
    if PYRAPL_AVAILABLE and is_linux:
        pyRAPL.setup()
    else:
        print("pyRAPL unavailable. Using CPU usage for power estimation.")

    for model_type in ['standard', 'matmul_free']:
        print(f"\nRunning {model_type} model...")
        
        start_time = time.time()
        cpu_usage = []
        energy_meter = None
        
        if PYRAPL_AVAILABLE and is_linux:
            energy_meter = pyRAPL.Measurement(f"{model_type}_energy")
            energy_meter.begin()

        try:
            def measure_callback():
                cpu_percent = psutil.cpu_percent()
                cpu_usage.append(cpu_percent)
            
            ids_processor, test_data = integrate_with_hardware_efficient_ids(
                data_dir=data_dir,
                binary_classification=True,
                chunk_size=chunk_size,
                perform_hyperparameter_tuning=False,
                load_model=load_model,
                model_type=model_type,
                epochs=epochs,
                callback=measure_callback
            )

            predictions, prediction_scores = ids_processor.detect_anomalies(
                test_data['X_test'], 
                return_scores=True
            )

            precision, recall, _ = precision_recall_curve(
                test_data['y_test'],
                prediction_scores
            )
            pr_auc = auc(recall, precision)
            
            classification_metrics = classification_report(
                test_data['y_test'],
                predictions,
                output_dict=True
            )

        finally:
            if energy_meter:
                energy_meter.end()

        end_time = time.time()
        runtime = end_time - start_time
        peak_cpu = max(cpu_usage) if cpu_usage else 0
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
        
        if energy_meter:
            energy_consumed = energy_meter.result.energy / 1e6
        else:
            power_readings = [estimate_cpu_power(usage) for usage in cpu_usage]
            energy_consumed = np.mean(power_readings) * runtime if power_readings else 0

        results[model_type] = {
            'runtime_seconds': runtime,
            'peak_cpu_usage_percent': peak_cpu,
            'avg_cpu_usage_percent': avg_cpu,
            'energy_consumed_joules': energy_consumed,
            'classification_report': classification_metrics,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'pr_auc': pr_auc,
            'cpu_usage': cpu_usage
        }

        print(f"\n{model_type.upper()} MODEL RESULTS:")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Peak CPU Usage: {peak_cpu:.1f}%")
        print(f"Average CPU Usage: {avg_cpu:.1f}%")
        print(f"Energy Consumed: {energy_consumed:.2f} Joules")
        print(f"PR-AUC Score: {pr_auc:.3f}")
        print("\nClassification Report:")
        for label in ['0.0', '1.0']:
            metrics = classification_metrics[label]
            print(f"\nClass {label}:")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-score: {metrics['f1-score']:.3f}")

        plot_metrics(results, model_type)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and measure IDS models")
    parser.add_argument('--data_dir', type=str, default='data/',
                    help='Directory containing the data files')
    parser.add_argument('--chunk_size', type=int, default=50000,
                    help='Chunk size for data processing')
    parser.add_argument('--load_model', action='store_true',
                    help='Load saved models instead of training')
    parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs')

    args = parser.parse_args()

    try:
        results = run_models_and_measure(
            args.data_dir,
            args.chunk_size,
            load_model=args.load_model,
            epochs=args.epochs
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results_{timestamp}.json', 'w') as f:
            serializable_results = {}
            for model_type, model_results in results.items():
                serializable_results[model_type] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in model_results.items()
                }
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to results_{timestamp}.json")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()