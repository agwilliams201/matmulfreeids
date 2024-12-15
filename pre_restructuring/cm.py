import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import IDSProcessor
from measure_models import integrate_with_hardware_efficient_ids

def generate_confusion_matrix(data_dir, model_type='matmul_free'):
    """Generate and display a confusion matrix for the given model."""
    print("Loading data and model...")
    ids_processor, test_data = integrate_with_hardware_efficient_ids(
        data_dir=data_dir,
        binary_classification=True,
        load_model=True,
        model_type=model_type
    )

    print("Generating predictions...")
    predictions = ids_processor.detect_anomalies(test_data['X_test'])

    print("Calculating confusion matrix...")
    cm = confusion_matrix(test_data['y_test'], predictions)
    
    print("Displaying confusion matrix...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attack"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_type}")
    plt.savefig(f"confusion_matrix_{model_type}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confusion matrix for IDS model")
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing the data files')
    parser.add_argument('--model_type', type=str, choices=['matmul_free', 'standard'],
                        default='matmul_free', help='Type of model to use')

    args = parser.parse_args()

    generate_confusion_matrix(
        data_dir=args.data_dir,
        model_type=args.model_type
    )