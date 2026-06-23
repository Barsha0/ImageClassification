# Evaluation to run model on the validation image, compare its prediction with real answer

# Basically -> training = studying, evaluate = taking the final exam

def evaluate_model(model, val_data):
    loss, accuracy = model.evaluate(val_data)

    print(f"Validation Accuracy : {accuracy * 100:.2f}%")
    print(f"Validation Loss : {loss:.4f}")