import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import load_data
from model import build_model

def train_model(data_dir):
    train_data, val_data = load_data(data_dir)
    
    model = build_model()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=[early_stopping, model_checkpoint]
    )

    print("Model training complete!")
    return model

if __name__ == "__main__":
    train_model('data')
