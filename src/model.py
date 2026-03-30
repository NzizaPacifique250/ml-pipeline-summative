import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.preprocessing import get_data_generators, IMAGE_SIZE

def build_model():
    """
    Builds the MobileNetV2 architecture with a custom classification head.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(train_dir, validation_dir, model_save_path='models/model.keras', epochs=10):
    """
    Trains the model from scratch based on provided directories.
    """
    train_gen, val_gen, _ = get_data_generators(train_dir, validation_dir)
    
    model = build_model()
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

    steps_per_epoch = train_gen.samples // train_gen.batch_size if train_gen.samples >= train_gen.batch_size else 1
    validation_steps = val_gen.samples // val_gen.batch_size if val_gen.samples >= val_gen.batch_size else 1

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    # Ensure model is saved if not caught by checkpoint
    if not os.path.exists(model_save_path):
        model.save(model_save_path)
    
    return history

def retrain_model(train_dir, validation_dir, existing_model_path='models/model.keras', new_model_path='models/model_retrained.keras', epochs=5):
    """
    Retrains an existing model using new data mixed in the train_dir.
    It unfreezes some layers or just continues training the custom head.
    """
    if not os.path.exists(existing_model_path):
        print("Existing model not found. Training from scratch.")
        return train_model(train_dir, validation_dir, model_save_path=new_model_path, epochs=epochs)
        
    print(f"Loading existing model from {existing_model_path}")
    model = load_model(existing_model_path)
    
    train_gen, val_gen, _ = get_data_generators(train_dir, validation_dir)
    
    # Optional: We could unfreeze the top layers of MobileNetV2, but just fitting again is fine.
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(new_model_path, save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

    steps_per_epoch = train_gen.samples // train_gen.batch_size if train_gen.samples >= train_gen.batch_size else 1
    validation_steps = val_gen.samples // val_gen.batch_size if val_gen.samples >= val_gen.batch_size else 1

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    if not os.path.exists(new_model_path):
        model.save(new_model_path)
        
    # Replace the old model with the new one
    try:
        import shutil
        shutil.copyfile(new_model_path, existing_model_path)
    except Exception as e:
        print(f"Could not replace original model: {e}")
        
    return history
