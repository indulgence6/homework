import numpy as np
import os
from model import accuracy

def train(model, X_train, y_train, X_val, y_val, params):
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(params['epochs']):
        # Learning rate decay
        lr = params['lr'] * (params['lr_decay'] ** (epoch // params['lr_step']))
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch SGD
        for i in range(0, len(X_train), params['batch_size']):
            X_batch = X_shuffled[i:i+params['batch_size']]
            y_batch = y_shuffled[i:i+params['batch_size']]
            
            # Forward and backward
            logits = model.forward(X_batch)
            grads = model.backward(X_batch, y_batch, params['lambda_reg'])
            
            # Update parameters
            model.W1 -= lr * grads[0]
            model.b1 -= lr * grads[1]
            model.W2 -= lr * grads[2]
            model.b2 -= lr * grads[3]
        
        # Validation
        val_logits = model.forward(X_val)
        val_loss = model.compute_loss(val_logits, y_val, params['lambda_reg'])
        val_acc = accuracy(val_logits, y_val)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.savez('/home/add_disk/zengchengjie/HW/best_model.npz', 
                    W1=model.W1, b1=model.b1,
                    W2=model.W2, b2=model.b2)
        
        # Record history
        history['train_loss'].append(model.compute_loss(
            model.forward(X_train), y_train, params['lambda_reg']))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{params['epochs']} | "
              f"LR: {lr:.4f} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    return history