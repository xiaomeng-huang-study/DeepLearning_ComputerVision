import tqdm
from collections import defaultdict
import torch
import pathlib
import os 
from torch.utils.tensorboard import SummaryWriter


def train(model: torch.nn.Module, epochs: int, optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.modules.loss._Loss, train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, device: torch.cuda.device = None, 
          scheduler: torch.optim.lr_scheduler.LRScheduler = None, clip_grads: bool = False,
          save_best_path: str = None, tb_log_dir: str = None, start_epoch: int = 0):
    
    history = defaultdict(list)

    best_accuracy = 0
    
    writer = SummaryWriter(log_dir= tb_log_dir)

    # create save path
    if save_best_path is not None:
        pathlib.Path(os.path.dirname(save_best_path)).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        # training
        with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as bar:
            model.train()
            bar.set_description(f"Epoch {epoch+1:02d} / {(start_epoch + epochs):02d}")

            loss_tmp = 0
            accuracy_tmp = 0
            for i, data in bar:
                inputs, labels = data

                if device is not None:
                    inputs, labels = inputs.to(device=device), labels.to(device=device)
                
                if inputs.isnan().any():
                    print("INPUT CONTAINS NAN. SKIPPING BATCH")
                    continue
                # Forward to get output
                outputs = model(inputs)
                
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad() 
                
                # Calculate Loss
                loss = criterion(outputs, labels)
                accuracy = (labels.argmax(dim=1) == outputs.argmax(dim=1)).float().mean()

                loss_tmp += float(loss)
                accuracy_tmp += float(accuracy)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # clip grads
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # Updating parameters
                optimizer.step()
                bar.set_postfix({"loss": loss_tmp / (i + 1), "accuracy": accuracy_tmp / (i + 1)})
        
        # add to history
        history["loss"].append(loss_tmp / len(train_loader))
        history["accuracy"].append(accuracy_tmp / len(train_loader))

        writer.add_scalar('Loss/train', loss_tmp / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', accuracy_tmp / len(train_loader), epoch)

        # evaluate model
        with tqdm.tqdm(enumerate(val_loader), total=len(val_loader)) as bar:
            bar.set_description(f"  - eval")

            model.eval()
            with torch.no_grad():
                loss_tmp = 0
                accuracy_tmp = 0
                for i, data in bar:
                    inputs, labels = data

                    if device is not None:
                        inputs, labels = inputs.to(device=device), labels.to(device=device)

                    outputs = model(inputs)
                        
                    loss = criterion(outputs, labels)
                    accuracy = (labels.argmax(dim=1) == outputs.argmax(dim=1)).float().mean()

                    loss_tmp += float(loss)
                    accuracy_tmp += float(accuracy)

                    if i < len(val_loader) - 1:
                        bar.set_postfix({"loss": history["loss"][-1], "accuracy": history["accuracy"][-1], 
                                        "val_loss": loss_tmp / (i + 1), "val_accuracy": accuracy_tmp / (i + 1)})
                    else:
                        # add to history
                        history["val_loss"].append(loss_tmp / len(val_loader))
                        history["val_accuracy"].append(accuracy_tmp / len(val_loader))
                            
                        bar.set_postfix({"loss": history["loss"][-1], "accuracy": history["accuracy"][-1], 
                                        "val_loss": history["val_loss"][-1], "val_accuracy": history["val_accuracy"][-1]})
        
        writer.add_scalar('Loss/val', loss_tmp / len(val_loader), epoch)
        writer.add_scalar('Accuracy/val', accuracy_tmp / len(val_loader), epoch)
        
        # update best accuracy
        best_accuracy = max(best_accuracy, history["val_accuracy"][-1])
        if save_best_path is not None:
            torch.save(model, save_best_path)
        
        # update lr scheduler
        if scheduler is not None:
            scheduler.step(history["val_loss"][-1])
        
        
        print()
        
    writer.flush()
    writer.close()
    return history