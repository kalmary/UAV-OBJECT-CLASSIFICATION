import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from torch.utils.data import DataLoader

import pathlib as pth
import sys

from model import YOLO
from _data_loader import *

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.yolo_accuracy import YOLOMetrics
from utils.yolo_loss import YOLOLoss, ModelEMA
from utils.anchors_search import find_optimal_anchors
from utils.non_max_suppression import non_max_suppression

from utils.nn_utils.src.accuracy_metrics import get_dataset_len, calculate_class_weights
from utils.nn_utils.src.file_handling import wrap_hist

from tqdm import tqdm
from typing import Union, Generator

def train_model(training_dict: dict) -> Union[Generator[tuple[nn.Module, dict], None, None],
                                              Generator[tuple[None, dict], None, None]]:
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_gpu
    device_loss = device_gpu

    # FIX 1: resolution_xy as tuple (350, 350) instead of int 350
    train_dataset = Dataset(path_dir=training_dict['data_path_train'],
                            resolution_xy=(350, 350),  # ← FIXED: tuple
                            batch_size=training_dict['batch_size'],
                            shuffle=True,
                            device=device_loader)

    trainLoader = DataLoader(train_dataset,
                             batch_size=None,
                             num_workers = 10,
                             pin_memory=True)
    


    # FIX 2: base_dir → path_dir
    val_dataset = Dataset(path_dir=training_dict['data_path_val'],  # ← FIXED: path_dir
                          resolution_xy=(350, 350),  # ← FIXED: tuple
                          batch_size=training_dict['batch_size'],
                          shuffle=False,
                          device=device_loader)

    valLoader = DataLoader(val_dataset,
                           batch_size=None,
                           num_workers = 10,
                           pin_memory=True)
    try:

        total_t = get_dataset_len(trainLoader)
        total_v = get_dataset_len(valLoader)

        class_weights_t = calculate_class_weights(loader = trainLoader,  # TODO change to YOLO version
                                                num_classes=training_dict['num_classes'], 
                                                method='effective',
                                                total=total_t, 
                                                device=device_loss,
                                                verbose=False)
        
        class_weights_v = calculate_class_weights(loader = valLoader, 
                                                num_classes=training_dict['num_classes'], 
                                                method='effective',
                                                total=total_v, 
                                                device=device_loss,
                                                verbose=False)
        
        # FIX 3: recreate dataset with weights - resolution_xy as tuple
        train_dataset = Dataset(path_dir=training_dict['data_path_train'],
                                resolution_xy=(350, 350),  # ← FIXED: tuple
                                batch_size=training_dict['batch_size'],
                                shuffle=True,
                                weights=class_weights_t,
                                device=device_loader)

        trainLoader = DataLoader(train_dataset,
                                batch_size=None,
                                num_workers = 10,
                                pin_memory=True)
        


        # FIX 4: base_dir → path_dir
        val_dataset = Dataset(path_dir=training_dict['data_path_val'],  # ← FIXED: path_dir
                            resolution_xy=(350, 350),  # ← FIXED: tuple
                            batch_size=training_dict['batch_size'],
                            shuffle=False,
                            weights=class_weights_v,
                            device=device_loader)

        valLoader = DataLoader(val_dataset,
                            batch_size=None,
                            num_workers = 10,
                            pin_memory=True)
        

        anchors_t = find_optimal_anchors(trainLoader, 
                            num_anchors=9,
                            img_size=350,
                            max_samples=10000,
                            total=total_t,
                            verbose=False)

        anchors_v = find_optimal_anchors(valLoader,
                            num_anchors=9,
                            img_size=350,
                            max_samples=10000,
                            total=total_v,
                            verbose=False)


        if training_dict['model'] is None:
            model = YOLO(config_name=training_dict['model_config'],
                            num_classes=training_dict['num_classes'])
        else:
            model = training_dict['model']

        model.to(training_dict['device'])
        ema = ModelEMA(model, decay=0.9999)
        
        criterion_t = YOLOLoss(num_classes=training_dict['num_classes'],
                                anchors=anchors_t,
                                img_size=training_dict['img_size'],
                                focal_alpha=class_weights_t,
                                focal_gamma=training_dict['focal_loss_gamma']).to(device_loss)
        
        criterion_v = YOLOLoss(num_classes=training_dict['num_classes'],
                               anchors=anchors_v,
                               img_size=training_dict['img_size'],
                               focal_alpha=class_weights_v,
                               focal_gamma=training_dict['focal_loss_gamma']).to(device_loss)
        
        metrics = YOLOMetrics(num_classes=training_dict['num_classes'])

        optimizer = optim.AdamW(model.parameters(), lr = training_dict['learning_rate'], weight_decay=training_dict['weight_decay'])

        scheduler = OneCycleLR(
            optimizer,
            max_lr=training_dict['learning_rate'],  # Example: Adjust based on your LR range test
            total_steps=total_t*training_dict['epochs'],
            pct_start=training_dict['pc_start'],  # % of steps for warm-up
            anneal_strategy='cos',  # Cosine annealing
            div_factor=training_dict['div_factor'],  # Initial LR will be max_lr / x
            final_div_factor=training_dict['final_div_factor'],  # Final LR
            cycle_momentum=True  # Cycle momentum as well
        )

        loss_hist = []

        loss_v_hist = []
        mAP05_hist = []
        mAP0595_hist = []
        precision_hist = []
        recall_hist = []
        f1_hist = []


        repeat_pbar = tqdm(range(training_dict['train_repeat']), 
                            desc="Training Repetition", 
                            unit="repeat",
                            position=1, 
                            leave=False) 

        for _ in repeat_pbar:

            epoch_pbar = tqdm(range(training_dict['epochs']), 
                                desc="Epoch Progress", 
                                unit="epoch",
                                position=2, 
                                leave=False) 

            for epoch in epoch_pbar:

                epoch_loss_t = 0.
                epoch_loss_v = 0.

                epoch_samples_t = 0
                epoch_samples_v = 0


                progressbar_t = tqdm(trainLoader, 
                                        desc=f"Epoch training {epoch+1}/ {training_dict['epochs']}", 
                                        total=total_t, 
                                        position=3,
                                        leave=False)
                
                model.train(True)
                
                for batch_x, batch_y in progressbar_t:
                    
                    batch_x = batch_x.to(training_dict['device'])
                    outputs = model(batch_x)

                    outputs = outputs.to(device_loss)
                    batch_y = batch_y.to(device_loss)

                    loss_t = criterion_t(outputs, batch_y)

                    optimizer.zero_grad()
                    loss_t.backward()
                    optimizer.step()
                    ema.update(model)

                    try:
                        scheduler.step()
                    except Exception:
                        pass

                    current_lr = optimizer.param_groups[0]['lr']

                    epoch_loss_t += loss_t.item() * batch_y.size(0)
                    epoch_samples_t += batch_y.size(0)

                    avg_loss_t = epoch_loss_t / epoch_samples_t

                    progressbar_t.set_postfix({
                        "Loss_train": f"{avg_loss_t:.6f}",
                        # "Acc_train": f"{avg_accuracy_t:.6f}",
                        "learning_rate": f"{current_lr:.10f}"
                    })

                loss_hist.append(avg_loss_t)

                metrics.reset()
                progressbar_v = tqdm(valLoader, desc=f"Epoch validation {epoch + 1}/ {training_dict['epochs']}", total=total_v, position=3, leave=False)



                model.eval()
                with torch.no_grad():
                    for batch_x, batch_y in progressbar_v:

                        batch_x = batch_x.to(training_dict['device'])
                        outputs = model(batch_x)

                        outputs = [o.to(device_loss) for o in outputs]

                        batch_y = batch_y.to(device_loss)

                        loss_v = criterion_v(outputs, batch_y)

                        predictions = non_max_suppression(outputs)
                        metrics.update(predictions, batch_y)

                        epoch_loss_v += loss_v.item() * batch_y.size(0)
                        epoch_samples_v += batch_y.size(0)

                        avg_loss_v = epoch_loss_v / epoch_samples_v
                        progressbar_v.set_postfix({
                            "Loss_val": f"{avg_loss_v:.6f}"
                        })

                loss_v_hist.append(avg_loss_v)
                


                metrics_values = metrics.compute()
                mAP05_hist.append(metrics_values['mAP@0.5'])
                mAP0595_hist.append(metrics_values['mAP@0.5:0.95'])
                precision_hist.append(metrics_values['precision'])
                recall_hist.append(metrics_values['recall'])
                f1_hist.append(metrics_values['f1'])

                hist_dict = wrap_hist(loss_hist = loss_hist,
                                      loss_hist_val = loss_v_hist,
                                      mAP05_hist = mAP05_hist,
                                      mAP0595_hist = mAP0595_hist,
                                      precision_hist = precision_hist,
                                      recall_hist = recall_hist,
                                      f1_hist = f1_hist)

                yield model, hist_dict

                epoch_pbar.set_postfix({
                    "Loss_train": f"{avg_loss_t:.6f}",
                    "Loss_val": f"{avg_loss_v:.6f}",
                    "mAP@0.5_val": f"{hist_dict['mAP@0.5']:.6f}",
                    "mAP@0.5:0.95_val": f"{hist_dict['mAP@0.5:0.95']:.6f}",
                    "precision:": f"{hist_dict['precision']:.6f}",
                    "recall:": f"{hist_dict['recall']:.6f}",
                    "f1:": f"{hist_dict['f1']:.6f}",
                    "learning_rate_max": f"{training_dict['learning_rate']:.10f}"
                })

    except Exception as e:
        print(f"Error during training: {e}")
        try:
            del model
        except Exception as e:
            pass
        torch.cuda.empty_cache()
        yield None, {}


