import math
from typing import Iterable, Optional # Added Optional

import torch
from iopath.common.file_io import g_pathmgr as pathmgr

# Corrected import paths
from util import lr_sched as lr_sched 
from util import misc as misc       


def check_for_nan_inf(tensor, name="tensor"):
    """Helper function to check for NaN/Inf in tensors"""
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")
        return True
    return False

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    fp32=False,
    # data_loader_val: Optional[Iterable] = None, # NEW: Validation data loader
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = args.print_freq if hasattr(args, 'print_freq') else 20 # Added print_freq from args

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # --- Training Phase ---
    epoch_train_loss = 0.0
    num_train_batches = 0

    for data_iter_step, (samples, _) in enumerate( # Assuming dataloader returns (samples, targets_or_empty_list)
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        # targets are not used in MAE pretraining for the model input, but kept for compatibility
        # targets = [tgt.to(device, non_blocking=True) for tgt in targets] # If targets exist

        if len(samples.shape) == 6: # Handling potential extra dimension from some video datasets
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        mask_ratio = args.mask_ratio

        # with torch.cuda.amp.autocast(enabled=not fp32):
        with torch.amp.autocast('cuda', enabled=not fp32):
            loss, _, _, _, _ = model( # model call might vary based on MAE implementation
                samples,
                targets=None, # Explicitly pass None if targets are not used by model forward for MAE
                mask_ratio=mask_ratio,
                mask_strategy=args.masking_strategy,
            )

        loss_value = loss.item()
        epoch_train_loss += loss_value
        num_train_batches += 1

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            # Clean up checkpoints if needed (optional, depends on your checkpointing)
            # for _ in range(args.num_checkpoint_del):
            #     try:
            #         path = misc.get_last_checkpoint(args)
            #         pathmgr.rm(path)
            #         print(f"Removed checkpoint {path}")
            #     except Exception as e:
            #         print(f"Error removing checkpoint: {e}")
            #         pass
            raise Exception(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=args.mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("iter_train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    avg_epoch_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
    if log_writer is not None:
        log_writer.add_scalar("epoch_train_loss", avg_epoch_train_loss, epoch)
    print(f"Epoch: {epoch}, Average Train Loss: {avg_epoch_train_loss:.4f}")


    # # --- Validation Phase (NEW) ---
    # if data_loader_val:
    #     model.eval() # Set model to evaluation mode
    #     epoch_val_loss = 0.0
    #     num_val_batches = 0
    #     val_metric_logger = misc.MetricLogger(delimiter="  ") # Separate logger for validation
    #     val_header = "Val_Epoch: [{}]".format(epoch)

    #     for data_iter_step, (samples, _) in enumerate(
    #          val_metric_logger.log_every(data_loader_val, print_freq, val_header) # Use val_metric_logger
    #     ):
    #         samples = samples.to(device, non_blocking=True)
    #         # targets = [tgt.to(device, non_blocking=True) for tgt in targets] # If targets exist

    #         if len(samples.shape) == 6:
    #             b, r, c, t, h, w = samples.shape
    #             samples = samples.reshape(b * r, c, t, h, w)

    #         mask_ratio = args.mask_ratio # Use same mask ratio or a fixed one for val

    #         with torch.no_grad(): # No gradient calculation for validation
    #             # with torch.cuda.amp.autocast(enabled=not fp32):
    #             with torch.amp.autocast('cuda', enabled=not fp32):
    #                 loss, _, _, _, _ = model(
    #                     samples,
    #                     targets=None,
    #                     mask_ratio=mask_ratio,
    #                     mask_strategy=args.masking_strategy,
    #                 )
            
    #         loss_value = loss.item()
    #         epoch_val_loss += loss_value
    #         num_val_batches += 1
    #         val_metric_logger.update(loss=loss_value)


    #     avg_epoch_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
    #     if log_writer is not None:
    #         log_writer.add_scalar("epoch_val_loss", avg_epoch_val_loss, epoch)
    #     print(f"Epoch: {epoch}, Average Validation Loss: {avg_epoch_val_loss:.4f}")
        
    #     # gather the stats from all processes for validation
    #     val_metric_logger.synchronize_between_processes()
    #     print("Averaged Validation stats:", val_metric_logger)
        # model.train() # Set model back to training mode
        
    model.train() # Set model back to training mode

    # gather the stats from all processes for training
    metric_logger.synchronize_between_processes()
    print("Averaged Training stats:", metric_logger)
    
    # Return a dictionary including both train and val loss if computed
    stats_to_return = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if data_loader_val:
    #     stats_to_return['val_loss'] = avg_epoch_val_loss # Or use val_metric_logger.loss.global_avg
    return stats_to_return
