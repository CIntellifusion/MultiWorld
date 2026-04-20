import os, torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.save_optimizer(accelerator, optimizer, f"optimizer-step-{self.num_steps}.ckpt")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            self.save_optimizer(accelerator, optimizer, f"optimizer-epoch-{epoch_id}.ckpt")


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.save_optimizer(accelerator, optimizer, f"optimizer-step-{self.num_steps}.ckpt")
    
    def save_optimizer(self, accelerator: Accelerator, optimizer: torch.optim.Optimizer, file_name: str):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save({"optimizer": optimizer.state_dict(),
                              "num_steps": self.num_steps}, path, safe_serialization=False)

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

class ModelTSLogger(ModelLogger):
    def __init__(self, output_path, 
                 remove_prefix_in_ckpt=None, 
                 state_dict_converter=lambda x:x,
                ):
        # TODO: remove dependency on accelerator
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.create_tb_writer()
    
    def is_main_process(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0
    
    def create_tb_writer(self):
        output_path = self.output_path
        # create TensorBoard writer and directory only on the main process
        if self.is_main_process():
            self.tb_writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard'))
            os.makedirs(self.output_path, exist_ok=True)
            
            print(f"TensorBoard logs will be saved to: {os.path.join(output_path, 'tensorboard')}")
            print(f"View with: tensorboard --logdir {os.path.join(output_path, 'tensorboard')}")
        else:
            self.tb_writer = None
            
    def log(self, tag, value):
        if self.is_main_process(): 
            self.tb_writer.add_scalar(tag, value, self.num_steps)
            
    def on_step_end(self, accelerator, model, optimizer, save_steps=None, loss=None, **kwargs):
        self.num_steps += 1
        # Gather loss from all processes
        gathered_loss = None
        if loss is not None:
            if accelerator is not None:
                # use accelerator.gather to collect loss from all processes
                gathered_loss = accelerator.gather(loss).mean().item()
            else:
                gathered_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        # log only on the main process
        if self.is_main_process():
            # Log loss to TensorBoard
            if gathered_loss is not None:
                self.log('Training/Loss', gathered_loss)
            num_step_per_epoch = kwargs.get('num_step_per_epoch', "unknown")
            epoch_id = kwargs.get('epoch_id', -1)
            step_time = kwargs.get('step_time', -1.0)
            print(f"Epoch {epoch_id} Step [{self.num_steps}/{num_step_per_epoch}] Step Time [{step_time:.2f}] - Loss: {gathered_loss:.3f} rank: [{accelerator.process_index}/{accelerator.num_processes}]")

        if save_steps is not None and self.num_steps % save_steps == 0:
            print(f"Saving model at step {self.num_steps}...")
            self.save_model(accelerator, model,  f"step-{self.num_steps}.safetensors")
            self.save_optimizer(accelerator, optimizer, f"optimizer-step-{self.num_steps}.ckpt")
    
    def on_epoch_end(self, accelerator, model, optimizer, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")
        print(f"Epoch {epoch_id} completed - Total steps: {self.num_steps}")
        self.save_optimizer(accelerator, optimizer, f"optimizer-epoch-{epoch_id}.ckpt")

    def on_training_end(self, accelerator, model, optimizer, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.save_optimizer(accelerator, optimizer, f"optimizer-step-{self.num_steps}.ckpt")
        
        # close TensorBoard writer only on the main process
        if accelerator.is_main_process:
            self.tb_writer.close()
            print(f"Training completed. Total steps: {self.num_steps}")
            print(f"TensorBoard logs saved to: {os.path.join(self.output_path, 'tensorboard')}")


  
