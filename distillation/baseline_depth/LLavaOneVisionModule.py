import pytorch_lightning as pl
import torch
from transformers import LlavaOnevisionForConditionalGeneration


class LlavaOnevisionModule(pl.LightningModule):
    def __init__(self, model_name, processor, learning_rate=2e-5):
        super().__init__()
        self.model_name = model_name
        

        self.learning_rate = learning_rate
        
        self.processor = processor
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name, 
            low_cpu_mem_usage=True,
        )
        self.model.train()

        self.config = self.model.config
        
        self.pad_token_id = (self.processor.tokenizer.eos_token_id 
                             if self.processor.tokenizer.pad_token_id is None 
                             else self.processor.tokenizer.pad_token_id)


    def freeze_all_except_last_n(self, n):
        """
        Freeze all layers except the last `n` layers of the vision encoder 
        and language model decoder.

        Args:
            n (int): Number of last layers to keep trainable.
        """
        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last `n` layers of the Vision Transformer
        total_vision_layers = len(self.model.vision_tower.vision_model.encoder.layers)
        for name, param in self.model.vision_tower.vision_model.encoder.layers.named_parameters():
            layer_number = int(name.split(".")[0])  # Extract layer index
            if layer_number >= (total_vision_layers - n):  # Unfreeze last `n` layers
                param.requires_grad = True

        # Unfreeze the last `n` layers of the Language Model Decoder
        total_decoder_layers = len(self.model.language_model.model.layers)
        for name, param in self.model.language_model.model.layers.named_parameters():
            layer_number = int(name.split(".")[0])  # Extract layer index
            if layer_number >= (total_decoder_layers - n):  # Unfreeze last `n` layers
                param.requires_grad = True

        # Log trainable layers for verification
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters: {trainable_params}")


    def make_vision_fully_trainable(self):
        """
        Unfreeze all layers in the vision tower (vision transformer) so they are fully trainable.
        """
        # Unfreeze all parameters in the Vision Transformer
        for name, param in self.model.vision_tower.named_parameters():
            param.requires_grad = True

        # Log trainable layers for verification
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters (Vision Fully Trainable): {trainable_params}")


    def forward(self, input_ids, rgb_pixel_values, depth_pixel_values, labels , image_sizes):

        inputs = {
            'input_ids': input_ids,
            'pixel_values': depth_pixel_values,
            'labels': labels,  
            'image_sizes': image_sizes
        }

        outputs = self.model(**inputs)

        return outputs


    def training_step(self, batch, batch_idx):
        # Unpack the batch dictionary
        input_ids = batch["depth_input_ids"]
        rgb_pixel_values = batch["rgb_pixel_values"]
        depth_pixel_values = batch["depth_pixel_values"]
        labels = batch["labels"]
        image_sizes = batch["image_sizes"]
        # Forward pass
        outputs = self(input_ids=input_ids, rgb_pixel_values=rgb_pixel_values,depth_pixel_values=depth_pixel_values, labels=labels , image_sizes=image_sizes)
        
        # Calculate loss
        loss = outputs.loss
        
        batch_size = batch["depth_input_ids"].size(0)
        # Log the validation loss
        # self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)    
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log training loss
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch dictionary
        input_ids = batch["depth_input_ids"]
        rgb_pixel_values = batch["rgb_pixel_values"]
        depth_pixel_values = batch["depth_pixel_values"]
        labels = batch["labels"]
        image_sizes = batch["image_sizes"]

        
        # Forward pass
        outputs = self(input_ids=input_ids, rgb_pixel_values=rgb_pixel_values, depth_pixel_values=depth_pixel_values, labels=labels , image_sizes=image_sizes)
        


        # Calculate validation loss
        loss = outputs.loss
        
        batch_size = batch["depth_input_ids"].size(0)
        # Log the validation loss
        # self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # # Log validation loss
        # self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)