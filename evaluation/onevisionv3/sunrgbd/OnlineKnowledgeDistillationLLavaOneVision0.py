import pytorch_lightning as pl
import torch
from transformers import LlavaOnevisionForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F
import logging as log
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from LLavaOneVisionModule import LlavaOnevisionModule
print("Current directory in KD :", os.getcwd())

class OnlineKnowledgeDistillationLLavaOneVision(pl.LightningModule):
    def __init__(self, model_name_student, model_name_teacher, processor, learning_rate=2e-5):
        super().__init__()
        # self.model_name_student = model_name_student
        self.learning_rate = learning_rate   

        # Load the student and teacher models
        self.processor = processor

        # self.student_model = student_model
        # self.teacher_model = teacher_model

        print("Loading Teacher Model.....")

        # self.teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        #     model_name_teacher, 
        #     low_cpu_mem_usage=True,
        # )

        checkpoint_path = "XXXX/checkpoints/baseline_rgb/llava-onevision7b-epoch=03-val_loss=0.0071.ckpt"
        self.teacher_model = LlavaOnevisionModule.load_from_checkpoint(
                checkpoint_path,
                low_cpu_mem_usage=True,
                model_name=model_name_teacher,
                processor=processor,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        self.teacher_model = self.teacher_model.model
        self.teacher_model.eval()
        
        # Freeze all parameters of the teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print("Teacher Model Loaded")


        print("Loading Student Model.....")
        self.student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name_student, 
            low_cpu_mem_usage=True,
        )
        print("Student Model Loaded")


        self.student_model

        self.config = self.student_model.config
        
        self.pad_token_id = (self.processor.tokenizer.eos_token_id 
                             if self.processor.tokenizer.pad_token_id is None 
                             else self.processor.tokenizer.pad_token_id)
        
        #For loss computation
        # self.mse_loss_fn = nn.MSELoss()
        self.soft_target_loss_weight = 0.5
        self.ce_loss_weight = 0.5
        self.T = 2.0
        

        # # Placeholders for hooks
        # self.student_representation_output = None
        # self.teacher_representation_output = None

        # self.layer_idx = 0
        # self._register_hooks(self.layer_idx)


        # self.freeze_middle_n_layers_teacher(7)



    def setup(self, stage=None):
        # Teacher set to evaluation mode during setup
        self.student_model.train()
        self.teacher_model.eval()

        # self.student_model.gradient_checkpointing_enable()
        # self.teacher_model.gradient_checkpointing_enable()


            


    def _register_hooks(self, layer_idx):
        # Adjust the layer index for both student and teacher models
        layer_student = self.student_model.language_model.model.layers[layer_idx].self_attn.q_proj
        layer_teacher = self.teacher_model.language_model.model.layers[layer_idx].self_attn.q_proj




        # Register forward hooks
        layer_student.register_forward_hook(self.hook_fn_student)
        layer_teacher.register_forward_hook(self.hook_fn_teacher)

    def hook_fn_student(self, module, input, output):
        self.student_representation_output = output

    def hook_fn_teacher(self, module, input, output):
        self.teacher_representation_output = output


    def kl_divergence_loss(self,soft_student_probs, soft_teacher_targets, temperature):
        loss = F.kl_div(
            soft_student_probs.log(),
            soft_teacher_targets,
            reduction='batchmean'
        ) * (temperature ** 2)
        return loss
    


    def mse_loss(self, input, target):
        input = input.to(torch.float32)
        target = target.to(torch.float32)

        # Calculate MSE
        return torch.mean((input - target) ** 2)
    
    def compute_loss(self, teacher_logits, student_outputs):
            """
            Computes the total loss for knowledge distillation, including:
            - Mean Squared Error (MSE) loss between the student and teacher representations
            - KL Divergence loss between the soft teacher and student distributions
            - Cross-Entropy loss of student labels
            
            Args:
                teacher_logits (torch.Tensor): Logits output by the teacher model.
                student_outputs: The outputs from the student model, containing logits and loss.
            
            Returns:
                torch.Tensor: The total computed loss.
            """
            #Resizing teacher logits for it to allign with the student
            student_logits = student_outputs.logits
            teacher_logits = teacher_logits[:, :, :student_logits.size(2)]

            # MSE loss between student and teacher representations            
            # mse_loss = self.mse_loss(self.student_representation_output, self.teacher_representation_output)

            # Soft targets from teacher and softmax log probabilities from student


            
            soft_teacher_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
            soft_student_probs = nn.functional.log_softmax(student_logits / self.T, dim=-1)

            # KL Divergence loss
            # print("soft_teacher_targets.shape:", soft_teacher_targets.shape)
            # print("soft_teacher_targets.logits.shape:", soft_teacher_targets.shape)
            # print("soft_student_probs.shape:", soft_student_probs.shape)

            kl_divergence_loss = torch.sum(soft_teacher_targets * (soft_teacher_targets.log() - soft_student_probs)) / soft_student_probs.size(0) * (self.T ** 2)

            # Cross-Entropy loss from student outputs
            student_label_loss = student_outputs.loss

            # Total loss combining all components
            loss = (self.soft_target_loss_weight * kl_divergence_loss) + (self.ce_loss_weight * student_label_loss) 


            return loss

    def forward(self, batch):

        rgb_pixel_values = batch["rgb_pixel_values"]
        rgb_input_ids = batch["rgb_input_ids"]

        depth_input_ids = batch["depth_input_ids"]
        depth_pixel_values = batch["depth_pixel_values"]

        labels = batch["labels"]
        image_sizes = batch["image_sizes"]
        # question_id = batch["question_id"]


        # self.teacher_representation_output = teacher_rep
        # self.teacher_logits = teacher_logits

        # Teacher model forward pass
        rgb_inputs = {
            'input_ids': rgb_input_ids,
            'pixel_values': rgb_pixel_values,
            'labels': labels,  
            'image_sizes': image_sizes
        }
        teacher_outputs = self.teacher_model(**rgb_inputs)
        teacher_logits = teacher_outputs.logits

        # Student model forward pass
        depth_inputs = {
            'input_ids': depth_input_ids,
            'pixel_values': depth_pixel_values,
            'labels': labels,  
            'image_sizes': image_sizes
        }
        student_outputs = self.student_model(**depth_inputs)

        total_loss = self.compute_loss(teacher_logits, student_outputs)

        return total_loss
    

    def freeze_middle_n_layers_teacher(self, n):
        """
        Freeze `n` layers in the middle of the teacher model's language model.

        Args:
            n (int): Number of layers to freeze from the middle.
        """
        total_layers = len(self.teacher_model.language_model.model.layers)

        # Calculate the start and end indices for the middle layers to freeze
        start_idx = (total_layers - n) // 2
        end_idx = start_idx + n

        # Freeze all layers of the teacher model initially
        for param in self.teacher_model.language_model.model.parameters():
            param.requires_grad = False

        # Unfreeze layers outside the middle range (before start_idx and after end_idx)
        for idx, layer in enumerate(self.teacher_model.language_model.model.layers):
            if idx < start_idx or idx >= end_idx:
                for param in layer.parameters():
                    param.requires_grad = True

        # Log frozen and trainable layers for verification
        frozen_layers = [
            idx for idx, layer in enumerate(self.teacher_model.language_model.model.layers)
            if not any(param.requires_grad for param in layer.parameters())
        ]
        trainable_layers = [
            idx for idx, layer in enumerate(self.teacher_model.language_model.model.layers)
            if any(param.requires_grad for param in layer.parameters())
        ]
        print(f"Teacher Model - Frozen layers: {frozen_layers}")
        print(f"Teacher Model - Trainable layers: {trainable_layers}")



    def training_step(self, batch, batch_idx):
        # Forward pass
        loss = self(batch)
        
        # batch_size = batch["rgb_input_ids"].size(0)
        # Log the validation loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        loss = self(batch)
        
        # batch_size = batch["rgb_input_ids"].size(0)
        # Log the validation loss
        # self.log("val_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss,on_step=False ,on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    