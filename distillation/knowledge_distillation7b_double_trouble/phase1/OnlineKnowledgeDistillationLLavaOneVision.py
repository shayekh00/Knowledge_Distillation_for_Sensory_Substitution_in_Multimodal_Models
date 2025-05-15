import pytorch_lightning as pl
import torch
from transformers import LlavaOnevisionForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F
import logging as log
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from LLavaOneVisionModule import LlavaOnevisionModule
print("Current directory in KD :", os.getcwd())

class OnlineKnowledgeDistillationLLavaOneVision(pl.LightningModule):
    def __init__(self, model_name_student, model_name_teacher, processor, learning_rate=1e-5, phase=1):
        super().__init__()

        self.phase = phase
        # self.model_name_student = model_name_student
        self.learning_rate = learning_rate   

        # Load the student and teacher models
        self.processor = processor

        # self.student_model = student_model
        # self.teacher_model = teacher_model
        print("Phase:", self.phase)

        print("Loading Teacher Model.....")

        self.teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name_teacher, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # checkpoint_path = "XXXXXX/checkpoints/baseline_rgb/llava-onevision0.5b-epoch=00-val_loss=0.0129.ckpt"
        # self.teacher_model = LlavaOnevisionModule.load_from_checkpoint(
        #         checkpoint_path,
        #         low_cpu_mem_usage=True,
        #         model_name=model_name_teacher,
        #         processor=processor,
        #         torch_dtype=torch.float16,
        #         device_map="auto"
        #     )
        # self.teacher_model = self.teacher_model.model
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
        # self.print_non_language_layers()

        # self.freeze_student_language_layers()


        # self.student_model
        


        self.config = self.student_model.config
        
        self.pad_token_id = (self.processor.tokenizer.eos_token_id 
                             if self.processor.tokenizer.pad_token_id is None 
                             else self.processor.tokenizer.pad_token_id)
        
        #For loss computation
        # self.mse_loss_fn = nn.MSELoss()
        self.soft_target_loss_weight = 0.1
        self.ce_loss_weight = 0.5

        self.gamma= 0.8
        self.T = 0.8
        

        # Placeholders for hooks
        self.student_representation_output = None
        self.teacher_representation_output = None

        self.layer_idx = 0
        self._register_hooks(self.layer_idx)


        # self.freeze_middle_n_layers_teacher(7)



    def setup(self, stage=None):
        # Teacher set to evaluation mode during setup
        self.student_model.train()
        self.teacher_model.eval()

        # self.student_model.gradient_checkpointing_enable()
        # self.teacher_model.gradient_checkpointing_enable()


            


    def _register_hooks(self, layer_idx):
        # # Adjust the layer index for both student and teacher models
        # layer_student = self.student_model.language_model.model.layers[layer_idx].self_attn.q_proj
        # layer_teacher = self.teacher_model.language_model.model.layers[layer_idx].self_attn.q_proj

        # # Register forward hooks
        # layer_student.register_forward_hook(self.hook_fn_student)
        # layer_teacher.register_forward_hook(self.hook_fn_teacher)

        """Registers hooks on the last layer of the vision encoder."""
        teacher_layer = self.teacher_model.vision_tower.vision_model.post_layernorm
        student_layer = self.student_model.vision_tower.vision_model.post_layernorm

        teacher_layer.register_forward_hook(self.hook_fn_teacher)
        student_layer.register_forward_hook(self.hook_fn_student)


    def hook_fn_student(self, module, input, output):
        self.student_representation_output = output

    def hook_fn_teacher(self, module, input, output):
        self.teacher_representation_output = output
    
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
        self.log('val_loss', loss,on_step=False ,on_epoch=True, prog_bar=True, logger=True)

        return loss


    def compute_loca_loss(self, teacher_logits, student_logits, student_loss, labels, alpha=0.8):
        """
        Compute Logit Calibration (LoCa) loss for Knowledge Distillation.

        Args:
            teacher_logits (torch.Tensor): Logits from the teacher model.
            student_logits (torch.Tensor): Logits from the student model.
            labels (torch.Tensor): Ground truth labels.
            alpha (float): Scaling factor for LoCa adjustment (0 < alpha < 1).

        Returns:
            torch.Tensor: LoCa knowledge distillation loss.
        """
        # Align teacher logits to student logits (handle different number of classes)
        teacher_logits = teacher_logits[:, :, :student_logits.size(2)]

        # Convert logits to probabilities
        teacher_probs = nn.functional.softmax(teacher_logits / self.T, dim=-1)
        student_probs = nn.functional.softmax(student_logits / self.T, dim=-1)

        eps = 1e-8  # a small value
        safe_student_probs = torch.clamp(student_probs, min=eps)  # Avoid log(0)


        # Identify the target probabilities (correct class probabilities)
        target_probs_teacher = teacher_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # p_gt
        target_probs_student = student_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Identify the most confident non-target class (mis-instruction risk)
        _, teacher_klogits = teacher_probs.topk(2, dim=-1)  # Get top 2 classes
        teacher_klogits = teacher_klogits[:, :, 1]  # Second most confident class (mis-instruction risk)

        # Extract probabilities of the second-highest class
        non_target_probs_teacher = teacher_probs.gather(2, teacher_klogits.unsqueeze(-1)).squeeze(-1)  # p_klogits

        # Compute threshold σ
        sigma = 1 / (1 - target_probs_teacher + non_target_probs_teacher)

        # Compute scaling factor s
        s = alpha * sigma

        # Calibrate logits
        loca_teacher_probs = teacher_probs.clone()
        loca_teacher_probs[:, :, labels] = 1 - s * (teacher_probs.sum(dim=-1) - target_probs_teacher)  # Ensure sum=1
        loca_teacher_probs[:, :, teacher_klogits] = s * non_target_probs_teacher  # Scale non-target logits

        # Compute KL divergence loss between calibrated teacher logits and student logits
        loca_loss = F.kl_div(
            safe_student_probs.log(),
            loca_teacher_probs,
            reduction='mean',
        ) * (self.T ** 2)

        return loca_loss + student_loss



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    


    # working forward function
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
        student_logits = student_outputs.logits
        student_loss = student_outputs.loss

        # Convert feature maps to embeddings
        teacher_features = self.teacher_representation_output.mean(dim=1)  # [B, 1152]
        student_features = self.student_representation_output.mean(dim=1)  # [B, 1152]

        # Normalize embeddings
        teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        student_features = F.normalize(student_features, p=2, dim=-1)

        if self.phase == 1:
            total_loss = self.compute_vision_loss(student_features, teacher_features, teacher_logits, student_outputs)

        if self.phase == 2:
            total_loss = self.compute_loca_loss(teacher_logits, student_logits, student_loss, labels)  
            # self.compute_language_loss(teacher_logits, student_outputs)

        if self.phase == 3:
            loca_loss = self.compute_loca_loss(teacher_logits, student_logits, student_loss, labels) 
            # print("LoCa Loss:", loca_loss, "student_loss :", student_loss)
            total_loss = (self.gamma) * loca_loss + (1-self.gamma) * student_loss
            # + self.contrastive_loss(student_features, teacher_features) 

            # total_loss = self.compute_combined_loss(student_features, teacher_features, teacher_logits, student_outputs, loca_loss)

        # # Compute contrastive loss
        # contrastive_loss_value = self.contrastive_loss(student_features, teacher_features)
        # total_loss = contrastive_loss_value

        # total_loss = self.compute_language_loss(teacher_logits, student_outputs)

        return total_loss
    

    def compute_combined_loss(self,student_features, teacher_features, teacher_logits, student_outputs, loca_loss):
            """
            Computes the total loss for knowledge distillation, including:
            - Contrastive loss between teacher and student embeddings
            - KL Divergence loss between the soft teacher and student distributions
            - Cross-Entropy loss of student labels
            
            Args:
                teacher_logits (torch.Tensor): Logits output by the teacher model.
                student_outputs: The outputs from the student model, containing logits and loss.
            
            Returns:
                torch.Tensor: The total computed loss.
            """
            # #Resizing teacher logits for it to allign with the student
            # student_logits = student_outputs.logits
            # teacher_logits = teacher_logits[:, :, :student_logits.size(2)]

            # soft_teacher_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
            # soft_student_probs = nn.functional.log_softmax(student_logits / self.T, dim=-1)


            # # kl_divergence_loss = torch.sum(soft_teacher_targets * (soft_teacher_targets.log() - soft_student_probs)) / soft_student_probs.size(0) * (self.T ** 2)
            # kl_divergence_loss = F.kl_div(
            #                     soft_student_probs, 
            #                     soft_teacher_targets, 
            #                     reduction='mean',
            #                     log_target = True
            #                 ) * (self.T ** 2)
            
            contrastive_loss_value = self.contrastive_loss(student_features, teacher_features)

            # # # Cross-Entropy loss from student outputs
            # student_label_loss = student_outputs.loss

            # Total loss combining all components
            loss = loca_loss  + contrastive_loss_value


            return loss
    

    def compute_vision_loss(self,student_features, teacher_features, teacher_logits, student_outputs):
            """
            Computes the total loss for knowledge distillation, including:
            - Contrastive loss between teacher and student embeddings
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

            soft_teacher_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
            soft_student_probs = nn.functional.log_softmax(student_logits / self.T, dim=-1)


            # kl_divergence_loss = torch.sum(soft_teacher_targets * (soft_teacher_targets.log() - soft_student_probs)) / soft_student_probs.size(0) * (self.T ** 2)
            kl_divergence_loss = F.kl_div(
                                soft_student_probs, 
                                soft_teacher_targets, 
                                reduction='mean'
                            ) * (self.T ** 2)
            
            contrastive_loss_value = self.contrastive_loss(student_features, teacher_features)

            # # Cross-Entropy loss from student outputs
            # student_label_loss = student_outputs.loss

            # Total loss combining all components
            loss = (self.soft_target_loss_weight * kl_divergence_loss) + (self.ce_loss_weight * contrastive_loss_value) 


            return loss
    
    def compute_language_loss(self, teacher_logits, student_outputs):
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
    

    def contrastive_loss(self, student_features, teacher_features, temperature=0.07):
        """
        Compute contrastive loss (NT-Xent Loss) between teacher and student embeddings.

        Args:
            student_features (torch.Tensor): Student model vision features.
            teacher_features (torch.Tensor): Teacher model vision features.
            temperature (float): Temperature scaling factor.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        # Normalize features
        student_features = F.normalize(student_features, p=2, dim=-1)
        teacher_features = F.normalize(teacher_features, p=2, dim=-1)

        # Compute cosine similarity
        logits = torch.mm(student_features, teacher_features.T) / temperature

        # NT-Xent loss
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss
    

    def print_non_language_layers(self):
        # print("\n===== Teacher Model (Excluding Language Model) =====")
        # for name, module in self.teacher_model.named_modules():
        #     if "language_model" not in name:  # Exclude language model layers
        #         print(name, "->", module)

        print("\n===== Student Model (Excluding Language Model) =====")
        for name, module in self.student_model.named_modules():
            if "language_model" not in name:  # Exclude language model layers
                print(name, "->", module)




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

    def freeze_student_language_layers(self):
        """
        Freezes all the language model layers of the student model.
        """
        if hasattr(self.student_model, "language_model"):
            for param in self.student_model.language_model.model.parameters():
                param.requires_grad = False
            print("Student Model - All language layers are frozen.")

        # Log which layers are frozen
        frozen_layers = [
            idx for idx, layer in enumerate(self.student_model.language_model.model.layers)
            if not any(param.requires_grad for param in layer.parameters())
        ]
        print(f"Student Model - Frozen language layers: {frozen_layers}")


    
    def unfreeze_student_language_layers(self):
        """
        Freezes all the language model layers of the student model except the vision encoder.
        """
        if hasattr(self.student_model, "language_model"):
            for param in self.student_model.language_model.model.parameters():
                param.requires_grad = True
            print("Student Model - All language layers are unfrozen.")

        # Unfreeze vision encoder
        if hasattr(self.student_model, "vision_model"):
            for param in self.student_model.vision_model.parameters():
                param.requires_grad = True
            print("Student Model - Vision encoder layers are trainable.")

    def freeze_student_vision_layers(self):
        """
        Freezes all the vision model layers of the student model.
        """
        if hasattr(self.student_model, "vision_tower"):
            for param in self.student_model.vision_tower.parameters():
                param.requires_grad = False
            print("Student Model - All vision layers are frozen.")

        # # Log which layers are frozen
        # frozen_layers = [
        #     name for name, param in self.student_model.vision_tower.named_parameters() if not param.requires_grad
        # ]
        # print(f"Student Model - Frozen vision layers: {frozen_layers}")

    def unfreeze_student_vision_layers(self):
        """
        Freezes all the vision model layers of the student model.
        """
        if hasattr(self.student_model, "vision_tower"):
            for param in self.student_model.vision_tower.parameters():
                param.requires_grad = True
            print("Student Model - All vision layers are frozen.")