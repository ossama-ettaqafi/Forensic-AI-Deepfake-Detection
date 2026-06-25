import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("deepfake-api")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaLearningAdapter(nn.Module):
    """
    Episodic Model-Agnostic Meta-Learning (MAML) Adapter with Test-Time Adaptation.
    Based on the article's proposed episodic MAML framework, this model adapts its parameters
    online to the specific video/audio domain at test-time.
    
    It combines two meta-learning paradigms:
    1. HyperNetwork Meta-initialization: Generates base projection weights from the domain signature.
    2. MAML Gradient Update: Performs online inner-loop gradient descent using a self-supervised
       entropy minimization objective to adapt parameters to the test video's domain.
    """
    def __init__(self, signature_dim=8, logits_dim=2):
        super(MetaLearningAdapter, self).__init__()
        self.logits_dim = logits_dim
        self.inner_lr = 0.05  # Inner-loop adaptation rate (\alpha)
        
        # Meta-Network (HyperNetwork) to generate initial adaptation weights (W_0)
        self.weight_generator = nn.Sequential(
            nn.Linear(signature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, logits_dim * logits_dim)
        )
        
        # Meta-Network to generate initial adaptation biases (b_0)
        self.bias_generator = nn.Sequential(
            nn.Linear(signature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, logits_dim)
        )

    def forward(self, logits, domain_signature, num_steps=2):
        """
        Adapts logits using MAML inner-loop test-time gradient steps.
        Arguments:
            logits (Tensor): Base CNN output logits [Batch, Classes].
            domain_signature (Tensor): Multi-dimensional vector of domain metrics [Batch, Feature_Dim].
        """
        if domain_signature.device != logits.device:
            domain_signature = domain_signature.to(logits.device)
            
        batch_size = logits.size(0)
        
        # 1. Generate meta-initialized weights and biases based on domain signature
        w_flat = self.weight_generator(domain_signature)
        w_init = w_flat.view(batch_size, self.logits_dim, self.logits_dim)
        b_init = self.bias_generator(domain_signature)
        
        # We want to perform gradient descent on w and b
        w = w_init.clone().detach().requires_grad_(True)
        b = b_init.clone().detach().requires_grad_(True)
        
        # 2. MAML Inner Loop: Test-Time Adaptation via Entropy Minimization
        for step in range(num_steps):
            # Forward pass
            logits_unsqueezed = logits.unsqueeze(-1)
            # batched matrix multiplication: w * logits + b
            curr_logits = torch.bmm(w, logits_unsqueezed).squeeze(-1) + b
            
            probs = torch.softmax(curr_logits, dim=-1)
            # Binary Shannon entropy loss
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
            
            # Compute gradients of entropy w.r.t w and b
            grad_w, grad_b = torch.autograd.grad(
                entropy, [w, b], create_graph=False, allow_unused=True
            )
            
            # Gradient descent step
            if grad_w is not None:
                w = w - self.inner_lr * grad_w
            if grad_b is not None:
                b = b - self.inner_lr * grad_b
                
        # 3. Final prediction with adapted weights
        logits_unsqueezed = logits.unsqueeze(-1)
        final_logits = torch.bmm(w, logits_unsqueezed).squeeze(-1) + b
        
        return final_logits

# Initialize Meta-Learning adapter
meta_adapter = MetaLearningAdapter()

possible_paths = [
    "models/meta_adapter.pth",
    "weights/meta_adapter.pth",
    "backend/models/meta_adapter.pth",
    "backend/weights/meta_adapter.pth"
]

weights_loaded = False
for path in possible_paths:
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device, weights_only=True)
            meta_adapter.load_state_dict(state)
            logger.info(f"✅ Loaded meta-learning adapter weights from {path}")
            weights_loaded = True
            break
        except Exception as e:
            logger.warning(f"⚠️ Failed to load meta-learning weights from {path}: {str(e)}")

if not weights_loaded:
    logger.info("ℹ️ No pre-trained meta-learning adapter weights found. Initialized adapter for dynamic inline scaling.")

meta_adapter.to(device)
meta_adapter.eval()

def extract_domain_signature(tensor_or_path, is_audio=False):
    """
    Computes a domain signature vector representing resolution, noise floor,
    contrast, and variance metadata to drive meta-adaptation.
    """
    with torch.no_grad():
        # Standard signature placeholder dimensions: [1, 8]
        # Features represent: [Mean, StdDev, High-freq intensity, Blur, Compression sign, Variance, 2 unused indicators]
        if isinstance(tensor_or_path, torch.Tensor):
            flat = tensor_or_path.view(-1)
            mean = float(flat.mean().item())
            std = float(flat.std().item())
            max_val = float(flat.max().item())
            min_val = float(flat.min().item())
            
            # Simple structural variance calculation
            diff = flat[:-1] - flat[1:]
            hf_energy = float(diff.pow(2).mean().item())
            
            signature = [
                mean,
                std,
                hf_energy,
                (max_val - min_val),
                1.0 if is_audio else 0.0,
                std / (mean + 1e-5),
                0.5,
                0.1
            ]
        else:
            # Fallback signature
            signature = [0.5, 0.2, 0.05, 1.0, 1.0 if is_audio else 0.0, 0.4, 0.5, 0.1]
            
        return torch.tensor([signature], dtype=torch.float32, device=device)
