import torch
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np

def compute_gmm_confidence(losses):
    """将 loss 分布用 GMM 分成干净和噪声两类"""
    losses_np = losses.cpu().numpy().reshape(-1, 1)
    
    gmm = GaussianMixture(
        n_components=2, 
        random_state=42, 
        max_iter=100,
        tol=1e-4
    )
    gmm.fit(losses_np)
    
    probs = gmm.predict_proba(losses_np)
    clean_idx = np.argmin(gmm.means_.flatten())   # 均值小的成分是干净样本
    
    confidences = torch.tensor(probs[:, clean_idx], dtype=torch.float32)
    return confidences


@torch.no_grad()
def get_all_per_sample_losses(model, loss_fn, dataloader, device):
    """收集全量数据的 per-sample loss，用于 GMM"""
    model.eval()
    all_losses = []
    
    for batch_feat, batch_label, _ in tqdm(dataloader, desc="Collecting losses for GMM"):
        batch_feat = batch_feat.to(device)
        batch_label = batch_label.to(device)
        
        _, per_sample_loss = loss_fn(batch_feat, batch_label, return_per_sample=True)
        all_losses.append(per_sample_loss.cpu())
    
    return torch.cat(all_losses)
