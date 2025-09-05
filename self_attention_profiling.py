import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import psutil
import os
from typing import List, Tuple, Dict
from collections import defaultdict
import pandas as pd
from scipy import stats

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(attention_output)
        
        return output

class AttentionProfiler:
    def __init__(self, d_model: int = 512, num_heads: int = 8, batch_size: int = 32):
        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size

    def count_self_attention_flops(self, batch_size: int, seq_len: int, d_model: int, num_heads: int) -> int:
        d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V (3 operations)
        linear_flops = 3 * batch_size * seq_len * d_model * d_model
        
        # Q @ K^T computation
        qk_flops = batch_size * num_heads * seq_len * seq_len * d_k
        
        # Softmax (approximated as 5 ops per element)
        softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
        
        # Attention @ V computation
        av_flops = batch_size * num_heads * seq_len * seq_len * d_k
        
        # Final linear transformation
        final_linear_flops = batch_size * seq_len * d_model * d_model
        
        total_flops = linear_flops + qk_flops + softmax_flops + av_flops + final_linear_flops
        return total_flops
        
    def profile_attention(self, seq_lengths: List[int], num_trials: int = 10, 
                         device: str = 'cpu') -> Dict:
        results = defaultdict(list)
        
        for seq_len in seq_lengths:
            print(f"Profiling sequence length: {seq_len}")
            
            # Create model and move to device
            model = SelfAttention(self.d_model, self.num_heads).to(device)
            model.eval()
            
            # Theoretical FLOPS
            theoretical_flops = self.count_self_attention_flops(
                self.batch_size, seq_len, self.d_model, self.num_heads
            )
            
            trial_times = []
            trial_memory = []
            
            for trial in range(num_trials):
                # Create input tensor
                x = torch.randn(self.batch_size, seq_len, self.d_model, device=device)
                
                # Clear cache and collect garbage
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                memory_before = torch.cuda.memory_allocated() / 1024 / 1024
                
                if device == 'cuda':
                    with torch.no_grad():
                        _ = model(x)
                    torch.cuda.synchronize()
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = model(x)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024
                memory_used = memory_after - memory_before
                
                trial_times.append(end_time - start_time)
                trial_memory.append(memory_used)
                
                del x, output
                
            results['seq_len'].append(seq_len)
            results['device'].append(device)
            results['theoretical_flops'].append(theoretical_flops)
            results['mean_time'].append(np.mean(trial_times))
            results['std_time'].append(np.std(trial_times))
            results['se_time'].append(stats.sem(trial_times))
            results['mean_memory'].append(np.mean(trial_memory))
            results['std_memory'].append(np.std(trial_memory))
            results['se_memory'].append(stats.sem(trial_memory))
            
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        return dict(results)
    
    def create_plots(self, results_gpu: Dict = None):
        df_gpu = pd.DataFrame(results_gpu)
        df_gpu['device'] = 'GPU'

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Self-Attention Profiling Results', fontsize=16, fontweight='bold')
        
        # 1. Computational Complexity (FLOPS)
        ax1 = axes[0]
        for device in df_gpu['device'].unique():
            device_data = df_gpu[df_gpu['device'] == device]
            ax1.loglog(device_data['seq_len'], device_data['theoretical_flops'], 
                      'o-', label=f'{device}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Theoretical FLOPs')
        ax1.set_title('Computational Complexity (FLOPs)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Wall Clock Time
        ax2 = axes[1]
        for device in df_gpu['device'].unique():
            device_data = df_gpu[df_gpu['device'] == device]
            ax2.errorbar(device_data['seq_len'], device_data['mean_time'], 
                        yerr=device_data['se_time'], 
                        fmt='o-', label=f'{device}', linewidth=2, markersize=6,
                        capsize=5, capthick=2)
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Wall Clock Time (seconds)')
        ax2.set_title('Execution Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Usage
        ax3 = axes[2]
        for device in df_gpu['device'].unique():
            device_data = df_gpu[df_gpu['device'] == device]
            ax3.errorbar(device_data['seq_len'], device_data['mean_memory'], 
                        yerr=device_data['se_memory'], 
                        fmt='o-', label=f'{device}', linewidth=2, markersize=6,
                        capsize=5, capthick=2)
        
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('self_attention_profiling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_gpu

def main():
    seq_lengths = [10, 100, 1000, 10000]
    num_trials = 20
    d_model = 512
    num_heads = 16
    batch_size = 1
    
    profiler = AttentionProfiler(d_model, num_heads, batch_size)
    
    results_gpu = None
    if torch.cuda.is_available():
        print("\nProfiling on GPU...")
        results_gpu = profiler.profile_attention(seq_lengths, num_trials, device='cuda')
    
    df_results = profiler.create_plots(results_gpu)
    df_results.to_csv('self_attention_profiling_results.csv', index=False)

if __name__ == "__main__":
    main()