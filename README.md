# Compression of Vision Transformer by Reduction of Kernel Complexity

Code for Compression of Vision Transformer by Reduction of Kernel Complexity



## Abstract

Self-attention and transformer architectures have become foundational components in modern deep learning. Recent efforts have integrated transformer blocks into compact neural architectures for computer vision, giving rise to various efficient vision transformers. In this work, we introduce Transformer with Kernel Complexity Reduction, or KCR-Transformer, a compact transformer block equipped with differentiable channel selection, guided by a novel and sharp theoretical generalization bound. To reduce the substantial computational cost of the MLP layers, the KCR-Transformer performs channel selection on the outputs of its self-attention layer.
Furthermore, we provide a rigorous theoretical analysis establishing a tight generalization bound for networks equipped with KCR-Transformer blocks. Leveraging such strong theoretical results, the channel pruning by KCR-Transformer is conducted in a generalization-aware manner, ensuring that the resulting network retains a provably small generalization error. 
Our KCR-Transformer is compatible with many popular and compact transformer networks, such as ViT and Swin, and it reduces the FLOPs of the vision transformers while maintaining or even improving the prediction accuracy. In the experiments, we replace all the transformer blocks in the vision transformers with KCR-Transformer blocks, leading to KCR-Transformer networks with different backbones. The resulting KCR-Transformers achieve superior performance on various computer vision tasks, achieving even better performance than the original models with even less FLOPs and parameters.

## Demo

Environments

```
torch>=1.7
torchvision
pyyaml
huggingface_hub
safetensors>=0.2
numpy
```

Demo Run

``````
./distributed_train_cv.sh 4 /data_dir/imagenet/ --model KCR-vit-b \
-b 64 --workers 8 --pin-mem \
--sched cosine --epochs 300 --min-lr 0.00001 --lr 0.0001 --warmup-lr 0.0001 \
--opt adamw --weight-decay 0.01 \
--model-ema --model-ema-decay 0.9995 \
--aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp \
--smoothing 0.1 \
--output /output_dir \
--checkpoint-hist 20 \
--valsubset-ratio 0.1 \
``````

