# Leaderboard
- UPDATED: 2024-08-09
- Number of evaluation datasets: 7
- Datasets: [Atlas](#atlas), [DeepFashion-InShop](#deepfashion-inshop), [DeepFashion-Multimodal](#deepfashion-multimodal), [Fashion200k](#fashion200k), [iMaterialist](#imaterialist), [KAGL](#kagl), [Polyvore](#polyvore)
## Average Results
### Text-To-Image (Averaged across 6 datasets)
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.163       | 0.077      | 0.249       | 0.165     |
| Marqo-FashionCLIP          | 0.192       | 0.094      | 0.290       | 0.200     |
| Marqo-FashionSigLIP        | **0.231**   | **0.121**  | **0.340**   | **0.239** |
| OpenFashionCLIP            | 0.132       | 0.060      | 0.204       | 0.135     |
| ViT-B-16-laion2b_s34b_b88k | 0.174       | 0.088      | 0.261       | 0.180     |
| ViT-B-16-SigLIP-webli      | 0.212       | 0.111      | 0.314       | 0.214     |
### Category-To-Product (Averaged across 5 datasets)
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.684     | 0.681     | 0.686     | 0.741     |
| Marqo-FashionCLIP          | 0.705     | 0.734     | 0.676     | 0.776     |
| Marqo-FashionSigLIP        | **0.737** | **0.758** | **0.716** | **0.812** |
| OpenFashionCLIP            | 0.646     | 0.653     | 0.639     | 0.720     |
| ViT-B-16-laion2b_s34b_b88k | 0.662     | 0.673     | 0.652     | 0.743     |
| ViT-B-16-SigLIP-webli      | 0.688     | 0.690     | 0.685     | 0.751     |
### Sub-Category-To-Product (Averaged across 4 datasets)
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.657     | 0.676     | 0.638     | 0.733     |
| Marqo-FashionCLIP          | 0.707     | 0.747     | 0.667     | 0.772     |
| Marqo-FashionSigLIP        | **0.725** | **0.767** | **0.683** | **0.811** |
| OpenFashionCLIP            | 0.598     | 0.619     | 0.578     | 0.689     |
| ViT-B-16-laion2b_s34b_b88k | 0.638     | 0.651     | 0.624     | 0.712     |
| ViT-B-16-SigLIP-webli      | 0.643     | 0.643     | 0.643     | 0.726     |
## Atlas
### Text-To-Image
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.135       | 0.055      | 0.214       | 0.140     |
| Marqo-FashionCLIP          | 0.155       | 0.074      | 0.236       | 0.172     |
| Marqo-FashionSigLIP        | **0.211**   | **0.109**  | **0.313**   | **0.244** |
| OpenFashionCLIP            | 0.113       | 0.048      | 0.178       | 0.116     |
| ViT-B-16-laion2b_s34b_b88k | 0.145       | 0.068      | 0.222       | 0.160     |
| ViT-B-16-SigLIP-webli      | 0.200       | 0.097      | 0.304       | 0.229     |
### Sub-Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.621     | 0.618     | 0.624     | 0.678     |
| Marqo-FashionCLIP          | 0.668     | **0.706** | 0.629     | 0.725     |
| Marqo-FashionSigLIP        | **0.687** | **0.706** | **0.668** | **0.744** |
| OpenFashionCLIP            | 0.603     | 0.647     | 0.559     | 0.668     |
| ViT-B-16-laion2b_s34b_b88k | 0.591     | 0.588     | 0.594     | 0.637     |
| ViT-B-16-SigLIP-webli      | 0.654     | 0.676     | 0.632     | 0.715     |
## DeepFashion-InShop
### Text-To-Image
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.107       | 0.039      | 0.175       | 0.259     |
| Marqo-FashionCLIP          | 0.144       | 0.048      | 0.239       | **0.331** |
| Marqo-FashionSigLIP        | **0.149**   | **0.050**  | **0.248**   | 0.330     |
| OpenFashionCLIP            | 0.085       | 0.030      | 0.140       | 0.221     |
| ViT-B-16-laion2b_s34b_b88k | 0.112       | 0.039      | 0.185       | 0.267     |
| ViT-B-16-SigLIP-webli      | 0.111       | 0.036      | 0.186       | 0.255     |
### Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.744     | 0.750     | 0.738     | 0.781     |
| Marqo-FashionCLIP          | 0.662     | 0.625     | 0.700     | 0.722     |
| Marqo-FashionSigLIP        | **0.800** | **0.812** | **0.787** | **0.865** |
| OpenFashionCLIP            | 0.697     | 0.750     | 0.644     | 0.794     |
| ViT-B-16-laion2b_s34b_b88k | 0.697     | 0.688     | 0.706     | 0.766     |
| ViT-B-16-SigLIP-webli      | 0.697     | 0.688     | 0.706     | 0.757     |
### Sub-Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.787     | **0.875** | 0.700     | **0.875** |
| Marqo-FashionCLIP          | **0.819** | **0.875** | **0.762** | **0.875** |
| Marqo-FashionSigLIP        | 0.806     | **0.875** | 0.738     | **0.875** |
| OpenFashionCLIP            | 0.731     | 0.750     | 0.713     | 0.768     |
| ViT-B-16-laion2b_s34b_b88k | 0.738     | 0.750     | 0.725     | 0.797     |
| ViT-B-16-SigLIP-webli      | 0.650     | 0.625     | 0.675     | 0.742     |
### Color-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.023     | 0.021     | 0.024     | 0.041     |
| Marqo-FashionCLIP          | 0.032     | 0.035     | 0.029     | 0.055     |
| Marqo-FashionSigLIP        | **0.038** | **0.041** | **0.034** | **0.066** |
| OpenFashionCLIP            | 0.021     | 0.020     | 0.022     | 0.041     |
| ViT-B-16-laion2b_s34b_b88k | 0.027     | 0.031     | 0.023     | 0.050     |
| ViT-B-16-SigLIP-webli      | 0.033     | 0.036     | 0.031     | 0.062     |
## DeepFashion-Multimodal
### Text-To-Image
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP                | 0.014       | 0.003      | 0.026       | 0.008     |
| Marqo-FashionCLIP          | 0.014       | 0.004      | 0.024       | 0.009     |
| Marqo-FashionSigLIP        | 0.019       | **0.007**  | 0.031       | **0.015** |
| OpenFashionCLIP            | 0.010       | 0.002      | 0.018       | 0.006     |
| ViT-B-16-laion2b_s34b_b88k | 0.012       | 0.005      | 0.019       | 0.008     |
| ViT-B-16-SigLIP-webli      | **0.020**   | **0.007**  | **0.033**   | 0.013     |
### Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.675     | 0.688     | 0.662     | 0.729     |
| Marqo-FashionCLIP          | 0.747     | **0.812** | 0.681     | 0.812     |
| Marqo-FashionSigLIP        | **0.762** | **0.812** | **0.713** | **0.859** |
| OpenFashionCLIP            | 0.641     | 0.625     | 0.656     | 0.742     |
| ViT-B-16-laion2b_s34b_b88k | 0.637     | 0.625     | 0.650     | 0.731     |
| ViT-B-16-SigLIP-webli      | 0.681     | 0.688     | 0.675     | 0.773     |
## Fashion200k
### Text-To-Image
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.139       | 0.057      | 0.221       | 0.103     |
| Marqo-FashionCLIP          | 0.173       | 0.066      | 0.280       | 0.125     |
| Marqo-FashionSigLIP        | **0.244**   | **0.111**  | **0.378**   | **0.187** |
| OpenFashionCLIP            | 0.115       | 0.047      | 0.184       | 0.086     |
| ViT-B-16-laion2b_s34b_b88k | 0.146       | 0.062      | 0.230       | 0.108     |
| ViT-B-16-SigLIP-webli      | 0.204       | 0.091      | 0.317       | 0.155     |
### Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.870     | 0.800     | 0.940     | 0.900     |
| Marqo-FashionCLIP          | **1.000** | **1.000** | **1.000** | **1.000** |
| Marqo-FashionSigLIP        | 0.960     | **1.000** | 0.920     | **1.000** |
| OpenFashionCLIP            | 0.990     | **1.000** | 0.980     | **1.000** |
| ViT-B-16-laion2b_s34b_b88k | 0.990     | **1.000** | 0.980     | **1.000** |
| ViT-B-16-SigLIP-webli      | 0.850     | 0.800     | 0.900     | 0.850     |
### Sub-Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.635     | 0.613     | 0.658     | 0.753     |
| Marqo-FashionCLIP          | 0.703     | 0.742     | **0.665** | 0.797     |
| Marqo-FashionSigLIP        | **0.708** | **0.774** | 0.642     | **0.868** |
| OpenFashionCLIP            | 0.581     | 0.613     | 0.548     | 0.755     |
| ViT-B-16-laion2b_s34b_b88k | 0.624     | 0.645     | 0.603     | 0.750     |
| ViT-B-16-SigLIP-webli      | 0.600     | 0.581     | 0.619     | 0.716     |
### Fine-Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.051     | 0.070     | 0.033     | 0.107     |
| Marqo-FashionCLIP          | 0.065     | 0.087     | 0.042     | 0.132     |
| Marqo-FashionSigLIP        | **0.104** | **0.147** | **0.060** | **0.203** |
| OpenFashionCLIP            | 0.077     | 0.108     | 0.047     | 0.160     |
| ViT-B-16-laion2b_s34b_b88k | 0.059     | 0.081     | 0.037     | 0.119     |
| ViT-B-16-SigLIP-webli      | 0.095     | 0.135     | 0.055     | 0.190     |
## iMaterialist
### Category-To-Image
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.510     | 0.543     | 0.477     | 0.664     |
| Marqo-FashionCLIP          | 0.539     | 0.552     | 0.525     | 0.693     |
| Marqo-FashionSigLIP        | **0.572** | **0.590** | **0.553** | **0.722** |
| OpenFashionCLIP            | 0.525     | 0.571     | 0.479     | 0.707     |
| ViT-B-16-laion2b_s34b_b88k | 0.518     | 0.543     | 0.493     | 0.668     |
| ViT-B-16-SigLIP-webli      | 0.490     | 0.486     | 0.495     | 0.657     |
### Style-To-Image
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.348     | 0.381     | 0.314     | 0.501     |
| Marqo-FashionCLIP          | 0.336     | 0.333     | 0.338     | 0.499     |
| Marqo-FashionSigLIP        | 0.393     | **0.429** | 0.357     | 0.528     |
| OpenFashionCLIP            | **0.424** | **0.429** | **0.419** | **0.569** |
| ViT-B-16-laion2b_s34b_b88k | 0.300     | 0.286     | 0.314     | 0.450     |
| ViT-B-16-SigLIP-webli      | 0.312     | 0.286     | 0.338     | 0.481     |
### Neckline-To-Image
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.595     | 0.636     | 0.555     | 0.708     |
| Marqo-FashionCLIP          | 0.555     | 0.545     | 0.564     | 0.667     |
| Marqo-FashionSigLIP        | 0.423     | 0.364     | 0.482     | 0.458     |
| OpenFashionCLIP            | 0.623     | 0.636     | **0.609** | 0.727     |
| ViT-B-16-laion2b_s34b_b88k | 0.591     | 0.636     | 0.545     | 0.700     |
| ViT-B-16-SigLIP-webli      | **0.627** | **0.727** | 0.527     | **0.788** |
## KAGL
### Text-To-Image
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.263       | 0.122      | 0.404       | 0.217     |
| Marqo-FashionCLIP          | 0.303       | 0.159      | 0.446       | 0.265     |
| Marqo-FashionSigLIP        | **0.334**   | **0.179**  | **0.489**   | **0.293** |
| OpenFashionCLIP            | 0.208       | 0.091      | 0.325       | 0.170     |
| ViT-B-16-laion2b_s34b_b88k | 0.295       | 0.155      | 0.435       | 0.259     |
| ViT-B-16-SigLIP-webli      | 0.319       | 0.169      | 0.468       | 0.272     |
### Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.700     | **0.714** | 0.686     | **0.714** |
| Marqo-FashionCLIP          | 0.643     | **0.714** | 0.571     | **0.714** |
| Marqo-FashionSigLIP        | 0.629     | 0.571     | 0.686     | 0.643     |
| OpenFashionCLIP            | 0.493     | 0.429     | 0.557     | 0.500     |
| ViT-B-16-laion2b_s34b_b88k | 0.550     | 0.571     | 0.529     | 0.619     |
| ViT-B-16-SigLIP-webli      | **0.707** | **0.714** | **0.700** | **0.714** |
### Sub-Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.586     | 0.600     | 0.571     | 0.627     |
| Marqo-FashionCLIP          | 0.640     | 0.667     | 0.613     | 0.691     |
| Marqo-FashionSigLIP        | **0.699** | **0.711** | **0.687** | **0.756** |
| OpenFashionCLIP            | 0.479     | 0.467     | 0.491     | 0.566     |
| ViT-B-16-laion2b_s34b_b88k | 0.598     | 0.622     | 0.573     | 0.664     |
| ViT-B-16-SigLIP-webli      | 0.667     | 0.689     | 0.644     | 0.731     |
### Fine-Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.605     | 0.634     | 0.577     | 0.694     |
| Marqo-FashionCLIP          | **0.669** | **0.725** | 0.613     | **0.772** |
| Marqo-FashionSigLIP        | 0.667     | 0.704     | **0.629** | 0.763     |
| OpenFashionCLIP            | 0.515     | 0.542     | 0.489     | 0.607     |
| ViT-B-16-laion2b_s34b_b88k | 0.628     | 0.683     | 0.573     | 0.748     |
| ViT-B-16-SigLIP-webli      | 0.640     | 0.669     | 0.611     | 0.736     |
### Color-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.428     | 0.413     | 0.443     | 0.518     |
| Marqo-FashionCLIP          | 0.498     | 0.500     | 0.496     | 0.614     |
| Marqo-FashionSigLIP        | **0.545** | **0.587** | **0.502** | **0.664** |
| OpenFashionCLIP            | 0.472     | 0.478     | 0.465     | 0.587     |
| ViT-B-16-laion2b_s34b_b88k | 0.487     | 0.500     | 0.474     | 0.605     |
| ViT-B-16-SigLIP-webli      | 0.459     | 0.500     | 0.417     | 0.564     |
### Season-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.362     | **0.500** | 0.225     | 0.550     |
| Marqo-FashionCLIP          | 0.338     | 0.250     | **0.425** | 0.542     |
| Marqo-FashionSigLIP        | **0.425** | **0.500** | 0.350     | 0.500     |
| OpenFashionCLIP            | 0.400     | **0.500** | 0.300     | **0.583** |
| ViT-B-16-laion2b_s34b_b88k | 0.300     | 0.250     | 0.350     | 0.490     |
| ViT-B-16-SigLIP-webli      | 0.325     | 0.250     | 0.400     | 0.375     |
### Usage-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.244     | 0.250     | 0.237     | 0.275     |
| Marqo-FashionCLIP          | 0.256     | 0.250     | 0.263     | 0.354     |
| Marqo-FashionSigLIP        | 0.388     | 0.375     | **0.400** | 0.438     |
| OpenFashionCLIP            | 0.188     | 0.125     | 0.250     | 0.225     |
| ViT-B-16-laion2b_s34b_b88k | 0.263     | 0.250     | 0.275     | 0.292     |
| ViT-B-16-SigLIP-webli      | **0.444** | **0.500** | 0.388     | **0.500** |
## Polyvore
### Text-To-Image
| Model                      | AvgRecall   | Recall@1   | Recall@10   | MRR       |
|----------------------------|-------------|------------|-------------|-----------|
| FashionCLIP2.0                | 0.319       | 0.186      | 0.452       | 0.261     |
| Marqo-FashionCLIP          | 0.362       | 0.212      | 0.512       | 0.299     |
| Marqo-FashionSigLIP        | **0.427**   | **0.271**  | **0.585**   | **0.367** |
| OpenFashionCLIP            | 0.262       | 0.143      | 0.381       | 0.209     |
| ViT-B-16-laion2b_s34b_b88k | 0.335       | 0.198      | 0.472       | 0.277     |
| ViT-B-16-SigLIP-webli      | 0.421       | 0.268      | 0.574       | 0.359     |
### Category-To-Product
| Model                      | AvgP      | P@1       | P@10      | MRR       |
|----------------------------|-----------|-----------|-----------|-----------|
| FashionCLIP2.0                | 0.429     | 0.454     | 0.405     | 0.581     |
| Marqo-FashionCLIP          | 0.473     | 0.517     | 0.429     | 0.629     |
| Marqo-FashionSigLIP        | **0.534** | **0.594** | **0.474** | **0.695** |
| OpenFashionCLIP            | 0.410     | 0.462     | 0.358     | 0.565     |
| ViT-B-16-laion2b_s34b_b88k | 0.437     | 0.480     | 0.395     | 0.597     |
| ViT-B-16-SigLIP-webli      | 0.502     | 0.560     | 0.445     | 0.663     |
