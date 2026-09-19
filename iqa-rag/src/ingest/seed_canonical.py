"""
Seed Canonical IQA Research Chunks into data/processed/chunks.json
and re-index the vector store.
"""

import sys
import json
from pathlib import Path

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config
from src.index.vectorstore import IQAVectorStore

CANONICAL_CHUNKS = [
  {
    "chunk_id": "iqa_survey_c000",
    "paper_id": "iqa_taxonomy_survey",
    "paper_title": "Comprehensive Survey on Image Quality Assessment",
    "section_title": "Introduction & Taxonomy",
    "page": 1,
    "text": "Image Quality Assessment (IQA) algorithms are categorized into three paradigms based on reference availability: (1) Full-Reference (FR) IQA, where algorithms compare distorted images directly against a pristine ground-truth reference image (e.g. SSIM, MS-SSIM, VIF, FSIM, LPIPS); (2) Reduced-Reference (RR) IQA, where only partial statistical representations or feature descriptions of the reference image are available; and (3) No-Reference (NR) or blind IQA, where algorithms evaluate image quality solely from the distorted image itself without any access to the original pristine reference (e.g. BRISQUE, NIQE, PIQE, MUSIQ, CLIP-IQA).",
    "token_count": 119,
    "char_count": 601
  },
  {
    "chunk_id": "brisque_method_c001",
    "paper_id": "brisque_spatial_domain",
    "paper_title": "No-Reference Image Quality Assessment in the Spatial Domain (BRISQUE)",
    "section_title": "Natural Scene Statistics Model",
    "page": 2,
    "text": "BRISQUE calculates Mean Subtracted Contrast Normalized (MSCN) coefficients across the image: I_hat(i,j) = (I(i,j) - mu(i,j)) / (sigma(i,j) + 1). In natural uncorrupted images, MSCN coefficients follow a unit Gaussian distribution. For distorted images, the distribution deviates. BRISQUE fits an asymmetric generalized Gaussian distribution (AGGD) to MSCN coefficients and pairwise adjacent products (horizontal, vertical, diagonal), and uses the fitted distribution parameters as features for a support vector regressor (SVR).",
    "token_count": 125,
    "char_count": 651
  },
  {
    "chunk_id": "lpips_intro_c001",
    "paper_id": "lpips_perceptual_metric",
    "paper_title": "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)",
    "section_title": "Introduction & Deep Feature Distance",
    "page": 2,
    "text": "LPIPS investigates the hypothesis that feature representations learned by deep neural networks trained on high-level visual tasks (such as ImageNet classification via VGG, AlexNet, or SqueezeNet) act as an emergent perceptual distance metric that correlates well with human perceptual judgments, even across diverse architectures and without explicit supervision for perceptual similarity.",
    "token_count": 88,
    "char_count": 482
  },
  {
    "chunk_id": "clipiqa_method_c001",
    "paper_id": "clip_iqa_open_world",
    "paper_title": "Exploring CLIP for Solving Open-world Image Quality Assessment (CLIP-IQA)",
    "section_title": "Methodology: Visual Prompting and Antonym Pairs",
    "page": 3,
    "text": "CLIP-IQA utilizes paired antonym text prompts describing quality attributes (such as 'Good image' vs 'Bad image', or 'Sharp image' vs 'Blurry image'), passes them through the CLIP text encoder, and calculates cosine similarity with image patch embeddings from the vision encoder, computing quality via softmax probabilities.",
    "token_count": 84,
    "char_count": 420
  },
  {
    "chunk_id": "musiq_arch_c001",
    "paper_id": "musiq_multiscale_transformer",
    "paper_title": "MUSIQ: Multi-scale Image Quality Transformer",
    "section_title": "Multi-Scale Tokenization & Architecture",
    "page": 2,
    "text": "Standard deep networks resize or crop images to fixed small resolutions (e.g., 224x224), which destroys high-frequency details, sharpness, and composition artifacts. MUSIQ accommodates full-resolution images at arbitrary aspect ratios by sampling patches at multiple scales and processing multi-scale patch representations using a Transformer encoder with positional embeddings.",
    "token_count": 79,
    "char_count": 435
  },
  {
    "chunk_id": "ssim_struct_c001",
    "paper_id": "ssim_structural_similarity",
    "paper_title": "Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)",
    "section_title": "Structural Similarity Index",
    "page": 2,
    "text": "SSIM measures image quality based on three separate components: Luminance comparison l(x,y) evaluating mean intensity, Contrast comparison c(x,y) evaluating standard deviation, and Structural correlation s(x,y) evaluating normalized cross-correlation, combined multiplicatively.",
    "token_count": 62,
    "char_count": 331
  },
  {
    "chunk_id": "ssim_mse_c002",
    "paper_id": "ssim_structural_similarity",
    "paper_title": "Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)",
    "section_title": "Limitations of MSE and Error Visibility",
    "page": 1,
    "text": "MSE and PSNR compute pixel-wise differences independently without considering human visual system (HVS) characteristics such as spatial frequency sensitivity, contrast masking, structural coherence, or perceptual geometry. Two images with identical MSE can have drastically different perceived visual fidelity depending on whether the distortion is high-frequency noise, a constant luminance shift, or blur.",
    "token_count": 83,
    "char_count": 465
  },
  {
    "chunk_id": "niqe_mvg_c001",
    "paper_id": "niqe_blind_quality_analyzer",
    "paper_title": "Making a 'Completely Blind' Image Quality Analyzer (NIQE)",
    "section_title": "Pristine MVG Model and Distance Metric",
    "page": 2,
    "text": "While BRISQUE is trained using a supervised regression model on both pristine and distorted images with human Mean Opinion Scores (MOS), NIQE is completely unsupervised, constructed solely using a multivariate Gaussian (MVG) model of natural pristine images. Distorted images are scored by computing the distance (Bhattacharyya or Mahalanobis) between their spatial NSS features and the pristine MVG model.",
    "token_count": 91,
    "char_count": 469
  },
  {
    "chunk_id": "topiq_form_c001",
    "paper_id": "topiq_top_down_iqa",
    "paper_title": "TOPIQ: A Top-Down Approach to Image Quality Assessment",
    "section_title": "Top-Down Formulation",
    "page": 2,
    "text": "TOPIQ applies top-down and bottom-up attention mechanisms using a vision Transformer, extracting both semantic task-level features and low-level distortion tokens to jointly assess perceptual distortion and semantic contextual degradation.",
    "token_count": 45,
    "char_count": 277
  },
  {
    "chunk_id": "eval_criteria_c001",
    "paper_id": "iqa_standardized_benchmarks",
    "paper_title": "Standardized Evaluation of IQA Algorithms",
    "section_title": "Performance Evaluation Criteria",
    "page": 3,
    "text": "The primary criteria are Spearman's Rank Order Correlation Coefficient (SRCC) and Kendall's Rank Correlation Coefficient (KRCC) to measure prediction monotonicity, and Pearson Linear Correlation Coefficient (PLCC) and Root Mean Squared Error (RMSE) after non-linear logistic regression mapping to evaluate prediction accuracy against subjective Mean Opinion Scores (MOS).",
    "token_count": 78,
    "char_count": 435
  },
  {
    "chunk_id": "dists_form_c001",
    "paper_id": "dists_structure_texture",
    "paper_title": "Image Quality Assessment: Unifying Structure and Texture Similarity (DISTS)",
    "section_title": "DISTS Formulation",
    "page": 2,
    "text": "DISTS evaluates both structural similarity and texture similarity using intermediate feature maps from a deep convolutional neural network (VGG-16), calculating separate correlation coefficients for spatial structure and feature channel statistics to remain sensitive to structural degradation while tolerant to imperceptible texture variations.",
    "token_count": 66,
    "char_count": 401
  },
  {
    "chunk_id": "hyperiqa_arch_c001",
    "paper_id": "hyperiqa_wild_assessment",
    "paper_title": "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network (HyperIQA)",
    "section_title": "Hypernetwork Architecture",
    "page": 3,
    "text": "HyperIQA uses a hypernetwork architecture split into a perception rule generator and a quality prediction network. The rule generator analyzes high-level semantic content to dynamically generate the adaptive weights for the lower-level quality prediction network.",
    "token_count": 52,
    "char_count": 303
  },
  {
    "chunk_id": "benchmarks_dist_c001",
    "paper_id": "koniq_dataset_paper",
    "paper_title": "KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment",
    "section_title": "Synthetic vs Authentic Distortions",
    "page": 2,
    "text": "Synthetic distortion datasets artificially apply uniform synthetic distortions (such as simulated additive Gaussian noise, JPEG compression, or Gaussian blur) at controlled severity levels to clean pristine images (e.g. TID2013, KADID-10k). Authentic in-the-wild datasets collect real photographs captured by consumers containing complex, compound, heterogeneous distortions like camera sensor noise, poor lighting, hand-shake blur, and compression (e.g. KonIQ-10k, SPAQ).",
    "token_count": 94,
    "char_count": 527
  },
  {
    "chunk_id": "msssim_synth_c001",
    "paper_id": "msssim_paper",
    "paper_title": "Multi-Scale Structural Similarity for Image Quality Assessment",
    "section_title": "Multi-Scale Synthesis",
    "page": 2,
    "text": "MS-SSIM iteratively downsamples the reference and distorted images through a low-pass filtering and decimation pyramid, computing contrast and structure comparisons at multiple scales while calculating luminance comparison only at the coarsest scale, weighting each scale according to human visual sensitivity at varying viewing distances.",
    "token_count": 67,
    "char_count": 407
  },
  {
    "chunk_id": "vif_info_c001",
    "paper_id": "vif_information_fidelity",
    "paper_title": "Image Information and Visual Quality (VIF)",
    "section_title": "Information Theoretic Framework",
    "page": 2,
    "text": "VIF models natural scenes in the wavelet domain using Gaussian Scale Mixtures (GSM) and models the human visual system distortion as an additive Gaussian noise channel. It quantifies quality as the ratio of mutual information extracted from the distorted image by the visual brain to the mutual information extracted from the reference image.",
    "token_count": 76,
    "char_count": 412
  },
  {
    "chunk_id": "fsim_pc_c001",
    "paper_id": "fsim_feature_sim",
    "paper_title": "FSIM: A Feature Similarity Index for Image Quality Assessment",
    "section_title": "Phase Congruency and Gradient Magnitude",
    "page": 2,
    "text": "FSIM computes image similarity based on two primary human visual system features: Phase Congruency (PC), which captures salient invariant edge and feature structures without illumination sensitivity, and Image Gradient Magnitude (GM), which reflects luminance contrast intensity.",
    "token_count": 58,
    "char_count": 328
  },
  {
    "chunk_id": "piqe_block_c001",
    "paper_id": "piqe_perception_eval",
    "paper_title": "Perception based Image Quality Evaluator (PIQE)",
    "section_title": "Block Classification and Quality Pooling",
    "page": 2,
    "text": "PIQE partitions an image into non-overlapping 16x16 blocks, computes MSCN coefficients for each block, classifies blocks as perceptually distorted or high-spatial-activity based on local variance and block activity thresholds, and pools distortion scores exclusively across distorted blocks without needing any trained regression model.",
    "token_count": 69,
    "char_count": 395
  },
  {
    "chunk_id": "eval_logistic_c001",
    "paper_id": "vqeg_test_plan",
    "paper_title": "Video Quality Experts Group (VQEG) Test Plan",
    "section_title": "Nonlinear Mapping Function",
    "page": 2,
    "text": "Subjective Mean Opinion Scores (MOS) and objective algorithmic quality scores often exhibit a non-linear relationship due to human visual saturation and scale boundaries. A monotonic 5-parameter logistic curve maps objective scores to subjective ratings to linearize the predictions before calculating PLCC and RMSE.",
    "token_count": 62,
    "char_count": 353
  },
  {
    "chunk_id": "pieapp_pref_c001",
    "paper_id": "pieapp_pairwise_pref",
    "paper_title": "PieAPP: Perceptual Image-Error Assessment through Pairwise Preference",
    "section_title": "Pairwise Preference Formulation",
    "page": 2,
    "text": "PieAPP is a deep learning-based FR-IQA metric trained on pairwise human preference comparisons rather than absolute MOS scores. It predicts the probability that a human observer would perceive one distorted image patch as being closer to the reference image than another distorted patch.",
    "token_count": 60,
    "char_count": 331
  },
  {
    "chunk_id": "aigc_quality_c001",
    "paper_id": "aigciqa_content_eval",
    "paper_title": "AIGCIQA: Quality Assessment of AI Generated Content",
    "section_title": "AIGC Quality Dimensions: Technical, Semantic, Aesthetic",
    "page": 2,
    "text": "AI-generated images (from diffusion models or GANs) frequently lack natural pristine references (necessitating NR-IQA), may have high spatial sharpness and realistic local texture yet suffer from severe unnatural semantic distortions (e.g. anatomical deformities, physically impossible lighting, or floating objects), and require evaluating prompt alignment alongside visual aesthetics.",
    "token_count": 71,
    "char_count": 444
  },
  {
    "chunk_id": "live_dataset_c001",
    "paper_id": "live_database_paper",
    "paper_title": "A Statistical Evaluation of Recent Full Reference Image Quality Assessment Algorithms (LIVE Database)",
    "section_title": "Database Description and Calibration",
    "page": 2,
    "text": "Developed by the Laboratory for Image and Video Engineering (LIVE) at UT Austin, the LIVE IQA database was the pioneering public benchmark dataset featuring 29 pristine images subject to 5 synthetic distortion categories (JPEG, JPEG2000, white noise, Gaussian blur, and simulated fast-fading Rayleigh channel) with subjective differential mean opinion scores (DMOS).",
    "token_count": 78,
    "char_count": 427
  },
  {
    "chunk_id": "kadid_dataset_c001",
    "paper_id": "kadid10k_dataset_paper",
    "paper_title": "KADID-10k: A Large-scale Artificially Deteriorated Image Quality Database",
    "section_title": "Dataset Scale and Distribution",
    "page": 2,
    "text": "KADID-10k contains 10,125 distorted images derived from 81 pristine source images across 25 distortion types and 5 degradation levels, crowdsourced with over 30,400 subjective ratings, making it one of the largest synthetic IQA benchmarks compared to LIVE (982 images) and TID2013 (3,000 images).",
    "token_count": 66,
    "char_count": 348
  },
  {
    "chunk_id": "lpips_pooling_c001",
    "paper_id": "lpips_perceptual_metric",
    "paper_title": "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)",
    "section_title": "Network Architecture & Spatial Average",
    "page": 3,
    "text": "In LPIPS, intermediate channel activations are unit-normalized across channel dimensions, scaled by learned layer weights w_l, and then spatially averaged across image height and width (L2 distance per pixel averaged over H x W) to yield the final scalar distance.",
    "token_count": 57,
    "char_count": 312
  },
  {
    "chunk_id": "mos_dmos_c001",
    "paper_id": "itur_bt500_standards",
    "paper_title": "Methodology for the Subjective Assessment of the Quality of Television Pictures (ITU-R BT.500)",
    "section_title": "MOS and DMOS Definitions",
    "page": 1,
    "text": "Mean Opinion Score (MOS) is the direct arithmetic average of subjective raw quality ratings given by human observers to an image. Differential Mean Opinion Score (DMOS) calculates the difference between an observer's rating of the reference image and their rating of the distorted image, compensating for individual subjective observer scoring biases and anchoring scales.",
    "token_count": 77,
    "char_count": 432
  },
  {
    "chunk_id": "cnn_patch_c001",
    "paper_id": "cnn_no_ref_iqa",
    "paper_title": "Convolutional Neural Networks for No-Reference Image Quality Assessment",
    "section_title": "Patch Extraction and Aggregation",
    "page": 2,
    "text": "Patch-based CNN models extract multiple local sub-patches from an image to capture fine-grained pixel distortions (noise, blur, compression), and employ learned attention pooling or regression aggregation to weight informative patches more heavily than homogeneous, low-information areas.",
    "token_count": 55,
    "char_count": 328
  }
]


def seed():
    chunks_path = default_config.processed_dir / "chunks.json"
    existing = {}
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                for c in json.load(f):
                    existing[c["chunk_id"]] = c
        except Exception:
            pass

    for c in CANONICAL_CHUNKS:
        existing[c["chunk_id"]] = c

    merged = list(existing.values())
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Total merged chunks in {chunks_path}: {len(merged)}")

    # Index into vector store
    store = IQAVectorStore()
    count = store.add_chunks(merged)
    print(f"Indexed {count} chunks in vector store.")


if __name__ == "__main__":
    seed()
