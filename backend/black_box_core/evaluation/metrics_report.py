import os
import numpy as np
import faiss
import pickle
import json


# ==========================================================
# ======================= UTILITIES ========================
# ==========================================================

def load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    paths = metadata.get("paths")
    labels = metadata.get("labels")

    if paths is None or labels is None:
        raise ValueError("Metadata must contain 'paths' and 'labels'.")

    return paths, labels


def load_embeddings(embeddings_path):
    print("📦 Loading embeddings...")
    data = np.load(embeddings_path, allow_pickle=True)

    if "embeddings" not in data.files:
        raise ValueError("No 'embeddings' key found in embeddings file.")

    embeddings = data["embeddings"].astype("float32")

    print(f"✅ Total embeddings loaded: {len(embeddings)}")
    return embeddings


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return embeddings / norms


# ==========================================================
# ==================== CORE EVALUATION =====================
# ==========================================================

def evaluate_retrieval(
    index_path,
    embeddings_path,
    metadata_path,
    similarity="COSINE",
    top_k=5,
    num_eval_queries=10000,
    random_seed=42
):
    print("🔄 Loading FAISS index...")
    index = faiss.read_index(index_path)

    embeddings = load_embeddings(embeddings_path)
    print("📦 Loading metadata...")
    _, labels = load_metadata(metadata_path)

    if len(embeddings) != len(labels):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) != Labels count ({len(labels)})"
        )

    if similarity.upper() == "COSINE":
        print("🔁 Applying cosine normalization...")
        embeddings = normalize_embeddings(embeddings)

    n_total = len(embeddings)

    # ------------------------------------------------------
    # RANDOM QUERY SAMPLING
    # ------------------------------------------------------
    if num_eval_queries > n_total:
        num_eval_queries = n_total

    print(f"\n🎯 Evaluating on {num_eval_queries} RANDOM queries (out of {n_total})")

    np.random.seed(random_seed)
    query_indices = np.random.choice(
        n_total,
        size=num_eval_queries,
        replace=False
    )

    query_embeddings = embeddings[query_indices]

    print("⚡ Performing batch FAISS search...")
    D, I = index.search(query_embeddings, top_k + 1)

    top1_correct = 0
    top5_correct = 0
    reciprocal_rank_sum = 0
    total_gap = 0.0
    total_top2_score = 0.0

    labels = np.array(labels)
    identity_mode = len(set(labels)) < len(labels)

    for i, original_idx in enumerate(query_indices):

        retrieved_indices = I[i]
        scores = D[i]

        # Remove self-match
        mask = retrieved_indices != original_idx
        filtered_indices = retrieved_indices[mask][:top_k]
        filtered_scores = scores[mask][:top_k]

        true_label = labels[original_idx]

        if identity_mode:
            matches = (labels[filtered_indices] == true_label)

            if np.any(matches):
                first_match_rank = np.argmax(matches)

                if first_match_rank == 0:
                    top1_correct += 1

                if first_match_rank < top_k:
                    top5_correct += 1

                reciprocal_rank_sum += 1 / (first_match_rank + 1)

        else:
            if len(filtered_indices) > 0 and filtered_indices[0] == original_idx:
                top1_correct += 1

        # Score statistics
        if len(filtered_scores) > 1:
            total_top2_score += filtered_scores[1]
            total_gap += filtered_scores[0] - filtered_scores[1]

    results = {}

    if identity_mode:
        results["top1_accuracy"] = top1_correct / num_eval_queries
        results["top5_accuracy"] = top5_correct / num_eval_queries
        results["mrr"] = reciprocal_rank_sum / num_eval_queries
    else:
        results["self_top1_accuracy"] = top1_correct / num_eval_queries

    results["avg_top2_score"] = total_top2_score / num_eval_queries
    results["avg_similarity_gap"] = total_gap / num_eval_queries

    return results, num_eval_queries


# ==========================================================
# ===================== REPORT SAVING ======================
# ==========================================================

def save_results(
    results,
    model_name,
    dataset_name,
    similarity,
    total_samples,
    top_k=5
):
    metric_dir = os.path.join("outputs", "reports", "metric_reports")
    os.makedirs(metric_dir, exist_ok=True)

    filename = f"{model_name.lower()}_{dataset_name.lower()}_{similarity.lower()}_metrics.json"
    file_path = os.path.join(metric_dir, filename)

    # Skip if file already exists
    if os.path.exists(file_path):
        print(f"\n⏩ Metrics already exist at: {file_path}")
        print("Skipping evaluation save.")
        return file_path

    report = {
        "model": model_name,
        "dataset": dataset_name,
        "similarity": similarity,
        "top_k": top_k,
        "evaluated_queries": total_samples,
        "metrics": results
    }

    with open(file_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n📂 Metrics saved to: {file_path}")
    return file_path
