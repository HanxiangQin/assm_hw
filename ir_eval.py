import pickle
import numpy as np
import faiss
from typing import Dict, List, Tuple, Set
import pandas as pd
from collections import defaultdict
import json

class IRMetrics:
    @staticmethod
    def average_precision(retrieved: List, relevant: Set) -> float:
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(retrieved):
            if item in relevant:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(relevant)
    
    @staticmethod
    def reciprocal_rank(retrieved: List, relevant: Set) -> float:
        """Calculate reciprocal rank - 1/rank of first relevant item."""
        if len(relevant) == 0:
            return 0.0
        
        for i, item in enumerate(retrieved):
            if item in relevant:
                return 1.0 / (i + 1.0)
        
        return 0.0  # No relevant item found

class SciFacIndex:    
    def __init__(self, doc_embeddings_path: str, claim_embeddings_path: str):
        self.doc_embeddings_path = doc_embeddings_path
        self.claim_embeddings_path = claim_embeddings_path
        
        # Data structures
        self.doc_embeddings = None
        self.claim_embeddings = None
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
        self.doc_abstracts = {}
        self.claim_texts = {}
        self.claim_ids = []
        
        # FAISS index
        self.index = None
        self.embedding_dim = 1536
        
        # Metrics calculator
        self.metrics = IRMetrics()
        
    def load_embeddings(self):
        with open(self.doc_embeddings_path, "rb") as f:
            raw_doc_embeddings = pickle.load(f)
         
        with open(self.claim_embeddings_path, "rb") as f:
            raw_claim_embeddings = pickle.load(f)

        print(f"Loaded {len(raw_doc_embeddings)} document embeddings")
        print(f"Loaded {len(raw_claim_embeddings)} claim embeddings")

        # count = 0
        # for item in raw_doc_embeddings.items():
        #     print(item[0])
        #     count += 1
        #     if count > 10:
        #         break
        # for item in raw_claim_embeddings.items():
        #     print(item[0])
        #     count += 1
        #     if count > 20:
        #         break
        
        # Process documents
        doc_embedding_matrix = []
        for idx, (doc_key, embedding) in enumerate(raw_doc_embeddings.items()):
            doc_id, abstract = doc_key
            embedding_array = np.array(embedding, dtype=np.float32)
            
            self.doc_id_to_idx[doc_id] = idx
            self.idx_to_doc_id[idx] = doc_id
            self.doc_abstracts[doc_id] = abstract
            doc_embedding_matrix.append(embedding_array)
        
        self.doc_embeddings = np.vstack(doc_embedding_matrix)
        
        # Process claims
        claim_embedding_matrix = []
        for claim_key, embedding in raw_claim_embeddings.items():
            claim_id, claim_text = claim_key
            embedding_array = np.array(embedding, dtype=np.float32)
            
            self.claim_texts[claim_id] = claim_text
            self.claim_ids.append(claim_id)
            claim_embedding_matrix.append(embedding_array)
        
        self.claim_embeddings = np.vstack(claim_embedding_matrix)
        
        
    def build_index(self):
        faiss.normalize_L2(self.doc_embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.doc_embeddings)
        
    def search(self, claim_embedding: np.ndarray, k: int = 100) -> List[Tuple[int, float]]:
        query = claim_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        similarities, indices = self.index.search(query, k)
        
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx != -1:  # Valid index
                doc_id = self.idx_to_doc_id[idx]
                results.append((doc_id, float(sim)))
        
        return results
    
    def load_scifact_ground_truth(self, claims_file: str = "dataset/claims_train.jsonl") -> Dict[int, Set[int]]:
        validation_claims = []
        with open(claims_file, 'r', encoding='utf-8') as f:
            for line in f:
                validation_claims.append(json.loads(line.strip()))
        
        print(f"Loaded {len(validation_claims)} claims with evidence")
        
        ground_truth = {}
        claims_with_evidence = 0
        total_evidence_docs = 0
        
        for claim in validation_claims:
            claim_id = claim['id']
            cited_doc_ids = set()
            
            if claim.get('evidence'):
                for doc_id_str, evidence_list in claim['evidence'].items():
                    doc_id = int(doc_id_str)
                    for evidence in evidence_list:
                        if evidence.get('label') in ['SUPPORT', 'CONTRADICT']:
                            cited_doc_ids.add(doc_id)
                            break
            
            if claim.get('cited_doc_ids'):
                cited_doc_ids.update(claim['cited_doc_ids'])
            
            if cited_doc_ids:
                ground_truth[claim_id] = cited_doc_ids
                claims_with_evidence += 1
                total_evidence_docs += len(cited_doc_ids)
        
        return ground_truth
    
    def filter_ground_truth(self, ground_truth: Dict[int, Set[int]]) -> None:
        claims_in_embeddings = set(self.claim_ids)
        docs_in_gt = set()
        for doc_set in ground_truth.values():
            docs_in_gt.update(doc_set)
        docs_in_embeddings = set(self.doc_abstracts.keys())
        
        missing_docs = docs_in_gt - docs_in_embeddings
        if missing_docs:
            print(f"Missing docs (first 5): {list(missing_docs)[:5]}")
        
        filtered_gt = {}
        for claim_id, doc_ids in ground_truth.items():
            if claim_id in claims_in_embeddings:
                available_docs = doc_ids.intersection(docs_in_embeddings)
                if available_docs:
                    filtered_gt[claim_id] = available_docs
        
        return filtered_gt
    
    def evaluate_comprehensive(self, ground_truth: Dict[int, Set[int]], 
                             k_values: List[int] = [1, 10, 50]) -> Dict[str, float]:
        all_results = {}
        retrieval_results = {}
        for i, claim_id in enumerate(self.claim_ids):
            claim_embedding = self.claim_embeddings[i]
            retrieved = self.search(claim_embedding, k=max(k_values))
            retrieved_ids = [doc_id for doc_id, _ in retrieved]
            retrieval_results[claim_id] = retrieved_ids
        
        for k in k_values:
            map_scores = []
            mrr_scores = []
            
            for claim_id in self.claim_ids:
                if claim_id in ground_truth:
                    retrieved = retrieval_results[claim_id]
                    relevant = ground_truth[claim_id]
                    
                    map_scores.append(self.metrics.average_precision(retrieved[:k], relevant))
                    mrr_scores.append(self.metrics.reciprocal_rank(retrieved[:k], relevant))
            
            all_results[f'MAP@{k}'] = np.mean(map_scores)
            all_results[f'MRR@{k}'] = np.mean(mrr_scores)
        
        return all_results


def main():
    sci_fac_index = SciFacIndex(
        doc_embeddings_path="embeddings/scifact_evidence_embeddings.pkl",
        claim_embeddings_path="embeddings/scifact_claim_embeddings.pkl"
    )
    
    sci_fac_index.load_embeddings()
    sci_fac_index.build_index()
    
    ground_truth = sci_fac_index.load_scifact_ground_truth()
    filtered_ground_truth = sci_fac_index.filter_ground_truth(ground_truth)
    
    results = sci_fac_index.evaluate_comprehensive(filtered_ground_truth)
    
    for metric, value in results.items():
        print(f"{metric:10s}: {value:.4f}")

if __name__ == "__main__":
    main()
