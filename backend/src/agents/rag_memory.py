import os
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CustomsRAGMemory")

class CustomsMemoryRAG:
    def __init__(self, db_path: str = "./qdrant_customs_db"):
        """
        Initializes the Qdrant Vector Database and the Gemini Embedding Client.
        We use a local persistent database for easy deployment.
        """
        self.collection_name = "historical_customs_fraud"
        
        # Using active text embedding model with dimension truncation config
        self.embedding_model = "gemini-embedding-001"
        self.vector_size = 768  # Target size using Matryoshka Representation Learning
        
        # Initialize Gemini Client (Falls back to environment variable safely)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyB8JWkZ64ypSoPkJ2tBVyc2gz7oV5yTRTM")
        self.ai_client = genai.Client(api_key=self.gemini_api_key)
        
        # Initialize Qdrant Client (Persistent local storage)
        self.qdrant_client = QdrantClient(path=db_path)
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection to store case memory if not already present."""
        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            logger.info(f"🏗️ Creating new Qdrant memory collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Converts text into a mathematical vector using Gemini with dimension compression."""
        try:
            response = self.ai_client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.vector_size)
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {str(e)}")
            return [0.0] * self.vector_size

    def archive_resolved_case(self, file_id: int, exporter: str, verdict_data: dict, risk_tier: str):
        """
        Takes a finalized Chief Judge verdict and saves it into long-term memory.
        Call this function AFTER a transaction has been completely evaluated.
        """
        # Create a rich semantic string for the AI to embed
        searchable_text = (
            f"Exporter: {exporter}. "
            f"Risk Tier: {risk_tier}. "
            f"Hidden Patterns: {' '.join(verdict_data.get('hidden_patterns_uncovered', []))}. "
            f"Business Impact: {verdict_data.get('business_impact_assessment', '')}"
        )
        
        vector = self._get_embedding(searchable_text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_id))) # Deterministic UUID based on file_id
        
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "file_id": file_id,
                "exporter": exporter,
                "risk_tier": risk_tier,
                "historical_verdict": verdict_data
            }
        )
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        logger.info(f"💾 Case {file_id} successfully archived in Qdrant memory.")

    def retrieve_historical_context(self, current_tx: dict, top_k: int = 2) -> str:
        """
        Searches Qdrant for similar past cases based on the incoming transaction parameters.
        Returns a formatted text string to inject into the Chief Judge's prompt.
        """
        search_query = (
            f"Exporter: {current_tx.get('emetteur', 'Unknown')}. "
            f"Declared Value: {current_tx.get('valeur_fob', 0.0)} CFA."
        )
        
        query_vector = self._get_embedding(search_query)
        
        # Upgraded to modern query_points API endpoint 
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=0.75 # Only return highly relevant matches
        ).points
        
        if not search_results:
            return "No highly similar historical fraud patterns found for this entity or profile."
            
        context_blocks = []
        for hit in search_results:
            payload = hit.payload
            match_score = hit.score * 100
            exporter = payload.get('exporter', 'Unknown')
            tier = payload.get('risk_tier', 'Unknown')
            past_impact = payload.get('historical_verdict', {}).get('business_impact_assessment', 'No summary.')
            
            block = (
                f"- [Similarity: {match_score:.1f}%] Past Case for {exporter}:\n"
                f"  Resulted in {tier}. Past findings: {past_impact}"
            )
            context_blocks.append(block)
            
        return "\n\n".join(context_blocks)

# ======================================================================
# Quick Test Execution Block
# ======================================================================
if __name__ == "__main__":
    memory = CustomsMemoryRAG()
    print("🚀 Qdrant RAG Engine Initialized Successfully!")