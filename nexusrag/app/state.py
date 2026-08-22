def query_rag(self, query_text: str, top_k: int = 5):
        """
        Executes RAG pipeline safely by checking retriever & index state validity.
        """
        if not query_text or not query_text.strip():
            return "Please provide a valid question."

        try:
            # Ensure state targets valid pipeline reference
            if not hasattr(self, "pipeline") or self.pipeline is None:
                if hasattr(self, "init_pipeline"):
                    self.init_pipeline()
                else:
                    return "RAG Pipeline is not initialized properly."

            # Safe execution guard
            top_k_val = max(1, int(top_k)) if top_k else 5
            
            # Execute Pipeline directly with exception catching
            response = self.pipeline.run(
                query=query_text.strip(),
                top_k=top_k_val
            )
            return response

        except Exception as e:
            # Prevent Streamlit Redacted ValueError by logging raw exception internally
            import logging
            logging.error(f"Error executing query_rag in state.py: {str(e)}", exc_info=True)
            return f"An issue occurred while processing your request: {str(e)}"