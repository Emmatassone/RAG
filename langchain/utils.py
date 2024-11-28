from pdf_ingestor import PDFIngestor

pdf_ingestor = PDFIngestor()
pdf_ingestor.load_or_ingest("./traceability_chromadb")

def get_context(query):
    docs = pdf_ingestor.retrieve_context(query)
    return "\n\n".join(doc.page_content for doc in docs)