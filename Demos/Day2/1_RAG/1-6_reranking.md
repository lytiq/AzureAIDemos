```
# pip install cohere

import cohere

co = cohere.ClientV2(api_key="COHERE_API_KEY") # Get your free API key: https://dashboard.cohere.com/api-keys

# Define the documents
faqs_short = [
    {"text": "Reimbursing Travel Expenses: Easily manage your travel expenses by submitting them through our finance tool. Approvals are prompt and straightforward."},
    {"text": "Working from Abroad: Working remotely from another country is possible. Simply coordinate with your manager and ensure your availability during core hours."},
    {"text": "Health and Wellness Benefits: We care about your well-being and offer gym memberships, on-site yoga classes, and comprehensive health insurance."},
    {"text": "Performance Reviews Frequency: We conduct informal check-ins every quarter and formal performance reviews twice a year."}
]

# Add the user query
query = "Are there fitness-related perks?"

# Rerank the documents
results = co.rerank(query=query,
                    documents=faqs_short,
                    top_n=2,
                    model='rerank-english-v3.0')

print(results)

# Display the reranking results
def return_results(results, documents):    
    for idx, result in enumerate(results.results):
        print(f"Rank: {idx+1}") 
        print(f"Score: {result.relevance_score}")
        print(f"Document: {documents[result.index]}\n")
    
return_results(results, faqs_short)

# semi-structured data and unstructured data : https://docs.cohere.com/v2/docs/reranking-with-cohere
```

also see : https://github.com/AnswerDotAI/RAGatouille/blob/main/examples/04-reranking.ipynb