#!/bin/bash

# InfiniGram API Test Commands
# Usage: ./test_curl_commands.sh
# Make sure the API server is running on localhost:5000

BASE_URL="http://localhost:5000"
INDEX_NAME="v4_pileval_llama"

echo "=== InfiniGram API Test Commands ==="
echo "Base URL: $BASE_URL"
echo "Index: $INDEX_NAME"
echo ""

# Health Check Endpoints
echo "1. Health Check (new endpoint)"
curl -X GET $BASE_URL/api/health
echo -e "\n"

echo "2. Legacy Health Check"
curl -X GET $BASE_URL/health
echo -e "\n"

# Main Query Endpoints
echo "3. Count Query (Simple)"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown fox"
  }'
echo -e "\n"

echo "4. Count Query (with token IDs)"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "'$INDEX_NAME'",
    "query_ids": [1, 2, 3, 4]
  }'
echo -e "\n"

echo "5. Probability Query"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "prob",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown"
  }'
echo -e "\n"

echo "6. Next Token Distribution (NTD)"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "ntd",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown",
    "max_support": 10
  }'
echo -e "\n"

echo "7. InfiniGram Probability"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "infgram_prob",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown"
  }'
echo -e "\n"

echo "8. InfiniGram Next Token Distribution"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "infgram_ntd",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown",
    "max_support": 10
  }'
echo -e "\n"

echo "9. Search Documents"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "search_docs",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown fox",
    "maxnum": 5,
    "max_disp_len": 1000
  }'
echo -e "\n"

echo "10. Find Query"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "find",
    "index": "'$INDEX_NAME'",
    "query": "the quick brown fox"
  }'
echo -e "\n"

# CNF (Conjunctive Normal Form) Queries
echo "11. Count CNF Query"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "'$INDEX_NAME'",
    "query": "the quick AND brown fox OR lazy dog",
    "max_clause_freq": 1000,
    "max_diff_tokens": 100
  }'
echo -e "\n"

echo "12. Search Documents CNF Query"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "search_docs",
    "index": "'$INDEX_NAME'",
    "query": "the quick AND brown fox OR lazy dog",
    "maxnum": 5,
    "max_disp_len": 1000,
    "max_clause_freq": 1000,
    "max_diff_tokens": 100
  }'
echo -e "\n"

# Legacy Endpoint Testing
echo "13. Legacy Endpoint"
curl -X POST $BASE_URL/ \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "'$INDEX_NAME'",
    "query": "test query"
  }'
echo -e "\n"

# Error Testing
echo "14. Invalid Index"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "invalid_index",
    "query": "test"
  }'
echo -e "\n"

echo "15. Missing Required Fields"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test"
  }'
echo -e "\n"

echo "16. Invalid Query Type"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "invalid_type",
    "index": "'$INDEX_NAME'",
    "query": "test"
  }'
echo -e "\n"

# Advanced Testing
echo "17. Query with All Optional Parameters"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "search_docs",
    "index": "'$INDEX_NAME'",
    "query": "machine learning",
    "max_support": 20,
    "maxnum": 10,
    "max_disp_len": 2000,
    "max_clause_freq": 500,
    "max_diff_tokens": 50
  }'
echo -e "\n"

echo "18. Empty Query Test"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "prob",
    "index": "'$INDEX_NAME'",
    "query": ""
  }'
echo -e "\n"

echo "19. Long Query Test"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "'$INDEX_NAME'",
    "query": "this is a very long query that should test the maximum character limit and see how the API handles longer text inputs"
  }'
echo -e "\n"

echo "20. Special Characters Test"
curl -X POST $BASE_URL/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "count",
    "index": "'$INDEX_NAME'",
    "query": "test with special chars: !@#$%^&*()_+-=[]{}|;:,.<>?"
  }'
echo -e "\n"

echo "=== Test Complete ==="
echo ""
echo "Notes:"
echo "- Default port: 5000"
echo "- Available query types: count, prob, ntd, infgram_prob, infgram_ntd, search_docs, find"
echo "- Parameters: max_support, maxnum, max_disp_len, max_clause_freq, max_diff_tokens"
echo "- All responses include latency, token_ids, tokens, and either result or error"
echo ""
echo "To run the API server:"
echo "python api_server.py --MODE api --FLASK_PORT 5000" 