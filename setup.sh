#!/bin/bash
mkdir -p ~/.streamlit/

echo "\
[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml

# Run Streamlit with the port explicitly passed
streamlit run analysis_flow.py --server.port=$PORT
