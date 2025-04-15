#!/bin/bash
mkdir -p ~/.streamlit/

echo "\
[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml

# Now actually run your Streamlit app
streamlit run analysis_flow.py
