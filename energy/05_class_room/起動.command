#!/bin/zsh
cd "$(dirname "$0")"
TEACHING_PYTHON="/Users/fyu/Dropbox/Documents/classes/2026_Waseda/mini_dt_teaching/env/bin/python"
if [[ -x "$TEACHING_PYTHON" ]]; then
  "$TEACHING_PYTHON" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
else
  python3 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
fi
