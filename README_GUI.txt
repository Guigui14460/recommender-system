chmod a+x cgi-bin/index.py
chmod a+x cgi-bin/handler.py
python3 -m http.server --cgi
http://localhost:8000/cgi-bin/index.py