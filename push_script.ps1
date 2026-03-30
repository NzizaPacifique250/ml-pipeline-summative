git add src/
git commit -m "feat: implement ML core pipeline scripts"
git push

Start-Sleep -Seconds 2

git add api.py requirements.txt
git commit -m "feat: build FastAPI backend"
git push

Start-Sleep -Seconds 2

git add ui.py
git commit -m "feat: build Streamlit dashboard"
git push

Start-Sleep -Seconds 2

git add Dockerfile* docker-compose.yml
git commit -m "chore: add docker deployment configuration"
git push

Start-Sleep -Seconds 2

git add locustfile.py
git commit -m "test: add locust load test simulation"
git push

Start-Sleep -Seconds 2

git add README.md notebook/
git commit -m "docs: update README with project instructions"
git push

Start-Sleep -Seconds 2

git add .
git commit -m "chore: final cleanups"
git push
