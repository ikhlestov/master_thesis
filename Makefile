profile:
	kernprof -l 03_python_cilynder_gh.py; python -m line_profiler 03_python_cilynder_gh.py.lprof >> profiled.txt

upload_all:
	rsync -avh \
	    --include=.dockerignore \
	    --exclude=".*" \
	    --exclude=.ipynb_checkpoints \
	    --exclude=.git \
	    --exclude=.venv \
	    --exclude=.pytest_cache \
	    --exclude=__pycache__ \
	    --exclude=logs \
	    --exclude=htmlcov \
		--exclude=frames* \
		--exclude=videos* \
	    --exclude=*.sublime-project \
	    --exclude=*.sublime-workspace \
	    . illarion@213.32.26.74:~/Projects/master_degree

download_frames:
	# rsync -avh illarion@213.32.26.74:~/Projects/master_degree/frames/ frames
	# rsync -avh illarion@213.32.26.74:~/Projects/master_degree/frames_CYLINDER/ frames_CYLINDER
	rsync -avh illarion@213.32.26.74:~/Projects/master_degree/frames_paper_CYLINDER/ frames_paper_CYLINDER
# 	rsync -avh illarion@213.32.26.74:~/Projects/master_degree/frames_paper_SQUARE/ frames_paper_SQUARE
# 	rsync -avh illarion@213.32.26.74:~/Projects/master_degree/frames_paper_PLATE/ frames_paper_PLATE

download_videos:
	rsync -avh illarion@213.32.26.74:~/Projects/master_degree/videos/ videos

download_code:
	rsync -avh \
	    --include=.dockerignore \
	    --exclude=".*" \
	    --exclude=.ipynb_checkpoints \
	    --exclude=.git \
	    --exclude=.venv \
	    --exclude=.pytest_cache \
	    --exclude=__pycache__ \
	    --exclude=logs \
	    --exclude=htmlcov \
	    --exclude=frames \
	    --exclude=videos \
	    --exclude=*.sublime-project \
	    --exclude=*.sublime-workspace \
	    illarion@213.32.26.74:~/Projects/master_degree/ .

download_all:
	rsync -avh \
	    --include=.dockerignore \
	    --exclude=".*" \
	    --exclude=.ipynb_checkpoints \
	    --exclude=.git \
	    --exclude=.venv \
	    --exclude=.pytest_cache \
	    --exclude=__pycache__ \
	    --exclude=logs \
	    --exclude=htmlcov \
	    --exclude=*.sublime-project \
	    --exclude=*.sublime-workspace \
	    illarion@213.32.26.74:~/Projects/master_degree/ .
