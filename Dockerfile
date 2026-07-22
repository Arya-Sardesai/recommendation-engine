# Pipeline environment for the recommendation engine (corpus build, tagging,
# embedding, evaluation). CPU image: every stage runs correctly on CPU; the
# embedding stages are simply much faster on a host GPU, which is how they are
# run in practice (see requirements.txt for the CUDA note).
#
# The serving app has its own Dockerfile in the hf-space repo (deployed on
# Hugging Face Spaces); this image is for reproducing the data pipeline.
#
# Build:
#   docker build -t rec-engine-pipeline .
#
# Run (mount the repo so data/ and outputs persist on the host — raw dumps and
# built artifacts are gitignored and live outside the image on purpose):
#   docker run -it --rm -v "$(pwd)":/work rec-engine-pipeline
#
# Then inside the container, e.g.:
#   make movies-corpus
#   make smoke

FROM python:3.11-slim

# make for the Makefile; git for pip VCS installs if ever needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install dependencies first so Docker layer caching survives code edits
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the repo (a mounted volume at /work overrides this for live work;
# the COPY makes the image self-contained if run without a mount)
COPY . .

CMD ["bash"]
