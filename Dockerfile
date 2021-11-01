FROM continuumio/miniconda3:4.7.12

ADD ./environment.yml ./environment.yml

RUN conda install -n base -c conda-forge mamba && \
    mamba env update -n base -f ./environment.yml && \
    conda clean -afy

