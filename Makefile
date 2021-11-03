IMAGE := informer
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

dataset:
	mkdir -p data/ETT && \
		wget -O data/ETT/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv && \
		wget -O data/ETT/ETTh2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv && \
		wget -O data/ETT/ETTm1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv && \
		wget -O data/ETT/ETTm2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv && \
		wget -O data/ETT/ECL.csv "https://drive.google.com/uc?export=download&id=1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP" && \
		wget -O data/ETT/WTH.csv "https://drive.google.com/uc?export=download&id=1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG"

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

bash_docker:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

.require-module:
ifndef module
	$(error module is required)
endif
