ROOT_DIR = $(PWD)
NOTEBOOKS_DIR = $(ROOT_DIR)/notebooks
JSC_DIR = $(NOTEBOOKS_DIR)/jsc
E4_DIR = $(NOTEBOOKS_DIR)/e4

JSC_USER = $(MANTIK_UNICORE_USERNAME)
JSC_SSH = $(JSC_USER)@judac.fz-juelich.de
JSC_SSH_OPTIONS = -e "ssh -i $(JSC_SSH_PRIVATE_KEY_FILE)"

E4_SSH = $(E4_USERNAME)@$(E4_SERVER_IP)
E4_SSH_OPTIONS = -e "ssh -i $(E4_SSH_PRIVATE_KEY_FILE)"

IMAGE_NAME = a6

SSH_COPY_COMMAND = rsync -Pvra --progress

install:
	poetry install
	poetry run pip install -r requirements-cpu.txt

build-docker:
	sudo docker build -t $(IMAGE_NAME):latest -f docker/a6.Dockerfile .

build-apptainer: build-docker
	sudo apptainer build --force mlflow/$(IMAGE_NAME).sif apptainer/a6.def

build: build-docker build-apptainer

upload:
	$(SSH_COPY_COMMAND) $(JSC_SSH_OPTIONS) \
		mlflow/$(IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/$(IMAGE_NAME).sif

upload-e4:
	$(SSH_COPY_COMMAND) $(E4_SSH_OPTIONS) \
		mlflow/$(IMAGE_NAME).sif \
		$(E4_SSH):$(E4_PROJECT_DIR)/$(IMAGE_NAME).sif

build-docker-cuda:
	sudo docker build -t $(IMAGE_NAME)-cuda:latest -f docker/a6-cuda.Dockerfile .

build-apptainer-cuda: build-docker-cuda
	sudo apptainer build --force mlflow/$(IMAGE_NAME)-cuda.sif apptainer/a6-cuda.def

build-cuda: build-docker-cuda build-apptainer-cuda

upload-cuda:
	$(SSH_COPY_COMMAND) $(JSC_SSH_OPTIONS) \
		mlflow/$(IMAGE_NAME)-cuda.sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/$(IMAGE_NAME)-cuda.sif

upload-e4-cuda:
	$(SSH_COPY_COMMAND) $(E4_SSH_OPTIONS) \
		mlflow/$(IMAGE_NAME)-cuda.sif \
		$(E4_SSH):$(E4_PROJECT_DIR)/$(IMAGE_NAME)-cuda.sif

deploy-cuda: build-cuda upload-cuda-cuda

deploy-e4-cuda: build-cuda upload-e4-cuda

build-jsc-kernel:
	sudo apptainer build --force \
		$(JSC_DIR)/jupyter-kernel.sif \
		$(JSC_DIR)/jupyter_kernel_recipe.def

build-e4-kernel:
	sudo apptainer build --force \
		$(E4_DIR)/jupyter-kernel.sif \
		$(E4_DIR)/jupyter_kernel_recipe.def

define JSC_KERNEL_JSON
{
 "argv": [
   "singularity",
   "exec",
   "--cleanenv",
   "-B /usr/:/host/usr/, /etc/slurm:/etc/slurm, /usr/lib64:/host/usr/lib64, /opt/parastation:/opt/parastation, /usr/lib64/slurm:/usr/lib64/slurm, /usr/share/lua:/usr/share/lua, /usr/lib64/lua:/usr/lib64/lua, /opt/jsc:/opt/jsc, /var/run/munge:/var/run/munge",
   "$(JSC_PROJECT_DIR)/jupyter-kernel.sif",
   "python",
   "-m",
   "ipykernel",
   "-f",
   "{connection_file}"
 ],
 "language": "python",
 "display_name": "a6"
}
endef

export JSC_KERNEL_JSON

upload-jsc-kernel:
	# Copy kernel image file
	$(SSH_COPY_COMMAND) $(JSC_SSH_OPTIONS) \
		$(JSC_DIR)/jupyter-kernel.sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/jupyter-kernel.sif

	# Create kernel.json file
	$(eval KERNEL_FILE := $(JSC_DIR)/kernel.json)
	echo "$${JSC_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	$(eval KERNEL_PATH="/p/home/jusers/$(JSC_USER)/juwels/.local/share/jupyter/kernels/a6/")
	ssh $(JSC_SSH_OPTIONS) $(JSC_SSH) "mkdir -p $(KERNEL_PATH)"
	$(SSH_COPY_COMMAND) $(JSC_SSH_OPTIONS) $(KERNEL_FILE) $(JSC_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-jsc-kernel: build-jsc-kernel upload-jsc-kernel

define E4_KERNEL_JSON
{
 "argv": [
   "singularity",
   "exec",
   "--cleanenv",
   "-B /data:/data",
   "/home/$(E4_USERNAME)/jupyter-kernel.sif",
   "python",
   "-m",
   "ipykernel",
   "-f",
   "{connection_file}"
 ],
 "language": "python",
 "display_name": "a6"
}
endef

export E4_KERNEL_JSON

upload-e4-kernel:
	# Copy kernel image file
	$(SSH_COPY_COMMAND) $(E4_SSH_OPTIONS) \
		$(E4_DIR)/jupyter-kernel.sif \
		$(E4_SSH):/data/maelstrom/a6/jupyter-kernel.sif

	# Create kernel.json file
	$(eval KERNEL_FILE=$(E4_DIR)/kernel.json)
	echo "$${E4_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	$(eval KERNEL_PATH="/home/$(E4_USERNAME)/.local/share/jupyter/kernels/a6")
	ssh $(E4_SSH_OPTIONS) $(E4_SSH) "mkdir -p $(KERNEL_PATH)"
	$(SSH_COPY_COMMAND) $(E4_SSH_OPTIONS) $(KERNEL_FILE) $(E4_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-e4-kernel: build-e4-kernel upload-e4-kernel
