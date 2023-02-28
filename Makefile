ROOT_DIR = $(PWD)
NOTEBOOKS_DIR = $(ROOT_DIR)/notebooks
JSC_DIR = $(NOTEBOOKS_DIR)/jsc
E4_DIR = $(NOTEBOOKS_DIR)/e4

JSC_USER = ${MANTIK_UNICORE_USERNAME}
JSC_PROJECT_DIR = /p/project/deepacf/maelstrom/$(JSC_USER)
JSC_SSH = $(JSC_USER)@judac.fz-juelich.de
JSC_SSH_PRIVATE_KEY_FILE = -e "ssh -i $(HOME)/.ssh/jsc"

E4_USER = ${E4_USERNAME}
E4_IP = ${E4_SERVER_IP}
E4_SSH = $(E4_USER)@$(E4_IP)
E4_SSH_PRIVATE_KEY_FILE = -e "ssh -i $(HOME)/.ssh/e4"

IMAGE_NAME = a6
VISSL_IMAGE_NAME = vissl

SSH_COPY_COMMAND = rsync -Pvra --progress

install:
	poetry install

build-python:
	# Remove old build
	rm -rf dist/
	poetry build -f wheel

build-docker: build-python
	sudo docker build --no-cache -t $(IMAGE_NAME):latest -f docker/a6.Dockerfile .

build-apptainer: build-python
	sudo apptainer build --force mlflow/a6/$(IMAGE_NAME).sif apptainer/a6.def

build: build-docker build-apptainer

upload:
	$(SSH_COPY_COMMAND) $(JSC_SSH_PRIVATE_KEY_FILE) \
		mlflow/a6/$(IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/$(IMAGE_NAME).sif

deploy: build upload

build-jsc-kernel: build-python
	sudo apptainer build --force \
		$(JSC_DIR)/jupyter-kernel.sif \
		$(JSC_DIR)/jupyter_kernel_recipe.def

build-e4-kernel: build-python
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
	$(SSH_COPY_COMMAND) $(JSC_SSH_PRIVATE_KEY_FILE) \
		$(JSC_DIR)/jupyter-kernel.sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/jupyter-kernel.sif

	# Create kernel.json file
	$(eval KERNEL_FILE := $(JSC_DIR)/kernel.json)
	echo "$${JSC_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	$(eval KERNEL_PATH="/p/home/jusers/$(JSC_USER)/juwels/.local/share/jupyter/kernels/a6/")
	ssh $(JSC_SSH_PRIVATE_KEY_FILE) $(JSC_SSH) "mkdir -p $(KERNEL_PATH)"
	$(SSH_COPY_COMMAND) $(JSC_SSH_PRIVATE_KEY_FILE) $(KERNEL_FILE) $(JSC_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-jsc-kernel: build-jsc-kernel upload-jsc-kernel

define E4_KERNEL_JSON
{
 "argv": [
   "singularity",
   "exec",
   "--cleanenv",
   "-B /data:/data",
   "/home/$(E4_USER)/jupyter-kernel.sif",
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
	$(SSH_COPY_COMMAND) $(E4_SSH_PRIVATE_KEY_FILE) \
		$(E4_DIR)/jupyter-kernel.sif \
		$(E4_SSH):/home/${E4_USER}/jupyter-kernel.sif

	# Create kernel.json file
	$(eval KERNEL_FILE=$(E4_DIR)/kernel.json)
	echo "$${E4_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	$(eval KERNEL_PATH="/home/${E4_USER}/.local/share/jupyter/kernels/a6")
	ssh $(E4_SSH_PRIVATE_KEY_FILE) $(E4_SSH) "mkdir -p $(KERNEL_PATH)"
	$(SSH_COPY_COMMAND) $(E4_SSH_PRIVATE_KEY_FILE) $(KERNEL_FILE) $(E4_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-e4-kernel: build-e4-kernel upload-e4-kernel

# VISSL-related steps
build-vissl:
	sudo apptainer build --force mlflow/deepclusterv2/$(VISSL_IMAGE_NAME).sif apptainer/vissl.def

upload-vissl:
	$(SSH_COPY_COMMAND) $(JSC_SSH_PRIVATE_KEY_FILE) \
		mlflow/deepclusterv2/$(VISSL_IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/$(VISSL_IMAGE_NAME).sif

deploy-vissl: build-vissl upload-vissl
