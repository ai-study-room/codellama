docker run -d \
--device=/dev/davinci7 \
--device=/dev/davinci6 \
--device=/dev/davinci5 \
--device=/dev/davinci4 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci1 \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/dcmi:/usr/local/dcmi:ro \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
-v /usr/local/Ascend/nnae:/usr/local/Ascend/nnae:ro \
-v /usr/local/Ascend/nnrt:/usr/local/Ascend/nnrt:ro \
-v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
-v /usr/local/Ascend/version.info:/usr/local/Ascend/version.info:ro \
-v /data/disk1/lizhonghua/projects/codellama/CodeLlama-34b-Instruct:/codellama/CodeLlama-34b-Instruct \
-v /data/disk1/lizhonghua/projects/codellama/CodeLlama-70b-Instruct:/codellama/CodeLlama-70b-Instruct \
-v /data/disk1/lizhonghua/projects/codellama/CodeLlama-34b-Instruct-2:/codellama/CodeLlama-34b-Instruct-2 \
-v /data/disk1/lizhonghua/projects/codellama/CodeLlama-7b-Instruct:/codellama/CodeLlama-7b-Instruct \
-e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common \
-it codellama:v1 /bin/bash