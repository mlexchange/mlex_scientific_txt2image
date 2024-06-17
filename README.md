# scientific_txt2image
A diffusion model pipeline to generate realistic scientific dataset based on iterative human labels/feedback.

## Install
The full pipeline contains two parts, a diffusers generator and an enssemble classification process. Due to version conflicts (`xformers` are needed for finetuning and inferencing with diffusers on GPU), we install them in two different environments.

Install diffusers enviroment `pip install -r requirements-diffusers.txt`  
Install classification environment `pip install -r requirements-classification.txt`

## How to finetune diffusion model and classifiers?
1. The diffusers fine-tuning and inferencing based on scientific domain dataset (see `/data/metadata.jsonl` as an example). Go to `/src`, and use the command (example) below to train:
    ```
    accelerate launch train_text_to_image_lora.py \  
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \  
    --train_data_dir="als_data" \  
    --resolution=512 --center_crop --random_flip \  
    --train_batch_size=32 \  
    --num_train_epochs=100 \  
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \  
    --enable_xformers_memory_efficient_attention \  
    --seed=42 \  
    --output_dir="output" \  
    --validation_prompt="GISAXS data showing peaks"  
    ```  
2. Once the diffusers model is trained, weights are saved. Then use `python3 /src/generator.py` (need to modify `generator.py` for each prompt) to generate images for a given prompt.
3. Label the generated images.
4. Use `src/classifier/classification.ipynb` to train an assortment of computer vision models (vision transformers etc.) to classify the generated images and save their weights.
5. Full pipeline: generate realistic from diffusers and classifiers `python3 /src/inferece.py` (need to modify `generator.py` for each prompt).
6. An interactive interface to generate realistic scientific images from a prompt: `txt2image_widgets.ipynb` (serial) and `txt2image_widgets` (parallel).

