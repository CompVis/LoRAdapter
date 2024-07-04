
paths=(sd15-depth-128-only-res/struct sd15-hed-128-only-res/struct sd15-style-cross-160-h/style sdxl_b-lora_256/style)

for p in "${paths[@]}"
do
    mkdir -p checkpoints/$p
    wget -O checkpoints/$p/lora-checkpoint.pt https://huggingface.co/kliyer/LoRAdapter/resolve/main/$p/lora-checkpoint.pt?download=true
    wget -O checkpoints/$p/mapper-checkpoint.pt https://huggingface.co/kliyer/LoRAdapter/resolve/main/$p/mapper-checkpoint.pt?download=true
done