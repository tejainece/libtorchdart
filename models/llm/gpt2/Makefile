all: download

download:
	mkdir -p .
	wget -nc https://huggingface.co/gpt2/resolve/main/config.json
	wget -nc https://huggingface.co/gpt2/resolve/main/vocab.json
	wget -nc https://huggingface.co/gpt2/resolve/main/merges.txt
	wget -nc https://huggingface.co/gpt2/resolve/main/model.safetensors

clean:
	rm -f config.json vocab.json merges.txt model.safetensors
