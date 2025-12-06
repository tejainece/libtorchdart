all: download

download_20b:
	mkdir -p .
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/tokenizer.json
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/tokenizer_config.json
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/special_tokens_map.json
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/model.safetensors.index.json
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00000-of-00002.safetensors
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00001-of-00002.safetensors
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00002-of-00002.safetensors || true
	wget -nc https://huggingface.co/openai/gpt-oss-20b/resolve/main/chat_template.jinja

clean:
	rm -f config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors.index.json *.safetensors chat_template.jinja
