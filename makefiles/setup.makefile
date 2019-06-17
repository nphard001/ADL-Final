# setup.makefile: things you won't run them twice

# for "fashion-retrieval":
# just download original git repo, instead of fork
setup_duplicate:
	git clone https://github.com/nphard001/fashion_retrieval
	rm -rf fashion_retrieval/.git/
	tree -d .
	@echo "NOTES: original .git files are deleted"

setup_touch_init:
	touch fashion_retrieval/__init__.py
	touch fashion_retrieval/captioner/__init__.py
	touch fashion_retrieval/captioner/neuraltalk2/__init__.py

setup_symlink_chatbot:
	-ln -s ../nphard001/ chatbot/nphard001
	tree -d chatbot

install_chatbot:
	$(PIP) install line-bot-sdk
	$(PIP) install Pillow
